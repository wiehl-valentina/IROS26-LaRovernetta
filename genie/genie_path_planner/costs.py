from __future__ import annotations

import numpy as np


def path_cost_with_footprint(
    raw_topdown_score: np.ndarray,
    candidate_path: np.ndarray,
    alpha: float = 0.5,
    footprint_px: int = 20,
) -> float:
    """Sum average footprint costs along one path."""
    r_half = max(1, int(footprint_px)) // 2
    h, w = raw_topdown_score.shape[:2]
    total_cost = 0.0
    for row, col in candidate_path:
        r_coord = int(round(float(row)))
        c_coord = int(round(float(col)))
        r1 = max(0, r_coord - r_half)
        r2 = min(h, r_coord + r_half + 1)
        c1 = max(0, c_coord - r_half)
        c2 = min(w, c_coord + r_half + 1)
        if r1 >= r2 or c1 >= c2:
            continue
        region = raw_topdown_score[r1:r2, c1:c2]
        total_cost += float(np.exp(float(alpha) * region).mean())
    return total_cost


def goal_term(
    candidate_path: np.ndarray,
    goal_rc: tuple[int, int] | None,
    goal_weight: float,
    grid_size: int,
) -> float:
    """Penalizacion por terminar lejos de la meta.

    El costo de transitabilidad solo mide por donde se puede pisar, asi que en
    terreno uniformemente libre TODOS los caminos empatan y la eleccion queda
    arbitraria: el robot elige una direccion cualquiera, gira hacia ella,
    replanifica, vuelve a elegir otra relativa a su nueva orientacion, y
    termina describiendo circulos sin acercarse nunca al checkpoint.

    Este termino es lo que hace que "ir hacia la meta" sea algo que el planner
    optimiza y no un efecto secundario del clustering. Se escala por la
    cantidad de puntos del camino para que quede en las mismas unidades que
    path_cost_with_footprint (que SUMA sobre los puntos), de modo que
    goal_weight se lea como "costo por punto equivalente".

    goal_weight = 0 lo desactiva y reproduce exactamente el comportamiento
    anterior.
    """
    if goal_rc is None or float(goal_weight) <= 0.0:
        return 0.0
    path = np.asarray(candidate_path, dtype=np.float64)
    if path.size == 0:
        return 0.0
    end = path[-1]
    distance = float(np.hypot(end[0] - float(goal_rc[0]), end[1] - float(goal_rc[1])))
    return float(goal_weight) * (distance / float(max(1, grid_size))) * len(path)


def compute_paths_costs(
    candidate_paths: list[np.ndarray],
    raw_topdown_score: np.ndarray,
    alpha: float,
    footprint_px: int = 20,
    goal_rc: tuple[int, int] | None = None,
    goal_weight: float = 0.0,
    grid_size: int = 240,
) -> list[tuple[float, np.ndarray]]:
    return [
        (
            path_cost_with_footprint(
                raw_topdown_score,
                path,
                alpha=float(alpha),
                footprint_px=int(footprint_px),
            )
            + goal_term(path, goal_rc, goal_weight, grid_size),
            path,
        )
        for path in candidate_paths
    ]


def pick_final_path(
    paths_with_cost: list[tuple[float, np.ndarray]],
    best_k: int,
    num_samples: int,
    cost_map: np.ndarray,
    alpha: float,
    footprint_px: int,
    goal_rc: tuple[int, int] | None = None,
    goal_weight: float = 0.0,
    grid_size: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a final path from the best candidates.

    Lower cost is better, so the mix weights each candidate by inverse cost:
    the best path in the top-k dominates the average instead of the worst one
    (using the raw cost as the weight would do the opposite, since higher
    cost would get more weight).
    """
    if not paths_with_cost:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    paths_sorted = sorted(paths_with_cost, key=lambda x: x[0])
    top = paths_sorted[: int(max(1, best_k))]
    final_rows = np.zeros(int(num_samples), dtype=np.float32)
    final_cols = np.zeros(int(num_samples), dtype=np.float32)

    costs = np.array([p[0] for p in top], dtype=np.float32)
    weights = 1.0 / (costs + 1e-6)
    weights = weights / np.sum(weights)

    for i in range(int(num_samples)):
        for idx, (_cost, path) in enumerate(top):
            final_rows[i] += float(weights[idx]) * float(path[i][0])
            final_cols[i] += float(weights[idx]) * float(path[i][1])

    weighted_path = np.stack([final_rows, final_cols], axis=1)
    # El mismo termino de meta que compute_paths_costs: si no, se compararia
    # un costo puro de transitabilidad contra costos que ya lo incluyen, y el
    # camino promediado ganaria siempre.
    weighted_cost = path_cost_with_footprint(
        cost_map,
        weighted_path,
        alpha=float(alpha),
        footprint_px=int(footprint_px),
    ) + goal_term(weighted_path, goal_rc, goal_weight, grid_size)
    worst_top_cost = max(p[0] for p in top)
    if weighted_cost > worst_top_cost:
        best_path = np.asarray(top[0][1], dtype=np.float32)
        return best_path[:, 0].astype(np.float32), best_path[:, 1].astype(np.float32)
    return final_rows, final_cols


def _self_test() -> None:
    # Tres caminos de un solo punto, en fila sobre el eje x: costo 0.1 (mejor,
    # x=0), 0.5 (x=10) y 0.9 (peor, x=20). La mezcla tiene que quedar mucho
    # mas cerca del mejor camino que del peor.
    low = np.array([[0.0, 0.0]], dtype=np.float32)
    mid = np.array([[10.0, 0.0]], dtype=np.float32)
    high = np.array([[20.0, 0.0]], dtype=np.float32)
    paths_with_cost = [(0.9, high), (0.1, low), (0.5, mid)]
    # Costo muy bajo en todo el mapa: el promedio ponderado no dispara el
    # fallback a "mejor camino solo" (weighted_cost > worst_top_cost), asi
    # que el assert de abajo prueba de verdad los pesos de la mezcla.
    cost_map = np.full((30, 30), -50.0, dtype=np.float32)

    rows, cols = pick_final_path(
        paths_with_cost, best_k=3, num_samples=1, cost_map=cost_map,
        alpha=0.5, footprint_px=4,
    )
    x = float(rows[0])
    print(f"mezcla de x=[0,10,20] con costos [0.1,0.5,0.9]: x={x:.2f} (esperado < 5)")
    assert x < 5.0, "el peor camino sigue pesando de mas en la mezcla"

    # Termino de meta: dos caminos IGUAL de transitables (mapa uniforme), uno
    # terminando a la izquierda y otro a la derecha. Sin el termino la eleccion
    # es arbitraria; con el, tiene que ganar el que termina cerca de la meta.
    plano = np.zeros((240, 240), dtype=np.float32)
    izq = np.stack([np.linspace(239, 40, 20), np.linspace(120, 40, 20)], axis=1).astype(np.float32)
    der = np.stack([np.linspace(239, 40, 20), np.linspace(120, 200, 20)], axis=1).astype(np.float32)
    for meta_rc, esperado, lbl in [((40, 40), izq, "izquierda"), ((40, 200), der, "derecha")]:
        costeados = compute_paths_costs([izq, der], plano, alpha=1.0, footprint_px=4,
                                        goal_rc=meta_rc, goal_weight=1.0, grid_size=240)
        elegido = min(costeados, key=lambda x: x[0])[1]
        print(f"  meta a la {lbl:10} -> elige el camino que va a la "
              f"{'izquierda' if elegido is izq else 'derecha'}")
        assert elegido is esperado, "el costo no esta siguiendo la meta"

    sin_termino = compute_paths_costs([izq, der], plano, alpha=1.0, footprint_px=4)
    con_cero = compute_paths_costs([izq, der], plano, alpha=1.0, footprint_px=4,
                                   goal_rc=(40, 40), goal_weight=0.0, grid_size=240)
    assert [c for c, _ in sin_termino] == [c for c, _ in con_cero], \
        "goal_weight=0 tiene que reproducir exactamente el comportamiento anterior"
    print("  goal_weight=0 reproduce el costo original")

    print("Todos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
