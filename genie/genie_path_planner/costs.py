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


def compute_paths_costs(
    candidate_paths: list[np.ndarray],
    raw_topdown_score: np.ndarray,
    alpha: float,
    footprint_px: int = 20,
) -> list[tuple[float, np.ndarray]]:
    return [
        (
            path_cost_with_footprint(
                raw_topdown_score,
                path,
                alpha=float(alpha),
                footprint_px=int(footprint_px),
            ),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a final path from the best candidates.

    Lower cost is better. This preserves the current GeNIE planner behavior:
    average the top-k paths using inverse-cost weights, then fall back to
    the single best path if the averaged path is worse than the selected set.
    """
    if not paths_with_cost:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    paths_sorted = sorted(paths_with_cost, key=lambda x: x[0])
    top = paths_sorted[: int(max(1, best_k))]
    final_rows = np.zeros(int(num_samples), dtype=np.float32)
    final_cols = np.zeros(int(num_samples), dtype=np.float32)

    # FIX 
    """Principales cambios dentro de pick_final_path:
    Se reemplazó el cálculo de weights directo por la versión recíproca (inv = 1.0 / np.clip(...)), garantizando que los menores costos adquieran la mayor proporción de peso.
    Se incluyó el assert que comprueba en tiempo de ejecución que el índice del elemento con menor costo (np.argmin) coincida exactamente con el índice del mayor peso asignado (np.argmax)."""
    # --- Menor costo = mayor peso ---
    costs_arr = np.array([p[0] for p in top], dtype=np.float32)
    inv = 1.0 / np.clip(costs_arr, 1e-6, None)
    weights = inv / np.sum(inv)

    # Test de regresión rápido
    assert np.argmax(weights) == np.argmin(costs_arr), (
        f"El camino de menor costo (índice {np.argmin(costs_arr)}) debe tener el mayor peso (obtenido en índice {np.argmax(weights)})"
    )

    for i in range(int(num_samples)):
        for idx, (_cost, path) in enumerate(top):
            final_rows[i] += float(weights[idx]) * float(path[i][0])
            final_cols[i] += float(weights[idx]) * float(path[i][1])

    weighted_path = np.stack([final_rows, final_cols], axis=1)
    weighted_cost = path_cost_with_footprint(
        cost_map,
        weighted_path,
        alpha=float(alpha),
        footprint_px=int(footprint_px),
    )
    worst_top_cost = max(p[0] for p in top)
    if weighted_cost > worst_top_cost:
        best_path = np.asarray(top[0][1], dtype=np.float32)
        return best_path[:, 0].astype(np.float32), best_path[:, 1].astype(np.float32)
    return final_rows, final_cols