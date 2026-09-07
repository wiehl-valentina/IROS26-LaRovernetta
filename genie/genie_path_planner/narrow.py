"""Corredores angostos: filtrado por prefijo valido + centrado en el verde.

Destino: `genie/genie_path_planner/narrow.py` (archivo nuevo). Se engancha
desde `planner.py` con dos cambios de 3 lineas cada uno (ver el .md).

DOS PROBLEMAS DEL PLANNER ACTUAL EN PASTO ANGOSTO
--------------------------------------------------

**(1) El filtro es todo o nada.**
`path_selection.has_majority_of_high_cost_points()` recorre los primeros
`number_of_points_to_filter` (60 de 100) puntos del camino y devuelve True --
o sea, DESCARTA el camino entero -- apenas UN punto tiene suficientes celdas
caras en su footprint. Con `threshold_points_ratio: 0.02` y
`footprint_px: 12`, "suficientes" son ceil(0.02 * 13*13) = 4 celdas de 169.

Consecuencia directa de lo que pediste: un camino que va 1.5 m por el medio
del verde y recien al final roza pasto se tira a la basura igual que uno que
choca contra una pared a 10 cm. Si TODOS los caminos del banco tienen algo de
rojo al final -- que es exactamente lo que pasa en un corredor de pasto,
porque el corredor dobla y el banco es recto-ish -- `filtered_paths` queda
vacio, `plan.final_path_xy_m` viene None, el bridge cuenta un plan vacio y a
los 3 seguidos entra en recuperacion. Ese es el ciclo que ves.

Aca el filtro pasa a medir CUANTO camino limpio hay antes del primer punto
malo, y se queda con los caminos que tengan al menos `min_prefix_m` metros
utiles. El camino se devuelve ENTERO (mismo largo, `pick_final_path` mezcla
por indice y necesita que todos midan igual); quien recorta la cola es el
bridge, con `trim_path_xy()`, justo antes de seguirlo. Se planifica cada
`replan_every_m` = 1 m, asi que recortar a ~1.2 m de prefijo util no cuesta
nada de avance.

Ademas hay relajacion progresiva: si con `min_prefix_m` no sobrevive ninguno,
se reintenta con el 60%, y si tampoco, se queda con los caminos de prefijo
mas largo que haya. **Nunca devuelve lista vacia si algun camino avanza aunque
sea un poco.** El bridge tiene su propio freno (front_guard), asi que el
planner no necesita ser el que dice "no hay nada": esa doble negacion es la
que deja al rover plantado.

**(2) Nada empuja al camino al CENTRO del corredor.**
El costo es la media del footprint sobre celdas transitables. En un corredor
de 0.5 m con footprint de 0.30 m, pasar pegado al borde cuesta casi lo mismo
que pasar por el medio (unas pocas celdas de pasto entran al footprint). Con
`smooth_kernel: 3` la diferencia es todavia mas chata. Resultado: el rover
rifa el borde, el error de rumbo lo saca del corredor, y ahi si choca.

`apply_centering_bonus()` suma al mapa de costo una penalizacion por
ASIMETRIA lateral: para cada celda libre mide cuanto hay libre a su izquierda
y a su derecha en la misma fila, y penaliza |dL - dR| / (dL + dR). Una celda
en el eje del corredor no paga nada; una pegada al pasto paga el maximo.
Importante: penaliza la asimetria, NO la angostura -- un corredor de 0.35 m
no se penaliza por ser angosto, solo se pide ir por su eje. Eso es
literalmente "centrarse en lo verde aunque tenga muy poco ancho".

La penalizacion se aplica SOLO a celdas que ya estaban por debajo de
`threshold_cost` y se topea en `threshold_cost - eps`, para que centrar nunca
convierta una celda buena en una celda vetada por el filtro.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lateral_free_runs",
    "apply_centering_bonus",
    "first_bad_point_index",
    "filter_paths_by_valid_prefix",
    "trim_path_xy",
]


# ------------------------------------------------------- centrado en el verde

def lateral_free_runs(bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada celda, distancia en px a la celda mala mas cercana a su
    izquierda y a su derecha, en la MISMA fila.

    Vectorizado con el truco del acumulado de indices: O(H*W), sin bucles
    de Python. Si no hay celda mala de ese lado, la distancia es hasta el
    borde (asi el terreno abierto no queda con asimetria infinita).
    """
    bad = np.asarray(bad, dtype=bool)
    h, w = bad.shape
    idx = np.arange(w)[None, :]

    ultimo = np.where(bad, idx, -1)
    ultimo = np.maximum.accumulate(ultimo, axis=1)
    izq = (idx - ultimo).astype(np.float32)      # >=1; w si nunca hubo mala

    proximo = np.where(bad, idx, w)
    proximo = np.minimum.accumulate(proximo[:, ::-1], axis=1)[:, ::-1]
    der = (proximo - idx).astype(np.float32)

    return izq, der


def apply_centering_bonus(cost: np.ndarray, threshold_cost: float,
                          center_weight: float,
                          max_useful_px: int = 40) -> np.ndarray:
    """Suma al mapa de costo una penalizacion por estar descentrado en el
    corredor. Devuelve un mapa NUEVO (no toca el original).

    `max_useful_px` topea las distancias antes de calcular la asimetria: sin
    esto, en campo abierto con una pared lejana a la izquierda el termino
    empujaria al robot hacia el centro geometrico del BEV, que no significa
    nada. Con el tope, todo corredor mas ancho que ~2*max_useful_px queda con
    penalizacion cero y el comportamiento en abierto no cambia.
    """
    cost = np.asarray(cost, dtype=np.float32)
    if float(center_weight) <= 0.0:
        return cost
    bad = cost >= float(threshold_cost)
    if not np.any(bad):
        return cost

    izq, der = lateral_free_runs(bad)
    # "Estar en un corredor" = tener pasto/pared a los DOS lados dentro de
    # max_useful_px. Si de un lado no hay nada cerca, no es un corredor: es
    # terreno abierto con algo al costado, y ahi el costo del footprint ya
    # alcanza. Sin este gate, una pared lejana a la izquierda empujaria al
    # robot hacia el centro del BEV en campo abierto, que no significa nada.
    en_corredor = (izq < float(max_useful_px)) & (der < float(max_useful_px))
    total = izq + der
    asimetria = np.abs(izq - der) / np.maximum(total, 1e-6)   # 0 = centrado, ~1 = pegado
    pen = np.where(en_corredor, float(center_weight) * asimetria, 0.0).astype(np.float32)

    tope = float(threshold_cost) - 1e-3
    salida = cost.copy()
    libres = ~bad
    salida[libres] = np.minimum(cost[libres] + pen[libres], tope)
    return salida


# ------------------------------------------- filtrado por prefijo valido

def first_bad_point_index(cost: np.ndarray, path: np.ndarray, num_points: int,
                          footprint_px: int, threshold_points_ratio: float,
                          threshold_cost: float) -> int:
    """Indice del primer punto del camino cuyo footprint pisa demasiado costo
    alto. Devuelve `min(num_points, len(path))` si ninguno lo hace.

    Misma cuenta punto a punto que `has_majority_of_high_cost_points`, pero
    devolviendo DONDE en vez de un bool.
    """
    r_half = max(1, int(footprint_px)) // 2
    h, w = cost.shape[:2]
    n = int(min(int(num_points), len(path)))
    for i in range(n):
        row, col = path[i]
        r = int(round(float(row)))
        c = int(round(float(col)))
        r1, r2 = max(0, r - r_half), min(h, r + r_half + 1)
        c1, c2 = max(0, c - r_half), min(w, c + r_half + 1)
        region = cost[r1:r2, c1:c2]
        if region.size == 0:
            return i
        lim = int(np.ceil(float(threshold_points_ratio) * region.size))
        if int(np.sum(region >= float(threshold_cost))) >= lim:
            return i
    return n


def _prefix_len_m(path: np.ndarray, k: int, m_per_px) -> float:
    """Largo de arco, en metros, de los primeros k puntos del camino.

    `m_per_px` puede ser un escalar o una tupla `(m_por_px_fila,
    m_por_px_columna)`. La tupla NO es un lujo: el BEV de este repo no es
    cuadrado (con forward_range 2.0 y side_range 2.0 sale ~67x134) y
    `plan_on_bev` lo reescala a un grid cuadrado de grid_size x grid_size, asi
    que un pixel del planner mide distinto a lo largo que a lo ancho -- con la
    config actual, casi el doble. Medir el arco con un solo escalar da un
    error de hasta 2x en el largo del prefijo, que es justo el numero contra
    el que se compara `min_prefix_m`.
    """
    if k < 2:
        return 0.0
    p = np.asarray(path[:k], dtype=np.float64)
    d = np.diff(p, axis=0)
    if np.isscalar(m_per_px):
        return float(np.sum(np.linalg.norm(d, axis=1))) * float(m_per_px)
    m_fila, m_col = float(m_per_px[0]), float(m_per_px[1])
    d = d * np.array([m_fila, m_col])
    return float(np.sum(np.linalg.norm(d, axis=1)))


def filter_paths_by_valid_prefix(
    candidate_paths, cost: np.ndarray, num_points: int, footprint_px: int,
    threshold_points_ratio: float, threshold_cost: float,
    m_per_px, min_prefix_m: float = 1.0,
    relax_ratio: float = 0.6, hard_min_prefix_m: float = 0.25,
    keep_at_least: int = 6,
) -> tuple[list, list[float], dict]:
    """Devuelve `(caminos, prefijos_m, meta)`.

    `prefijos_m[i]` es cuantos metros del camino i son limpios antes del
    primer punto malo. `m_per_px` es un escalar o `(m_por_px_fila,
    m_por_px_columna)` -- ver `_prefix_len_m`.
    """
    caminos = list(candidate_paths)
    prefijos: list[float] = []
    for path in caminos:
        k = first_bad_point_index(cost, path, num_points, footprint_px,
                                  threshold_points_ratio, threshold_cost)
        prefijos.append(_prefix_len_m(path, k, m_per_px))
    pref = np.asarray(prefijos, dtype=np.float64)

    for nivel, umbral in (("estricto", float(min_prefix_m)),
                          ("relajado", float(min_prefix_m) * float(relax_ratio)),
                          ("minimo", float(hard_min_prefix_m))):
        sel = np.nonzero(pref >= umbral)[0]
        if sel.size:
            return ([caminos[i] for i in sel], [float(pref[i]) for i in sel],
                    {"prefix_tier": nivel, "prefix_threshold_m": round(umbral, 3),
                     "prefix_best_m": round(float(pref.max()), 3),
                     "kept": int(sel.size), "total": len(caminos)})

    # Ni siquiera hard_min: quedarse con los mejores igual. Que el planner
    # devuelva "nada" es peor que devolver el menos malo -- el freno frontal
    # es responsabilidad de front_guard, no del planner.
    orden = np.argsort(-pref)[:int(keep_at_least)]
    orden = [i for i in orden if pref[i] > 0.0]
    return ([caminos[i] for i in orden], [float(pref[i]) for i in orden],
            {"prefix_tier": "ultimo_recurso", "prefix_threshold_m": 0.0,
             "prefix_best_m": round(float(pref.max()) if pref.size else 0.0, 3),
             "kept": len(orden), "total": len(caminos)})


# --------------------------------------------------------- recorte en el bridge

def trim_path_xy(path_xy: np.ndarray, bev: np.ndarray, resolution_m: float,
                 half_width_m: float = 0.22, traversable_thresh: float = 0.30,
                 min_free_ratio: float = 0.55, min_keep_m: float = 0.5
                 ) -> tuple[np.ndarray, float]:
    """Corta el camino (en metros, marco robot: x=derecha, y=adelante) en el
    primer punto donde el corredor de `half_width_m` deja de estar libre en el
    BEV fresco. Devuelve `(camino_recortado, metros_conservados)`.

    Esto es lo que hace que "verde con un poco de rojo al final" sea util: se
    sigue el tramo verde y se replanifica antes de llegar al rojo, en vez de
    tirar el camino entero o de meterse en el rojo confiando en el plan.

    Nunca recorta por debajo de `min_keep_m`: si el primer punto ya esta feo,
    el que tiene que decidir es front_guard, no esto.
    """
    p = np.asarray(path_xy, dtype=np.float64)
    if p.ndim != 2 or len(p) < 2 or bev is None or bev.size == 0:
        return p, 0.0
    h, w = bev.shape[:2]
    c_half = max(1, int(round(half_width_m / resolution_m)))

    recorrido = 0.0
    corte = len(p)
    for i in range(len(p)):
        if i > 0:
            recorrido += float(np.hypot(p[i, 0] - p[i - 1, 0], p[i, 1] - p[i - 1, 1]))
        r = h - 1 - int(round(float(p[i, 1]) / resolution_m))
        c = w // 2 + int(round(float(p[i, 0]) / resolution_m))
        if not (0 <= r < h):
            break
        c0, c1 = max(0, c - c_half), min(w, c + c_half + 1)
        if c0 >= c1:
            break
        fila = bev[r, c0:c1]
        conocidas = fila[fila >= 0.0]
        if conocidas.size and float(np.mean(conocidas > traversable_thresh)) < min_free_ratio:
            if recorrido >= min_keep_m:
                corte = i
                break
    return p[:max(2, corte)], recorrido


# ---------------------------------------------------------------- self test

def _self_test() -> None:
    # --- centrado ---------------------------------------------------------
    # Corredor vertical de 20 px de ancho en un mapa de costo 240x240:
    # adentro cuesta 0.1 (libre), afuera 0.9 (pasto).
    cost = np.full((240, 240), 0.9, dtype=np.float32)
    cost[:, 110:130] = 0.1
    centrado = apply_centering_bonus(cost, threshold_cost=0.35, center_weight=0.20)
    eje, borde = centrado[120, 120], centrado[120, 111]
    print(f"  corredor de 20 px: costo en el eje={eje:.4f}  pegado al borde={borde:.4f}")
    assert borde > eje + 0.05, "el centrado tiene que preferir el eje del corredor"
    libres = cost < 0.35
    assert centrado[libres].max() <= 0.35, "el centrado nunca puede empujar una celda buena a ser vetada"
    assert abs(float(centrado[120, 10]) - 0.9) < 1e-6, "las celdas malas no se tocan"

    abierto = np.full((240, 240), 0.1, dtype=np.float32)
    abierto[:, :5] = 0.9
    ab2 = apply_centering_bonus(abierto, 0.35, 0.20)
    print(f"  campo abierto: penalizacion maxima={float((ab2 - abierto).max()):.4f} (esperado 0)")
    assert float((ab2 - abierto).max()) < 1e-6, "en abierto el centrado no debe hacer nada"

    # --- prefijo ----------------------------------------------------------
    # Camino recto por el medio del corredor que al final (fila < 60) se mete
    # en el pasto: el filtro viejo lo tiraba entero.
    filas = np.linspace(239, 20, 100)
    cols = np.full(100, 120.0)
    recto = np.stack([filas, cols], axis=1).astype(np.float32)
    mapa = np.full((240, 240), 0.9, dtype=np.float32)
    mapa[150:, 110:130] = 0.1     # corredor limpio solo hasta la fila 150

    m_per_px = 4.0 / 240.0        # 4 m de ancho de BEV sobre 240 px
    k = first_bad_point_index(mapa, recto, 60, 12, 0.02, 0.35)
    largo = _prefix_len_m(recto, k, m_per_px)
    print(f"  camino con rojo al final: primer punto malo en i={k}, prefijo util={largo:.2f} m")
    assert 0 < k < 60 and largo > 0.8

    kept, prefs, meta = filter_paths_by_valid_prefix(
        [recto], mapa, 60, 12, 0.02, 0.35, m_per_px, min_prefix_m=0.8)
    print(f"  filtro por prefijo -> {meta['kept']}/{meta['total']} caminos ({meta['prefix_tier']})")
    assert len(kept) == 1, "un camino con 1+ m util NO se descarta por tener rojo al final"

    # Todos los caminos malos: relajacion progresiva, nunca lista vacia.
    kept2, _, meta2 = filter_paths_by_valid_prefix(
        [recto], mapa, 60, 12, 0.02, 0.35, m_per_px, min_prefix_m=5.0)
    print(f"  pidiendo 5 m (imposible) -> {meta2['kept']} caminos, nivel '{meta2['prefix_tier']}'")
    assert len(kept2) == 1, "nunca puede devolver cero si algun camino avanza algo"

    # --- recorte ----------------------------------------------------------
    bev = np.full((134, 134), 0.05, dtype=np.float32)
    media = int(0.22 / 0.03)
    bev[:, 67 - media:67 + media] = 0.8
    bev[:40, :] = 0.05                     # rojo a partir de ~2.8 m... nada que cortar
    camino = np.stack([np.zeros(20), np.linspace(0, 1.9, 20)], axis=1)
    cortado, m = trim_path_xy(camino, bev, 0.03)
    print(f"  recorte con corredor limpio: {len(cortado)}/20 puntos, {m:.2f} m")
    assert len(cortado) == 20

    bev2 = bev.copy()
    r_pared = 133 - int(1.0 / 0.03)
    bev2[:r_pared + 1, :] = 0.05           # todo rojo mas alla de 1 m
    cortado2, m2 = trim_path_xy(camino, bev2, 0.03)
    fin = float(cortado2[-1, 1])
    print(f"  recorte con rojo a 1.0 m:    {len(cortado2)}/20 puntos, termina en y={fin:.2f} m")
    assert 0.5 <= fin <= 1.15, "tiene que cortar justo antes del rojo, no antes ni despues"

    print("narrow: todos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
