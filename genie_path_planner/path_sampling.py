from __future__ import annotations

import random

import numpy as np


def _sample_unique(population: list[tuple[int, int]], k: int, rng: random.Random) -> list[tuple[int, int]]:
    """
    Toma una muestra aleatoria de elementos únicos de una población de forma segura.

    CONCEPTOS CLAVE:
    - En Python, la función estándar `random.sample` lanza un error (ValueError) si se le pide 
      una cantidad de muestras `k` que sea mayor al tamaño total de la población disponible. 
      Esta función auxiliar previene ese fallo crítico limitando automáticamente `k` al tamaño 
      máximo de la población usando `min(int(k), len(population))`.

    Args:
        population: Lista de coordenadas o elementos disponibles para muestrear.
        k: Cantidad de muestras deseadas.
        rng: Instancia del generador de números aleatorios para garantizar reproducibilidad.

    Returns:
        Una lista de elementos (coordenadas) únicos seleccionados al azar.
    """
    if k <= 0 or not population:
        return []
    return rng.sample(population, k=min(int(k), len(population)))


def sample_goals(
    num_goals: int = 100,
    grid_size: int = 240,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """
    Muestrea metas (goals) distribuidas estratégicamente en los bordes superior, izquierdo 
    y derecho de la grilla del planificador de trayectorias.

    CONCEPTOS CLAVE:
    - Sistema de coordenadas de la grilla (BEV - Bird's Eye View): En las matrices de imágenes, 
      la Fila 0 está en la parte SUPERIOR (el horizonte o el límite "hacia adelante" del vehículo). 
      El robot inicia típicamente en la parte inferior (ej. fila 239).
    - Distribución heurística de metas: En la conducción real, lo más común es avanzar hacia adelante, 
      no girar en círculos. Por ello, la función distribuye aproximadamente el 30% de las metas en 
      los laterales (15% a la izquierda, 15% a la derecha) para simular giros o esquives, y asigna el 
      70% restante al borde superior (avance frontal).

    Args:
        num_goals: Cantidad total de metas a generar en los bordes.
        grid_size: Tamaño del lado de la grilla cuadrada (por defecto 240 píxeles/celdas).
        rng: Generador de números aleatorios (opcional).

    Returns:
        Lista de coordenadas (fila, columna) que representan las metas perimetrales a alcanzar.
    """
    rng = rng or random
    num_goals_edge = int((0.75 / 2.5) * int(num_goals))
    num_goals_top = int(num_goals) - 2 * num_goals_edge

    top_possible = [(0, col) for col in range(int(grid_size))]
    top_goals = _sample_unique(top_possible, num_goals_top, rng)

    max_side_row = max(0, min(int(grid_size) - 1, 179))
    valid_rows = list(range(0, max_side_row + 1))
    left_possible = [(row, 0) for row in valid_rows]
    right_possible = [(row, int(grid_size) - 1) for row in valid_rows]
    return top_goals + _sample_unique(left_possible, num_goals_edge, rng) + _sample_unique(
        right_possible, num_goals_edge, rng
    )


def sample_mid_for_goal(
    goal: tuple[int, int],
    robot: tuple[int, int],
    grid_size: int,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """
    Muestrea un punto de control intermedio (midpoint) para forzar y moldear una trayectoria cuadrática 
    desde el robot hasta la meta.

    CONCEPTOS CLAVE:
    - Espacio de control paramétrico: Si solo unimos el robot y la meta, obtenemos una línea recta. 
      Para generar curvas de evasión, necesitamos un punto intermedio que "atraiga" o defina la curvatura.
    - Heurística de Triangulación Segura: Si el punto intermedio se eligiera al azar en cualquier lugar 
      de la grilla, el robot podría hacer curvas en "S" invertidas, bucles cerrados o retroceder (algo 
      físicamente imposible a altas velocidades). 
      Esta función traza una línea recta imaginaria (usando la pendiente 'slope') entre el robot y la meta, 
      divide el área rectangular (bounding box) en dos triángulos (tri1, tri2) divididos por esa línea, 
      elige uno de los triángulos al azar (50/50) y selecciona un punto en su interior. Esto garantiza 
      matemáticamente que la curva será orgánica, cóncava en una sola dirección y apta para conducción.
    - Espejado lógico: Si la meta está en el borde derecho (gc == grid_size - 1), la función espeja el 
      problema hacia la izquierda, calcula el punto, y vuelve a espejar el resultado para no duplicar código.

    Args:
        goal: Coordenada de la meta (fila, columna).
        robot: Coordenada actual del robot (fila, columna).
        grid_size: Tamaño de la grilla.
        rng: Generador de números aleatorios.

    Returns:
        Una coordenada (fila, columna) que actuará como vértice o punto de control para la curva.
    """
    rng = rng or random
    gr, gc = int(goal[0]), int(goal[1])
    robot_row, robot_col = int(robot[0]), int(robot[1])
    grid_size = int(grid_size)

    if gr == 0:
        goal_col = gc
        mid_col = grid_size // 2
        left_bound = min(goal_col, mid_col)
        right_bound = max(goal_col, mid_col)
        rect_points = [(r, c) for r in range(0, grid_size) for c in range(left_bound, right_bound + 1)]
        slope = (robot_row - gr) / float(robot_col - goal_col) if robot_col != goal_col else float("inf")
        tri1 = [(r, c) for (r, c) in rect_points if r <= gr + slope * (c - goal_col)]
        tri2 = [(r, c) for (r, c) in rect_points if r >= gr + slope * (c - goal_col)]
        chosen = tri1 if rng.random() < 0.5 else tri2
        return rng.choice(chosen if chosen else rect_points)

    if gc == 0:
        rect_points = [(r, c) for r in range(gr, grid_size) for c in range(0, robot_col + 1)]
        slope = (robot_row - gr) / float(robot_col) if robot_col != 0 else float("inf")
        tri1 = [(r, c) for (r, c) in rect_points if r <= gr + slope * c]
        tri2 = [(r, c) for (r, c) in rect_points if r >= gr + slope * c]
        chosen = tri1 if rng.random() < 0.5 else tri2
        return rng.choice(chosen if chosen else rect_points)

    if gc == grid_size - 1:
        mirrored_goal = (gr, 0)
        mirrored_robot = (robot_row, grid_size - 1 - robot_col)
        mid_mir = sample_mid_for_goal(mirrored_goal, mirrored_robot, grid_size, rng=rng)
        return mid_mir[0], grid_size - 1 - mid_mir[1]

    if rng.random() <= 0.05:
        row_i = 0 if gr == 0 else rng.randint(0, max(0, gr))
    elif gr == grid_size - 1:
        row_i = max(0, grid_size - 1 - 150)
    else:
        row_i = rng.randint(gr, int(gr + (grid_size - gr) / 2))
    return row_i, rng.randint(0, grid_size - 1)


def quadratic_path(
    start: tuple[int, int],
    mid: tuple[int, int],
    goal: tuple[int, int],
    num_samples: int = 50,
) -> np.ndarray:
    """
    Genera una trayectoria polinomial cuadrática (de grado 2) que pasa de manera obligatoria y exacta 
    por los puntos de inicio, medio y meta especificados.

    CONCEPTOS CLAVE Y MATEMÁTICA:
    - Diferencia con una Curva Bézier: En una curva de Bézier estándar, el punto intermedio "atrae" la 
      curva, pero la curva rara vez lo toca. Esta función en cambio resuelve un sistema de ecuaciones 
      para formular un polinomio dependiente del tiempo 't' (donde t va de 0.0 a 1.0) tal que:
        * P(t=0.0) = start
        * P(t=0.5) = mid  (Garantizando que pase exactamente por el punto de control a mitad de tiempo)
        * P(t=1.0) = goal
    - La parametrización matemática por eje (Filas y Columnas) es: P(t) = a0 + a1*t + a2*t^2.
      Los coeficientes a0, a1, a2 (y b0, b1, b2) son el resultado analítico directo de resolver ese sistema.

    Args:
        start: Coordenada de inicio (robot).
        mid: Coordenada del punto de control (intermedio).
        goal: Coordenada final de la meta.
        num_samples: Cantidad de puntos en los que se evaluará el polinomio (intervalos de tiempo t).

    Returns:
        Arreglo bidimensional de NumPy de forma (num_samples + 1, 2) conteniendo la trayectoria suave.
    """
    (r0, c0) = start
    (rm, cm) = mid
    (r1, c1) = goal

    a0 = r0
    a1 = 4 * rm - 3 * r0 - r1
    a2 = (r1 - r0) - a1
    b0 = c0
    b1 = 4 * cm - 3 * c0 - c1
    b2 = (c1 - c0) - b1

    t = np.linspace(0.0, 1.0, int(num_samples) + 1, dtype=np.float64)
    rows = a0 + a1 * t + a2 * (t**2)
    cols = b0 + b1 * t + b2 * (t**2)
    return np.stack([rows, cols], axis=1)


def compute_arc_length(path: np.ndarray) -> np.ndarray:
    """
    Calcula la longitud de arco acumulada (es decir, la distancia física real recorrida) 
    a lo largo de una trayectoria de puntos discretos.

    CONCEPTOS CLAVE:
    - `np.diff`: Obtiene los vectores diferenciales de movimiento entre cada punto consecutivo (pasos).
    - `np.linalg.norm`: Aplica el teorema de Pitágoras (Distancia Euclidiana) a cada paso para saber su longitud.
    - `np.cumsum`: Suma acumulativa. El elemento [i] del resultado dice exactamente cuántos píxeles/metros 
      ha recorrido el vehículo desde el inicio (punto 0) hasta llegar al punto [i]. Se inserta un 0 inicial.

    Args:
        path: Arreglo 2D de coordenadas de la curva.

    Returns:
        Un arreglo 1D que contiene la distancia física acumulada en cada nodo de la trayectoria.
    """
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.insert(np.cumsum(segment_lengths), 0, 0)


def uniformly_sample_by_arclength(
    start: tuple[int, int],
    mid: tuple[int, int],
    goal: tuple[int, int],
    num_points: int = 50,
    high_res: int = 1000,
) -> np.ndarray:
    """
    Re-parametriza una curva cuadrática basándose en su longitud de arco para asegurar que los puntos 
    resultantes estén separados por la misma distancia espacial (equidistantes).

    CONCEPTOS CLAVE:
    - El Problema de Parametrización Temporal: Cuando evalúas un polinomio (como quadratic_path) en intervalos 
      de tiempo regulares (ej. t=0.1, 0.2, 0.3), los puntos en el mapa NO quedan espaciados uniformemente. 
      En las zonas de alta curvatura los puntos se aglomeran, y en las rectas se separan excesivamente 
      (la curva tiene "aceleraciones espaciales"). 
    - Por qué es fatal en robótica: Los controladores de motores asumen que la distancia entre puntos de una 
      trayectoria se recorrerá a velocidad estable; distancias desiguales provocan tirones y frenadas.
    - Solución (Arc-Length Parameterization): Se genera la curva original a una resolución altísima (1000 puntos), 
      se mide su distancia real acumulada, y mediante interpolación lineal (`np.interp`) se buscan a la inversa 
      qué valores exactos del tiempo 't' corresponden a subdivisiones perfectamente equidistantes de distancia.

    Args:
        start: Coordenada inicial.
        mid: Coordenada de curvatura.
        goal: Coordenada final.
        num_points: Cantidad final de puntos uniformes a generar.
        high_res: Resolución interna usada para calcular la curva de medición base (longitud de arco).

    Returns:
        Un arreglo de forma (num_points + 1, 2) con la trayectoria físicamente uniforme.
    """
    high_res_path = quadratic_path(start, mid, goal, num_samples=high_res)
    cum_arc_length = compute_arc_length(high_res_path)
    total_length = float(cum_arc_length[-1])
    if total_length <= 1e-8:
        return np.repeat(np.asarray(start, dtype=np.float64)[None, :], int(num_points) + 1, axis=0)

    target_lengths = np.linspace(0.0, total_length, int(num_points) + 1)
    t_high_res = np.linspace(0.0, 1.0, int(high_res) + 1)
    t_uniform = np.interp(target_lengths, cum_arc_length, t_high_res)

    (r0, c0) = start
    (rm, cm) = mid
    (r1, c1) = goal
    a0 = r0
    a1 = 4 * rm - 3 * r0 - r1
    a2 = (r1 - r0) - a1
    b0 = c0
    b1 = 4 * cm - 3 * c0 - c1
    b2 = (c1 - c0) - b1
    rows = a0 + a1 * t_uniform + a2 * (t_uniform**2)
    cols = b0 + b1 * t_uniform + b2 * (t_uniform**2)
    return np.stack([rows, cols], axis=1)


def is_path_inside_grid(path: np.ndarray, grid_size: int) -> bool:
    """
    Validación de seguridad geométrica: Verifica que absolutamente todos los puntos de la trayectoria 
    generada se encuentren dentro de los límites físicos (0 a grid_size) del mapa (grilla BEV).
    Evita fallos de "Index Out of Bounds" al consultar obstáculos.
    """
    p = np.asarray(path)
    return bool(np.all((p[:, 0] >= 0) & (p[:, 0] < grid_size) & (p[:, 1] >= 0) & (p[:, 1] < grid_size)))


def is_strictly_decreasing_rows(rows: np.ndarray) -> bool:
    """
    Validación cinemática: Asegura que el vehículo siempre avance "hacia adelante" respecto al horizonte.

    CONCEPTOS CLAVE:
    - En nuestro sistema de coordenadas matricial, la parte inferior de la imagen (donde está el robot) 
      tiene números de fila altos (ej. 239) y el horizonte superior es la fila 0.
    - Para que el robot avance frontalmente, el índice de fila debe ir disminuyendo constantemente.
      La condición `np.diff(rows) < 0` garantiza que nunca se genere un punto intermedio que obligue 
      al robot a poner reversa para seguir la curva cuadrática (elimina curvas en U o retrocesos).
    """
    return bool(np.all(np.diff(rows) < 0))


def sample_paths_polynomial(
    robot: tuple[int, int] = (239, 120),
    num_goals: int = 100,
    num_mid_points_per_goal: int = 30,
    num_samples: int = 100,
    grid_size: int = 240,
    goal: tuple[int, int] | None = None,
    include_random_goals: bool = True,
    random_seed: int | None = None,
) -> list[np.ndarray]:
    """
    Función orquestadora (Planner Core): Construye masivamente un "banco" de trayectorias cuadráticas 
    factibles desde el robot hacia diversas metas, las cuales serán evaluadas luego por un sistema de costos.

    CONCEPTOS CLAVE:
    - Planificación basada en Muestreo (Sampling-based Motion Planning): En entornos complejos, resolver 
      una trayectoria única matemáticamente es demasiado lento o inviable. En cambio, se "inunda" el 
      espacio con cientos de trayectorias curvas posibles (muestreo) y luego un algoritmo (como GeNIE o MPC) 
      filtra y elige la ruta con menor riesgo o penalidad.
    - El ciclo principal opera así:
      1. Define u obtiene múltiples metas perimetrales (`sample_goals`).
      2. Para CADA meta, genera ramificaciones creando decenas de puntos intermedios (`sample_mid_for_goal`).
      3. Traza curvas polinomiales re-parametrizadas por distancia constante hacia esas metas (`uniformly_sample_by_arclength`).
      4. Actúa como filtro: descarta toda trayectoria que salga del mapa o retroceda.
      5. Devuelve la colección élite de curvas dinámicamente viables.

    Args:
        robot: Posición inicial estática en el tensor/imagen (por defecto centro inferior: fila 239, col 120).
        num_goals: Número total de metas distintas a explorar.
        num_mid_points_per_goal: Nivel de ramificación (variantes de curvatura) para cada meta individual.
        num_samples: Cantidad de nodos espaciales que conformarán cada trayectoria final generada.
        grid_size: Dimensiones cuadradas de la grilla espacial (ej. 240 píxeles).
        goal: (Opcional) Un punto de destino obligatorio. Si se provee, se explorarán variantes hacia él.
        include_random_goals: Si se deshabilita y se provee un 'goal', el algoritmo únicamente 
                              explorará curvas dirigidas hacia ese objetivo exclusivo.
        random_seed: Semilla del generador pseudoaleatorio para reproducibilidad en simulaciones y pruebas.

    Returns:
        Una lista de arreglos bidimensionales (np.ndarray de tipo float32). Cada arreglo contiene 
        una trayectoria independiente validada y lista para su ejecución o evaluación de colisión.
    """
    rng = random.Random(random_seed) if random_seed is not None else random
    goals: list[tuple[int, int]] = []
    if goal is not None:
        gr, gc = int(goal[0]), int(goal[1])
        if 0 <= gr < int(grid_size) and 0 <= gc < int(grid_size):
            goals.append((gr, gc))

    if include_random_goals:
        remaining = max(0, int(num_goals) - len(goals))
        goals.extend(sample_goals(remaining, grid_size=int(grid_size), rng=rng))

    if not goals:
        goals = sample_goals(num_goals, grid_size=int(grid_size), rng=rng)

    all_paths: list[np.ndarray] = []
    robot_rc = (int(robot[0]), int(robot[1]))
    for goal_rc in goals:
        for _ in range(int(num_mid_points_per_goal)):
            mid = sample_mid_for_goal(goal_rc, robot_rc, int(grid_size), rng=rng)
            path = uniformly_sample_by_arclength(
                robot_rc,
                mid,
                goal_rc,
                num_points=int(num_samples),
                high_res=1000,
            )
            if is_path_inside_grid(path, int(grid_size)) and is_strictly_decreasing_rows(path[:, 0]):
                all_paths.append(path.astype(np.float32))
    return all_paths