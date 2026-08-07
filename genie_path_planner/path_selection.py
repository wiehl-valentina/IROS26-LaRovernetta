from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover - optional dependency failure handled at runtime.
    KMeans = None
    silhouette_score = None

_SKLEARN_THREADPOOL_BYPASS_READY = False


def has_majority_of_high_cost_points(
    raw_topdown_score: np.ndarray,
    candidate_path: np.ndarray,
    num_points: int,
    footprint_px: int = 20,
    threshold_points_ratio: float = 0.5,
    threshold_cost: float = 0.8,
) -> bool:
    """
    Evalúa si una trayectoria candidata es demasiado peligrosa, basándose en la proporción 
    de puntos de alto costo que el vehículo pisaría considerando su volumen (huella).

    CONCEPTOS CLAVE:
    - Evaluación por Huella (Footprint box): En lugar de mirar si la línea del centro de la trayectoria 
      toca un obstáculo, extrae una región cuadrada (ventana) alrededor de cada punto de la curva.
    - Tolerancia al riesgo (threshold_points_ratio): En mapas del mundo real siempre hay ruido en los sensores. 
      Si un solo píxel reporta peligro falso, no queremos descartar todo el camino. Esta función permite 
      que un pequeño porcentaje del área del vehículo (ej. < 5%) roce zonas de costo elevado antes de declarar 
      la trayectoria como inválida.

    Args:
        raw_topdown_score: Matriz 2D (grilla de costos) donde valores más altos indican mayor peligro.
        candidate_path: Arreglo (N, 2) con las coordenadas de la trayectoria a evaluar.
        num_points: Límite de puntos de la trayectoria a evaluar (útil para evaluar solo el futuro inmediato).
        footprint_px: Tamaño de la huella del robot en píxeles (crea una caja de este ancho/alto).
        threshold_points_ratio: Porcentaje (0.0 a 1.0) del área de la huella que debe superar el 
                                umbral de costo para considerar que hubo un choque en ese instante.
        threshold_cost: Valor mínimo en la grilla para considerar un píxel como "peligro alto".

    Returns:
        bool: True si la trayectoria es peligrosa (superó los umbrales), False si es segura.
    """
    r_half = max(1, int(footprint_px)) // 2
    h, w = raw_topdown_score.shape[:2]

    for row, col in candidate_path[: int(num_points)]:
        r_coord = int(round(float(row)))
        c_coord = int(round(float(col)))
        r1 = max(0, r_coord - r_half)
        r2 = min(h, r_coord + r_half + 1)
        c1 = max(0, c_coord - r_half)
        c2 = min(w, c_coord + r_half + 1)
        region = raw_topdown_score[r1:r2, c1:c2]
        
        # Si la ventana quedó fuera del mapa, se considera un área inválida y peligrosa.
        if region.size == 0:
            return True
            
        threshold_points = int(np.ceil(float(threshold_points_ratio) * region.size))
        # Cuenta cuántos píxeles en la ventana superan el límite de costo y comprueba si superan el ratio permitido.
        if int(np.sum(region >= float(threshold_cost))) >= threshold_points:
            return True
    return False


def filter_paths_with_high_costs(
    candidate_paths: Iterable[np.ndarray],
    raw_topdown_score: np.ndarray,
    num_points: int,
    footprint_px: int = 20,
    threshold_points_ratio: float = 0.05,
    threshold_cost: float = 0.8,
) -> list[np.ndarray]:
    """
    Actúa como un colador: recibe cientos de trayectorias, evalúa cada una y descarta 
    aquellas que son inherentemente peligrosas o que chocan con obstáculos.

    CONCEPTOS CLAVE:
    - Esta función es el paso previo obligatorio antes de hacer "Clustering". No tiene sentido 
      agrupar y analizar matemáticamente trayectorias que de todos modos nos llevarían a un choque.

    Args:
        candidate_paths: Iterable de arreglos que representan múltiples trayectorias posibles.
        raw_topdown_score: Mapa 2D de costos del entorno.
        num_points: Cantidad de pasos a mirar hacia adelante en cada trayectoria.
        footprint_px: Tamaño del vehículo en píxeles.
        threshold_points_ratio: Proporción máxima permitida de impacto (defecto 5%).
        threshold_cost: Costo límite para considerar un píxel como peligroso.

    Returns:
        Una lista filtrada que contiene únicamente las trayectorias (np.ndarray) que resultaron seguras.
    """
    result: list[np.ndarray] = []
    for path in candidate_paths:
        if not has_majority_of_high_cost_points(
            raw_topdown_score,
            path,
            num_points=num_points,
            footprint_px=footprint_px,
            threshold_points_ratio=threshold_points_ratio,
            threshold_cost=threshold_cost,
        ):
            result.append(path)
    return result


def _ensure_sklearn_threadpool_bypass() -> None:
    """
    Instala un controlador falso (no-op) para evadir un fallo crítico conocido en la librería scikit-learn.

    CONCEPTOS CLAVE (Hacking de Infraestructura):
    - El Problema: Scikit-learn usa por debajo una librería llamada `threadpoolctl` para paralelizar 
      operaciones matemáticas pesadas. En ecosistemas de robótica (como ROS2 o sistemas con multiprocessing 
      complejo), esto suele causar un estado de bloqueo permanente ("deadlock") o pausas (stalls) infinitas.
    - La Solución: Esta función intercepta los módulos internos de paralelización de sklearn e inyecta 
      clases falsas (`_NoopLimit` y `_NoopThreadpoolController`) que le dicen a la librería: "Sí, ya 
      controlé los hilos, no intentes hacerlo tú". Esto asegura que el algoritmo de clustering funcione 
      en un solo hilo de forma fluida y sin congelar el sistema de conducción del vehículo.
    """
    global _SKLEARN_THREADPOOL_BYPASS_READY
    if _SKLEARN_THREADPOOL_BYPASS_READY:
        return
    try:
        import sklearn.cluster._kmeans as sklearn_kmeans_module
        import sklearn.utils.parallel as sklearn_parallel
    except Exception:
        return

    class _NoopLimit:
        def __enter__(self) -> "_NoopLimit":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    class _NoopThreadpoolController:
        def limit(self, limits: object = None, user_api: object = None) -> _NoopLimit:
            del limits, user_api
            return _NoopLimit()

        def info(self) -> list[object]:
            return []

    controller = _NoopThreadpoolController()
    sklearn_parallel._threadpool_controller = controller
    sklearn_parallel._get_threadpool_controller = lambda: controller
    sklearn_kmeans_module._get_threadpool_controller = lambda: controller
    _SKLEARN_THREADPOOL_BYPASS_READY = True


def adaptive_kmeans(
    paths: list[np.ndarray],
    min_clusters: int = 1,
    max_clusters: int = 10,
    random_state: int = 42,
    inertia_weight: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Agrupa (clusters) automáticamente un conjunto de trayectorias en clases topológicas utilizando 
    K-Means y la métrica de Silueta (Silhouette Score) para descubrir la cantidad ideal de grupos.

    CONCEPTOS CLAVE:
    - Agrupación Topológica (Topological Clustering): Si hay un árbol en medio del camino, las curvas 
      se dividirán naturalmente en dos grupos: "esquivar por la izquierda" y "esquivar por la derecha". 
      En lugar de promediar todas las curvas juntas (lo cual chocaría de frente con el árbol), este 
      algoritmo usa Inteligencia Artificial (K-Means) para identificar esos grupos separados.
    - Vectores de Características (Flattening): K-Means no entiende curvas bidimensionales continuas, 
      por lo que cada trayectoria Nx2 se aplasta (`flatten`) en un solo vector unidimensional (1D).
    - Selección Adaptativa del factor 'K': El algoritmo no sabe de antemano cuántos obstáculos hay. 
      Prueba agrupando desde 1 hasta 10 grupos. Para cada prueba, calcula el `silhouette_score` (que 
      mide qué tan bien separados y compactos quedaron los grupos). Se queda con la configuración 
      que obtenga el puntaje matemático más alto.

    Args:
        paths: Lista de trayectorias (arreglos 2D) a clasificar.
        min_clusters: Cantidad mínima de grupos a probar.
        max_clusters: Cantidad máxima límite de grupos a probar.
        random_state: Semilla para hacer determinista el algoritmo.
        inertia_weight: Penalización leve para desempatar la puntuación cuando solo hay 1 cluster.

    Returns:
        Tupla con:
        1. best_labels: Arreglo 1D indicando a qué grupo pertenece cada trayectoria.
        2. best_centroids: Las trayectorias "promedio" que representan el centro de cada grupo.
        3. best_k: La cantidad de grupos que el algoritmo determinó como ideal.
    """
    if KMeans is None or silhouette_score is None:
        raise RuntimeError("scikit-learn is required when planner clustering is enabled")
    if len(paths) == 0:
        raise ValueError("paths is empty")
    _ensure_sklearn_threadpool_bypass()

    feature_vectors = np.array([np.asarray(path, dtype=np.float32).flatten() for path in paths])
    best_score = -np.inf
    best_k = int(min_clusters)
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None

    max_clusters = min(int(max_clusters), len(paths))
    for k in range(int(min_clusters), max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
        labels = kmeans.fit_predict(feature_vectors)
        if k == 1:
            score = -float(kmeans.inertia_) * float(inertia_weight)
        else:
            try:
                score = float(silhouette_score(feature_vectors, labels))
            except ValueError:
                score = -np.inf
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_centroids = kmeans.cluster_centers_

    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, best_k


def merge_centroids_by_angle(
    centroids: np.ndarray,
    angle_threshold_deg: float,
) -> tuple[np.ndarray, list[list[int]], list[float]]:
    """
    Fusiona clústeres (grupos) que resultaron matemáticamente distintos para K-Means, pero que 
    físicamente inician dirigiéndose en la misma orientación angular.

    CONCEPTOS CLAVE:
    - K-Means mira toda la trayectoria de principio a fin. Esto significa que puede crear dos grupos distintos 
      si dos caminos empiezan igual pero terminan en distintos puntos lejanos. 
    - Para la conducción del vehículo (que solo ejecuta los primeros metros inmediatos de la curva), 
      estas dos trayectorias exigen la misma acción del volante. Esta función analiza el ángulo 
      de salida de los centroides (usando los primeros ~30 pasos) y, si la diferencia angular es menor 
      al límite permitido, los fusiona en un solo súper-grupo para simplificar la decisión final.

    Args:
        centroids: Los caminos representativos de cada grupo detectado por K-Means.
        angle_threshold_deg: Umbral en grados. Si dos grupos difieren menos que esto en su inicio, se fusionan.

    Returns:
        Tupla con:
        1. merged_centroids: Los nuevos centroides unificados.
        2. groups: Lista de listas indicando qué índices originales se unificaron en cada nuevo grupo.
        3. centroid_angles: Los ángulos de orientación calculados de los centroides originales.
    """
    merged_centroids: list[np.ndarray] = []
    groups: list[list[int]] = []
    centroid_angles: list[float] = []

    for centroid in centroids:
        # Desaplasta el vector 1D devuelto por K-Means a su forma geométrica original Nx2
        path = centroid.reshape(-1, 2)
        start_point = path[0]
        # Toma un punto más adelante (lookahead index 30) para calcular la línea de tendencia de inicio
        end_point = path[min(30, len(path) - 1)]
        # Calcula el ángulo en radianes usando arcotangente 2 (considerando filas y columnas)
        angle_rad = math.atan2(end_point[0] - start_point[0], end_point[1] - start_point[1])
        centroid_angles.append(math.degrees(angle_rad))

    for idx, angle in enumerate(centroid_angles):
        best_group: list[int] | None = None
        best_diff = float("inf")
        # Revisa los grupos existentes para ver si este ángulo es compatible con alguno
        for group in groups:
            min_diff = min(abs(angle - centroid_angles[i]) for i in group)
            if min_diff < best_diff:
                best_diff = min_diff
                best_group = group
        
        # Si se encuentra un grupo y su diferencia de ángulo está dentro del límite, lo añade.
        if best_group is not None and best_diff < float(angle_threshold_deg):
            best_group.append(idx)
        else:
            # Si diverge demasiado, crea una nueva ramificación independiente.
            groups.append([idx])

    # Construye los nuevos trayectos promediando (fusionando) aquellos que se agruparon
    for group in groups:
        merged_centroids.append(np.mean(np.array([centroids[i] for i in group]), axis=0))
    return np.array(merged_centroids), groups, centroid_angles


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """
    Calcula la diferencia más corta posible entre dos ángulos considerando la naturaleza circular de 360 grados.
    Ejemplo: La diferencia entre 359° y 1° no es 358°, sino apenas 2°.
    """
    diff = abs(float(a_deg) - float(b_deg))
    return float(360.0 - diff if diff > 180.0 else diff)


def _path_orientation_deg(path_rc: np.ndarray, lookahead_index: int = 30) -> float:
    """
    Calcula la dirección inicial real (en grados) de una trayectoria en el mapa matricial, 
    usando un punto futuro (lookahead) para estabilizar la medición.
    """
    path = np.asarray(path_rc, dtype=np.float32).reshape(-1, 2)
    if path.shape[0] == 0:
        return 0.0
    start = path[0]
    end = path[min(max(1, int(lookahead_index)), path.shape[0] - 1)]
    return float(math.degrees(math.atan2(float(end[0] - start[0]), float(end[1] - start[1]))))


def select_best_group_by_closest_path_angle(
    paths: list[np.ndarray],
    labels: np.ndarray,
    groups: list[list[int]],
    robot_goal_angle: float,
    lookahead_index: int = 30,
) -> tuple[int, float]:
    """
    Selecciona la clase o grupo topológico definitivo basándose en qué grupo posee la 
    trayectoria individual que mejor se alinea con el objetivo global del robot.

    CONCEPTOS CLAVE:
    - Después de que las trayectorias pasaron todos los filtros de peligro, se agruparon por K-Means 
      (separando esquives izquierdos de derechos) y se fusionaron las redundantes, el robot tiene un menú 
      reducido de "Decisiones de Grupo".
    - ¿Cómo elige? Evalúa la orientación inicial de todas las trayectorias individuales sobrevivientes. 
      Encuentra la curva cuya dirección apunte de la manera más directa (menor diferencia de ángulo) hacia 
      la meta global final (`robot_goal_angle`). El grupo topológico al que pertenezca esa curva óptima 
      es declarado el grupo ganador.

    Args:
        paths: Lista de todas las trayectorias filtradas y seguras.
        labels: Arreglo de etiquetas devuelto por K-Means (a qué clúster pertenece cada trayectoria).
        groups: La lista de grupos fusionados por `merge_centroids_by_angle`.
        robot_goal_angle: El ángulo ideal en grados en el que se encuentra la meta final global.
        lookahead_index: Punto a evaluar de la curva para medir su ángulo (defecto 30 pasos).

    Returns:
        Tupla con:
        1. best_group_idx: El índice ganador en la lista `groups`.
        2. best_group_diff: La desviación angular mínima encontrada respecto a la meta ideal.
    """
    # 1. Encontrar la mejor curva individual (la más alineada) dentro de CADA etiqueta K-Means original.
    best_diff_by_label: dict[int, float] = {}
    for path, lab in zip(paths, labels):
        lab_i = int(lab)
        diff = _angle_diff_deg(_path_orientation_deg(path, lookahead_index), robot_goal_angle)
        prev = best_diff_by_label.get(lab_i)
        if prev is None or diff < prev:
            best_diff_by_label[lab_i] = diff

    best_group_idx = 0
    best_group_diff = float("inf")
    
    # 2. Iterar por los grupos fusionados y buscar cuál contiene la etiqueta original que resultó victoriosa.
    for gi, group in enumerate(groups):
        # group_best es la mejor diferencia de ángulo encontrada dentro de todas las etiquetas que conforman este grupo fusionado.
        group_best = min((best_diff_by_label.get(int(lab), float("inf")) for lab in group), default=float("inf"))
        
        # Guardamos el grupo que haya tenido la desviación más baja.
        if group_best < best_group_diff:
            best_group_idx = int(gi)
            best_group_diff = float(group_best)
            
    return best_group_idx, best_group_diff