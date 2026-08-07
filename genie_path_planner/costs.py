from __future__ import annotations

import numpy as np


def path_cost_with_footprint(
    raw_topdown_score: np.ndarray,
    candidate_path: np.ndarray,
    alpha: float = 0.5,
    footprint_px: int = 20,
) -> float:
    """Calcula el costo total de peligro para una trayectoria, considerando el volumen real del vehículo.

    En lugar de evaluar la trayectoria como una línea infinitamente delgada (lo que haría
    que el vehículo intente pasar a un milímetro de los obstáculos), esta función extrae una
    "huella" (footprint) cuadrada alrededor de cada punto del camino para verificar que el 
    cuerpo físico del vehículo quepa en el espacio seguro.

    CONCEPTOS CLAVE:
    - raw_topdown_score (Mapa de Costos): Es una grilla 2D (vista superior) donde cada píxel 
      tiene un valor numérico de peligro.
        * Valores cercanos a 0.0: Espacio libre, camino ideal y seguro.
        * Valores medios: Terreno rugoso o proximidad indeseable a carriles/bordes.
        * Valores altos / infinitos: Obstáculos impenetrables (paredes, peatones, agujeros).
    - alpha (Factor de Aversión al Riesgo): Es un multiplicador de "miedo". Se utiliza dentro
      de una función exponencial: exp(alpha * costo).
        * Por qué exponencial y no promedio simple: Si el 95% del vehículo está en zona libre (costo 0) 
          pero una esquina del parachoques toca una pared (costo 10), un promedio simple daría un costo 
          bajo (~0.5) y el vehículo avanzaría y chocaría. Al aplicar exp(alpha * 10), el valor de ese 
          único píxel estalla a un número gigantesco (ej. si alpha=1, exp(10) ~ 22026). Al sacar el 
          promedio de la ventana, ese número gigante domina el resultado, descartando el camino por 
          completo. A mayor alpha, más lejos se mantendrá el vehículo de cualquier obstáculo.
    - footprint_px (Huella en Píxeles): Tamaño del lado del cuadrado que representa las dimensiones
      físicas del agente sobre el mapa en cada paso del camino.

    Args:
        raw_topdown_score: Matriz 2D de costos donde mayor número indica mayor peligro.
        candidate_path: Arreglo de forma (N, 2) con secuencias de puntos [fila, columna].
        alpha: Factor de escala exponencial que penaliza agresivamente los obstáculos.
        footprint_px: Tamaño en píxeles del lado de la ventana cuadrada (cuerpo del agente).

    Returns:
        float: El costo acumulado a lo largo de toda la trayectoria. A menor costo, camino más seguro.
    """
    # Calculamos el radio (semiancho) en píxeles desde el centro del robot hasta su borde.
    # Se garantiza un mínimo de 1 píxel de radio para no tener divisiones por cero o cajas vacías.
    r_half = max(1, int(footprint_px)) // 2
    """
    Si el mapa es de una sola capa de datos (como un mapa de calor en blanco y negro), 
    su .shape será (800, 600).
    Si el mapa tuviera capas de color u otros canales extra (por ejemplo, rojo, verde, azul), 
    su .shape podría ser (800, 600, 3)
    Si la matriz fuera (800, 600, 3), el slicing [:2] ignora el 3 y se queda solo con (800, 600). 
    Esto hace que el código sea seguro y no falle sin importar qué tipo de imagen o mapa le pases.
    """
    h, w = raw_topdown_score.shape[:2]
    total_cost = 0.0

    # Recorremos cada punto temporal (paso) de la trayectoria
    for row, col in candidate_path:
        # Los planificadores generan coordenadas continuas (con decimales). 
        # Las redondeamos al índice discreto (entero) de la matriz de costos más cercano.
        r_coord = int(round(float(row)))
        c_coord = int(round(float(col)))

        """
        El código intenta dibujar un cuadrado (la huella del vehículo) alrededor del punto donde 
        se encuentra el robot en ese instante para calcular si va a chocar:
        Al extraer la altura (h) y el ancho (w), el código luego utiliza la función min() 
        para decirle al programa: "Si el cuadrado del vehículo se sale del mapa, recorta el cuadrado para que 
        llegue como máximo hasta el límite h o w"
        """
        r1 = max(0, r_coord - r_half)
        r2 = min(h, r_coord + r_half + 1)
        c1 = max(0, c_coord - r_half)
        c2 = min(w, c_coord + r_half + 1)

        # Si tras el recorte la ventana es inválida o el punto se salió completamente del mapa,
        # ignoramos este paso y continuamos.
        if r1 >= r2 or c1 >= c2:
            continue

        # Extraemos la submatriz que representa el espacio físico que ocupa el vehículo en ese instante
        region = raw_topdown_score[r1:r2, c1:c2]

        # APLICACIÓN DE LA PENALIZACIÓN EXPONENCIAL:
        # 1. Multiplicamos la región por alpha (aumentando la severidad de los peligros).
        # 2. Aplicamos np.exp() para hacer que los costos altos exploten exponencialmente.
        # 3. Tomamos el promedio (.mean()) de esa ventana para resumir el peligro en ese instante.
        # 4. Lo sumamos al costo total de la trayectoria.
        total_cost += float(np.exp(float(alpha) * region).mean())

    return total_cost


def compute_paths_costs(
    candidate_paths: list[np.ndarray],
    raw_topdown_score: np.ndarray,
    alpha: float,
    footprint_px: int = 20,
) -> list[tuple[float, np.ndarray]]:
    """Evaluación en lotes (batch processing) para múltiples trayectorias candidatas.

    CONTEXTO DE NAVEGACIÓN:
    Los algoritmos modernos de conducción autónoma o robótica (como GeNIE, MPC o planificadores 
    basados en muestreo) no buscan una sola trayectoria perfecta analíticamente. En su lugar, 
    generan de golpe decenas o cientos de líneas curvas posibles que avanzan hacia la meta y luego 
    utilizan esta función para calificarlas todas y ver cuál es la más segura.

    Args:
        candidate_paths: Lista que contiene cientos de arreglos (N, 2), cada uno es un camino posible.
        raw_topdown_score: El mapa 2D de peligros/costos del entorno.
        alpha: Multiplicador de aversión al riesgo para la función exponencial.
        footprint_px: Tamaño del vehículo en píxeles.

    Returns:
        list[tuple[float, np.ndarray]]: Devuelve la misma lista pero emparejando cada camino
        con su costo total calculado: [(costo_camino_1, camino_1), (costo_2, camino_2), ...].
    """
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
    """Selecciona el camino final fusionando las mejores trayectorias y aplicando un salvavidas de seguridad.

    ESTRATEGIA GE-NIE (Fusión + Salvavidas):
    En lugar de simplemente elegir la trayectoria número 1 y descartar el resto (lo que provocaría 
    que el volante del vehículo tiemble bruscamente si en el siguiente instante la trayectoria 2 pasa
    a ser la mejor), este algoritmo intenta generar un camino completamente nuevo, suave y estable
    haciendo un promedio ponderado punto a punto de las mejores "k" trayectorias (Top-K).

    POR QUÉ EXISTE EL FALLBACK:
    ¿Qué pasa si intentas promediar dos caminos seguros? Imagina un árbol en medio de la carretera: 
    tienes un camino seguro que lo esquiva por la izquierda (Costo bajo), y otro que lo esquiva por 
    la derecha (Costo bajo). ¡Si promedias las coordenadas de ambos, la trayectoria resultante va 
    directamente en línea recta contra el árbol en el centro!
    
    Para evitar este choque catastrófico, esta función recalcula el costo real en el mapa de la nueva 
    trayectoria promediada. Si detecta que el promedio cortó por un obstáculo (su costo superó al del 
    peor camino admitido en el Top-K), descarta el invento y activa el "salvavidas": devuelve la trayectoria
    original de menor costo.

    Args:
        paths_with_cost: Lista de tuplas (costo, trayectoria) previamente calculada.
        best_k: Número de trayectorias élite (las de menor costo) que se usarán para promediar.
        num_samples: Cantidad de pasos discretos (N) que componen cada trayectoria.
        cost_map: Mapa 2D de peligro/costo para evaluar si el camino promediado es seguro.
        alpha: Factor de aversión al riesgo exponencial.
        footprint_px: Tamaño físico del vehículo en píxeles.

    Returns:
        tuple[np.ndarray, np.ndarray]: Dos arreglos 1D conteniendo las coordenadas (filas, columnas)
        de la trayectoria definitiva que el controlador del motor del vehículo debe seguir.
    """
    # Caso base: si no llegó ninguna trayectoria válida, devolvemos arrays vacíos.
    if not paths_with_cost:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # 1. ORDENAMIENTO Y SELECCIÓN DEL TOP-K:
    # Ordenamos la lista de menor a mayor costo (de más seguro a más peligroso).
    paths_sorted = sorted(paths_with_cost, key=lambda x: x[0])
    # Nos quedamos únicamente con las "k" mejores opciones (asegurando al menos 1 opción).
    top = paths_sorted[: int(max(1, best_k))]

    # inicializa dos arreglos de tamaño num_samples en 0 
    # que almacenan las coordenadas de los tramos que componen a la trayectoria definitiva
    final_rows = np.zeros(int(num_samples), dtype=np.float32)
    final_cols = np.zeros(int(num_samples), dtype=np.float32)

    # 2. CÁLCULO DE PESOS PARA EL PROMEDIO:
    # Extraemos el valor del costo de las trayectorias del Top-K.
    # NOTA TÉCNICA: En este comportamiento heredado de GeNIE, se usan los propios costos 
    # como ponderadores directamente para equilibrar la contribución espacial de las curvas 
    # del top-k en la trayectoria intermedia resultante.
    # El camino número 1 (el del costo más bajo de todos) a veces puede ser una curva muy cerrada 
    # o un corte muy agresivo que pasa raspando el límite de lo que el algoritmo considera seguro.
    # Las otras trayectorias del Top-K (que tienen un costo ligeramente superior porque dan una vuelta un milímetro más ancha) 
    # son geométrica y espacialmente más "conservadoras" o abiertas.
    # Al usar los costos directamente como ponderadores, GeNIE logra un efecto de estabilización o 
    # contrapeso geométrico: tira de la trayectoria final un poco más hacia el centro del grupo 
    # de curvas candidatas, evitando que el vehículo adopte una trayectoria extremo o demasiado agresiva 
    # que dependa únicamente de la muestra número 1.
    weights = np.array([p[0] for p in top], dtype=np.float32)
    
    # Evitamos división por cero en caso de que todos los costos sean matemáticamente 0
    if float(np.sum(weights)) <= 1e-8:
        weights = np.ones_like(weights)
    
    # Convertimos los costos en porcentajes de influencia que suman 1.0. Si no hiciéramos esto,
    # el promedio ponderado magnificaría la escala física real, arrojando el camino fuera del mapa.
    weights = weights / np.sum(weights)

    # 3. FUSIÓN / PROMEDIO PONDERADO PUNTO A PUNTO:
    # Para cada paso temporal i desde 0 hasta num_samples:
    for i in range(int(num_samples)):
        # Recorremos cada una de las trayectorias del Top-K
        for idx, (_cost, path) in enumerate(top):
            # Sumamos la coordenada de ese paso, multiplicada por su porcentaje de peso
            final_rows[i] += float(weights[idx]) * float(path[i][0])
            final_cols[i] += float(weights[idx]) * float(path[i][1])

    # Apilamos las filas y columnas para reconstruir una trayectoria de forma (N, 2)
    weighted_path = np.stack([final_rows, final_cols], axis=1)

    # 4. EVALUACIÓN DE LA VERDAD DE LA NUEVA TRAYECTORIA:
    # Evaluamos en el mapa si la curva suave inventada por el promedio es realmente segura,
    # pasándola por el evaluador exponencial que considera el tamaño del robot.
    weighted_cost = path_cost_with_footprint(
        cost_map,
        weighted_path,
        alpha=float(alpha),
        footprint_px=int(footprint_px),
    )

    # 5. EL SALVAVIDAS DE SEGURIDAD (FALLBACK MECHANISM):
    # Encontramos cuál fue el costo más alto (el límite de peligro aceptado) dentro de nuestro Top-K.
    worst_top_cost = max(p[0] for p in top)
    
    # Verificación del "Árbol en el centro":
    # ¿El costo de la trayectoria promediada es PEOR que la peor de las trayectorias individuales del Top-K?
    if weighted_cost > worst_top_cost:
        # ¡Peligro! El promedio cruzó por una zona indeseable u obstáculo.
        # Descartamos el camino promediado completamente y retornamos la trayectoria absoluta número 1
        # (top[0][1]), que sabemos con certeza que es físicamente segura y la mejor evaluada.
        best_path = np.asarray(top[0][1], dtype=np.float32)
        return best_path[:, 0].astype(np.float32), best_path[:, 1].astype(np.float32)

    # Si weighted_cost es aceptable, el camino fusionado es seguro, estable y suave.
    # Devolvemos las coordenadas del promedio para que el vehículo navegue por ahí.
    return final_rows, final_cols