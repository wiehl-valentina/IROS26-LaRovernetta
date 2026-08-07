from __future__ import annotations

import math
from typing import Any

import numpy as np


def as_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    """
    Valida, convierte y reformatea cualquier entrada de datos a una matriz de NumPy
    con una forma y tipo exactos (por defecto, flotantes de 64 bits de alta precisión).

    ¿POR QUÉ SE USA ESTO?
    En robótica es común que las transformaciones lleguen en formatos inconsistentes
    desde distintos sensores o nodos de ROS (ej. una lista plana de 16 números, una tupla,
    o un array 1D). Si intentas multiplicar matrices con formas incorrectas, el código
    colapsaría en tiempo de ejecución. Esta función actúa como un filtro de seguridad
    riguroso que garantiza que la geometría tenga la forma exacta esperada antes de operar.

    Ejemplo:
        Una lista plana [1, 0, 0, 0, 0, 1, ...] de 16 elementos se reformatea 
        automáticamente a una matriz de transformación homogénea (4, 4).
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == shape:
        return arr
    # Si los datos vienen aplanados en 1D pero tienen la cantidad correcta de elementos,
    # los reformateamos (reshape) a la matriz 2D que el algoritmo necesita.
    if arr.size == shape[0] * shape[1]:
        return arr.reshape(shape)
    raise ValueError(f"{name} must have shape {shape}, got {arr.shape}") 

def pose_xy_yaw_to_matrix(xy_yaw: Any) -> np.ndarray:
    """
    Convierte una orientación simple 2D [x, y, yaw] en una matriz de transformación 
    homogénea de 3D de 4x4, utilizando la convención estándar de ejes de ROS.

    MATEMÁTICA Y CONVENCIÓN ROS:
    - Eje X (Adelante): Hacia donde apunta la nariz del robot.
    - Eje Y (Izquierda): Hacia el lateral izquierdo del chasis.
    - Eje Z (Arriba): Apuntando hacia el cielo (ortogonal al suelo).
    - Yaw (Ángulo de rotación θ): Rotación en radianes sobre el eje Z respecto a X.

    ¿POR QUÉ UNA MATRIZ 4x4 SI EL ROBOT SE MUEVE EN EL SUELO (2D)?
    Las matrices homogéneas de 4x4 permiten combinar rotaciones y traslaciones
    en una sola operación de multiplicación matricial. Si usáramos solo
    3x3, no podríamos representar la posición (x, y); si usáramos solo sumas, no 
    podríamos rotar el sistema de referencia. Es el estándar universal en robótica.

    Ejemplo:
        Un robot en x=2.0m, y=3.0m mirando hacia la izquierda (yaw = π/2 radianes ~ 1.57)
        generará una matriz donde el eje X apunta hacia el eje Y global (+90 grados).
    """
    # SANEAMIENTO Y PRE-CÁLCULO TRIGONOMÉTRICO:
    # 1. Convierte cualquier entrada (lista, tupla o matriz) a un arreglo 1D plano de flotantes 
    #    de 64 bits (.reshape(-1)) y valida estrictamente que contenga exactamente 3 elementos.
    # 2. Desempaqueta las coordenadas espaciales (x, y) y el ángulo de orientación (yaw).
    # 3. Pre-calcula el coseno (c) y el seno (s) de 'yaw' para no repetir operaciones pesadas 
    #    de coma flotante al construir la matriz de rotación en el siguiente paso.
    vals = np.asarray(xy_yaw, dtype=np.float64).reshape(-1)
    if vals.size != 3:
        raise ValueError(f"robot_pose_xy_yaw must contain [x, y, yaw_rad], got {vals}") 
    x, y, yaw = float(vals[0]), float(vals[1]), float(vals[2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    
    # Inicializamos una matriz identidad de 4x4 (puros ceros con unos en la diagonal principal)
    pose = np.eye(4, dtype=np.float64)
    
    # Rellenamos la submatriz superior izquierda de 3x3 con la rotación en el eje Z.
    # Esta es la matriz de rotación clásica de rotación en 2D en plano XY:
    # [ cos(θ)  -sin(θ)   0 ]
    # [ sin(θ)   cos(θ)   0 ]
    # [   0        0      1 ]
    pose[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Rellenamos la cuarta columna (índice 3) con el vector de traslación [x, y, z]
    pose[:3, 3] = np.array([x, y, 0.0], dtype=np.float64)
    return pose


def normalize_xy(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """
    Normaliza un vector 2D para que su longitud (magnitud) sea exactamente 1.0, 
    o devuelve un vector de seguridad (fallback) si el vector original es inválido.

    ¿POR QUÉ SE NECESITA ESTO?
    En geometría direccional (por ejemplo, "¿hacia qué dirección apunta la cámara?"), 
    la escala o largo del vector no nos importa, solo nos importa su dirección pura.
    Al dividir un vector por su norma Euclidiana, obtenemos un
    vector unitario (magnitud 1.0) que facilita trigonométricamente los cálculos espaciales.

    EL PROBLEMA DE LA DIVISIÓN POR CERO (WHY FALLBACK?):
    Si por un error de lectura del sensor el vector es [0.0, 0.0], su norma es 0. 
    Dividir por cero generaría un error NaN (Not a Number) que colapsaría el planificador.
    Este código evalúa si la norma es casi cero (< 1e-8); si lo es, utiliza un vector 
    de respaldo seguro, garantizando robustez total en el sistema autónomo.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(v))
    if norm > 1e-8:
        return v / norm
    
    # Si el vector original era inválido (cero), intentamos normalizar el vector fallback
    f = np.asarray(fallback, dtype=np.float64).reshape(2)
    f_norm = float(np.linalg.norm(f))
    if f_norm > 1e-8:
        return f / f_norm
    
    # Última línea de defensa: si todo falla, devolvemos el eje X unitario por defecto [1.0, 0.0]
    return np.array([1.0, 0.0], dtype=np.float64)


def camera_planar_axes(camera_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae los vectores unitarios 2D de dirección "adelante" e "izquierda" proyectados 
    al plano del suelo (XY) a partir de la matriz de postura de una cámara óptica.

    LA PESADILLA DE LAS CONVENCIONES ÓPTICAS VS. ROBÓTICAS:
    A diferencia de un chasis de robot (donde X es adelante y Z es arriba), las lentes 
    de las cámaras por convención óptica mundial utilizan un estándar completamente diferente:
    - Eje Z óptico: Es la profundidad (apunta hacia ADELANTE, hacia donde mira la lente).
    - Eje X óptico: Apunta hacia la DERECHA de la imagen.
    - Eje Y óptico: Apunta hacia ABAJO en la imagen.

    ¿QUÉ HACE ESTA FUNCIÓN?
    Toma la orientación 3D de la cámara, extrae su eje Z (columna índice 2) como la verdadera 
    dirección "adelante" y la proyecta al suelo aplastando la coordenada de altura. Luego calcula
    matemáticamente su eje izquierdo perpendicular girando 90 grados en el plano [-y, x].
    """
    pose = as_matrix(camera_pose, (4, 4), "camera_pose")
    r_world_cam = pose[:3, :3]
    
    # Columna 2 es el eje Z óptico (profundidad / adelante). Tomamos solo [:2] para proyectar al plano XY.
    forward_xy = r_world_cam[:, 2][:2].astype(np.float64)
    # Normalizamos por si la cámara está inclinada mirando al piso o al cielo (lo que acortaría la proyección XY)
    forward_xy = normalize_xy(forward_xy, r_world_cam[:, 0][:2].astype(np.float64))
    
    # Para obtener un vector ortogonal que apunte a la izquierda en un plano 2D,
    # si "adelante" es [x, y], entonces girar +90 grados (izquierda) da matemáticamente [-y, x].
    left_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float64)
    return forward_xy, left_xy


def base_planar_axes(base_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae los vectores unitarios 2D de dirección "adelante" e "izquierda" en el plano XY 
    a partir de la matriz de postura del chasis (base_pose) usando la convención ROS.

    A diferencia de la cámara, en el chasis (ROS):
    - Columna 0 (Eje X) representa directamente "Adelante".
    - Columna 1 (Eje Y) representa directamente "Izquierda".
    Simplemente extraemos estas dos columnas, tomamos sus componentes planas XY y las normalizamos.
    """
    pose = as_matrix(base_pose, (4, 4), "base_pose")
    r_world_base = pose[:3, :3]
    forward_xy = normalize_xy(r_world_base[:, 0][:2], np.array([1.0, 0.0], dtype=np.float64))
    left_xy = normalize_xy(r_world_base[:, 1][:2], np.array([0.0, 1.0], dtype=np.float64))
    return forward_xy, left_xy


def reference_frame_from_pose(
    pose: np.ndarray,
    frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Función envolvente (wrapper) unificada que devuelve el origen [x, y] y los vectores unitarios 
    direccionales (adelante, izquierda) para cualquier postura, adaptándose a su tipo de sensor o chasis.

    ¿POR QUÉ SE USA ESTO?
    Permite que el planificador GeNIE sea "agnóstico al sensor". El algoritmo principal no 
    necesita preocuparse por si las coordenadas vienen de una cámara de profundidad, un LiDAR 
    o la odometría de las ruedas; simplemente llama a esta función diciendo frame="camera" o 
    frame="base", y recibe siempre un sistema de coordenadas estandarizado para planificar.
    """
    pose = as_matrix(pose, (4, 4), "reference_pose")
    # El origen (posición x, y del sensor en el mundo) siempre está en los primeros 2 elementos 
    # de la cuarta columna (índice 3) de la matriz homogénea 4x4.
    origin_xy = pose[:2, 3].astype(np.float64)
    frame_l = str(frame).lower()
    if frame_l == "base":
        forward_xy, left_xy = base_planar_axes(pose)
    elif frame_l == "camera":
        forward_xy, left_xy = camera_planar_axes(pose)
    else:
        raise ValueError(f"reference frame must be 'base' or 'camera', got {frame!r}") 
    return origin_xy, forward_xy, left_xy


def goal_xy_to_bev_pixel(
    goal_x_right_m: float,
    goal_y_forward_m: float,
    bev_shape: tuple[int, int],
    resolution_m: float,
) -> tuple[int, int]:
    """
    Convierte una coordenada física en metros [derecha, adelante] al índice de píxel [fila, columna] 
    dentro de la imagen de vista aérea (Bird's Eye View - BEV).

    GEOMETRÍA DEL MAPA DE VISTA AÉREA (BEV):
    Las redes neuronales y grillas de costo ven una imagen digital 2D. En esta imagen:
    - El centro inferior de la imagen representa la posición actual del robot [0.0m, 0.0m].
    - El eje vertical (Filas/Rows): Fila 0 está en el TOPE (lejos adelante). Fila h-1 está ABAJO (en el robot).
    - El eje horizontal (Columnas/Cols): Columna 0 está a la IZQUIERDA extrema. El centro es (w // 2).
    
    LA TRANSFORMACIÓN MATEMÁTICA:
    1. Distancia vertical: Dividir los metros hacia adelante entre la resolución (ej. 10m / 0.1m_por_pixel = 100 píxeles).
       Como la fila 0 es el tope y la fila h-1 es el piso, restamos esos píxeles desde el borde inferior (h - 1 - píxeles).
    2. Distancia lateral: Dividir los metros hacia la derecha entre la resolución y sumárselos 
       a la columna central (w // 2).
    3. np.clip: Si la meta física está más lejos de lo que el mapa es capaz de mostrar, forzamos 
       la coordenada para que se quede en el borde de la imagen, evitando un desbordamiento de memoria (IndexError).

    Ejemplo:
        En un mapa de 200x200 píxeles con resolución de 0.1m/px, una meta a 10m adelante 
        y 0m a la derecha caerá exactamente en la Fila 99, Columna 100.
    """
    h, w = int(bev_shape[0]), int(bev_shape[1])
    res = float(resolution_m)
    if res <= 0.0:
        raise ValueError("resolution_m must be > 0") 
    row = h - 1 - int(np.floor(float(goal_y_forward_m) / res))
    col = (w // 2) + int(np.floor(float(goal_x_right_m) / res))
    return int(np.clip(row, 0, h - 1)), int(np.clip(col, 0, w - 1))


def bev_pixel_to_xy(
    rows: np.ndarray,
    cols: np.ndarray,
    bev_shape: tuple[int, int],
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Operación inversa: Convierte arreglos de índices de píxeles [filas, columnas] del mapa BEV 
    de vuelta a coordenadas físicas continuas en metros [x_right, y_forward] del mundo real.

    ¿POR QUÉ SE SUMA "+ 0.5" EN LA FÓRMULA? (El detalle de precisión sub-píxel):
    En computación gráfica, el índice entero de un píxel (ej. fila 10, columna 20) apunta a la 
    esquina superior izquierda de ese cuadro geométrico. Si calculáramos la distancia usando el 
    número entero directo, tendríamos un error sistemático que desviaría siempre al robot hacia una esquina.
    Al sumar "+ 0.5", nos trasladamos exactamente al CENTRO GEOMÉTRICO del píxel. Así, si un píxel 
    mide 10 cm x 10 cm, el cálculo nos devuelve la coordenada física justo en el medio del cuadrito, 
    logrando trayectorias extremadamente precisas cuando se comandan los motores del vehículo.
    """
    h, w = int(bev_shape[0]), int(bev_shape[1])
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    
    # Inversión matemática del eje Y con corrección al centro del píxel (+ 0.5)
    y_forward = ((float(h - 1) - r) + 0.5) * float(resolution_m)
    # Inversión del eje X desde el centro de la imagen con corrección al centro del píxel (+ 0.5)
    x_right = ((c - float(w // 2)) + 0.5) * float(resolution_m)
    
    return x_right.astype(np.float32), y_forward.astype(np.float32)