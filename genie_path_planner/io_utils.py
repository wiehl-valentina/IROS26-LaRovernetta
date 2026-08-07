from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """
    Carga un archivo de configuración (en formato JSON o YAML) desde el disco y devuelve 
    su contenido analizado junto con la ruta absoluta de la carpeta que lo contiene.
    
    CONCEPTOS CLAVE:
    - Path.expanduser().resolve(): Transforma rutas relativas o que usan '~' (home del usuario) 
      en rutas absolutas completas del sistema (ej. '/home/usuario/config.yaml').
    - Importación dinámica (import yaml): PyYAML solo se importa si el archivo tiene extensión 
      .yaml o .yml. Esto evita que el programa colapse exigiendo dependencias innecesarias si el 
      usuario solo planea usar archivos JSON.
    - Manejo de vacíos (data is None): Si un archivo de configuración está en blanco, evita 
      que el programa falle más adelante convirtiéndolo en un diccionario vacío {}.
    
    Args:
        path: Ruta (como string o objeto Path) al archivo de configuración.
    
    Returns:
        tuple[dict[str, Any], Path]: Una tupla donde el primer elemento es el diccionario con 
        los datos del archivo, y el segundo elemento es un objeto Path que apunta a la 
        carpeta contenedora (parent) para poder resolver otras rutas relativas más adelante.
    """
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    suffix = cfg_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyYAML is required to load YAML configs. Install pyyaml.") from exc
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config suffix {suffix!r}; use .yaml, .yml, or .json")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data, cfg_path.parent


def deep_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Extrae un valor de un diccionario altamente anidado utilizando notación de puntos (dot notation).
    
    CONCEPTOS CLAVE:
    - Notación de puntos (dot notation): En robótica y configuraciones complejas, los datos suelen 
      tener muchos niveles. En lugar de escribir código propenso a errores como:
      `data.get("robot", {}).get("camera", {}).get("fov", 90)`
      esta función permite usar una cadena simple y limpia: `deep_get(data, "robot.camera.fov")`.
    - Recorrido seguro: Separa la cadena por puntos (split) e itera nivel por nivel. Si en algún 
      momento la clave no existe o el nodo actual no es un diccionario, aborta la búsqueda y 
      devuelve el valor 'default' sin lanzar errores de tipo KeyError.
    
    Args:
        data: Diccionario principal donde se realizará la búsqueda.
        path: Cadena de texto con las claves separadas por puntos (ej. "nivel1.nivel2.clave").
        default: Valor a devolver si la ruta no existe en el diccionario. Por defecto es None.
    
    Returns:
        El valor encontrado en la ruta especificada, o el valor por defecto si la ruta es inválida.
    """
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(value: str | Path | None, base_dir: Path, repo_root: Path | None = None) -> Path | None:
    """
    Resuelve una ruta relativa para convertirla en una ruta absoluta validada, buscando 
    el archivo en diferentes directorios de respaldo (fallback).
    
    CONCEPTOS CLAVE:
    - Rutas Absolutas vs Relativas: Las configuraciones a menudo referencian otros archivos 
      (ej. "calibracion.npy"). Si la ruta ya es absoluta (ej. empieza con '/'), se respeta. 
      Si es relativa, se necesita saber "relativa a qué".
    - Resolución en cascada: Primero intenta resolver la ruta asumiendo que el archivo está 
      dentro del directorio base (base_dir, generalmente la carpeta donde está el archivo de configuración). 
      Si el archivo existe allí o si no hay un directorio raíz de repositorio (repo_root), 
      se asume esa ruta. De lo contrario, se busca a partir del directorio repo_root.
    
    Args:
        value: La ruta original a resolver (puede ser un string, Path o None).
        base_dir: Directorio prioritario para resolver rutas relativas.
        repo_root: Directorio secundario (raíz del proyecto) para buscar si falla el primero.
    
    Returns:
        La ruta absoluta resuelta como objeto Path, o None si la entrada era inválida/nula.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists() or repo_root is None:
        return candidate
    return (repo_root / path).resolve()


def load_matrix(path_or_value: Any, shape: tuple[int, int], base_dir: Path, repo_root: Path | None, name: str) -> np.ndarray:
    """
    Carga y formatea una matriz matemática garantizando que tenga la forma (shape) exacta requerida. 
    Puede cargarla desde un archivo en disco (.npy, .json, .yaml) o asimilarla si ya es una lista en memoria.
    
    CONCEPTOS CLAVE:
    - np.ndarray: Arreglo multidimensional de NumPy, el estándar para cálculos algebraicos.
    - Reformateo (reshape): En los archivos YAML/JSON es común escribir las matrices de rotación como 
      una lista plana unidimensional (ej. 16 elementos seguidos). Si la función exige una matriz 2D 
      (ej. forma 4x4), la condición `arr.size == shape[0] * shape[1]` la detecta y convierte la lista 
      plana a su correcta distribución bidimensional usando .reshape(shape).
    - Tipo de dato científico: Fuerza la conversión a float64 (coma flotante de doble precisión) 
      para evitar pérdidas de precisión catastróficas en cálculos geométricos y trigonométricos.
    
    Args:
        path_or_value: Ruta al archivo (str/Path) que contiene la matriz, o los datos directos en memoria.
        shape: Tupla que indica las dimensiones geométricas exactas esperadas (ej. (4, 4)).
        base_dir: Directorio base para resolver la ruta mediante resolve_path.
        repo_root: Directorio raíz del repositorio para resolución secundaria.
        name: Nombre identificador usado exclusivamente para mostrar mensajes de error descriptivos.
    
    Returns:
        Un arreglo bidimensional de NumPy (np.ndarray) con la forma exacta y tipo float64.
    """
    if isinstance(path_or_value, (str, Path)):
        path = resolve_path(path_or_value, base_dir=base_dir, repo_root=repo_root)
        if path is None or not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix == ".json":
            arr = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError("PyYAML is required to load YAML matrix files. Install pyyaml.") from exc
            arr = np.asarray(yaml.safe_load(path.read_text(encoding="utf-8")), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported {name} file suffix {suffix!r}: {path}")
    else:
        arr = np.asarray(path_or_value, dtype=np.float64)
    if arr.shape == shape:
        return arr.astype(np.float64)
    if arr.size == shape[0] * shape[1]:
        return arr.reshape(shape).astype(np.float64)
    raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")


def load_rgb_image(path: str | Path) -> np.ndarray:
    """
    Carga una imagen fotográfica desde el disco y la estandariza a un arreglo NumPy de color RGB.
    
    CONCEPTOS CLAVE:
    - PIL (Pillow): Librería estándar y eficiente en Python para decodificación de imágenes.
    - .convert("RGB"): Elimina variaciones indeseadas en el archivo de entrada. Por ejemplo, elimina 
      posibles canales de transparencia (el canal Alfa en imágenes RGBA) y expande imágenes en 
      escala de grises a 3 canales. Esto asegura que las redes neuronales siempre reciban exactamente 
      3 canales de profundidad (Rojo, Verde, Azul).
    - uint8 (Unsigned Integer 8-bit): Tipo de dato que almacena valores enteros positivos del 0 al 255. 
      Es el estándar universal de memoria para representar la intensidad de luz de un píxel.
    
    Args:
        path: Ruta al archivo de imagen.
    
    Returns:
        Un arreglo NumPy tridimensional (Alto x Ancho x 3 canales) de tipo uint8.
    """
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_depth_m(path: str | Path, unit: str = "m") -> np.ndarray:
    """
    Carga un mapa de profundidad (Depth Map) desde el disco y normaliza sus distancias para que 
    siempre estén expresadas en metros utilizando notación decimal (float32).
    
    CONCEPTOS CLAVE:
    - Mapas de Profundidad (Depth Maps): Son imágenes tomadas por sensores (como LiDAR o RealSense) 
      donde cada píxel no representa un color, sino la distancia física hacia el objeto enfocado.
    - Manejo de canales tridimensionales (depth[..., 0]): A veces los mapas de profundidad se guardan 
      por error en formato RGB de 3 canales duplicando la misma información. Si se detectan 3 dimensiones, 
      el código recorta únicamente el primer canal para trabajar con un mapa plano 2D.
    - Conversión métrica matemática: Las cámaras suelen codificar la distancia en números enteros 
      expresados en milímetros (mm) para ahorrar espacio (ej. 1500 mm). Esta función chequea si 
      la unidad indicada era 'mm' y divide matricialmente todo por 1000.0, garantizando que el resto 
      del algoritmo de navegación siempre opere en metros exactos.
    
    Args:
        path: Ruta al archivo de profundidad (soporta matrices rápidas .npy o archivos de imagen).
        unit: Unidad original del archivo, por defecto 'm' (metros). Soporta 'mm' (milímetros).
    
    Returns:
        Una matriz plana de NumPy (2D) tipo float32, donde cada celda es una distancia en metros.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Depth file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".npy":
        depth = np.asarray(np.load(p))
    else:
        depth = np.asarray(Image.open(p))
        if depth.ndim == 3:
            depth = depth[..., 0]

    depth = depth.astype(np.float32)
    unit_l = str(unit).lower()
    if unit_l in {"m", "meter", "meters"}:
        return depth
    if unit_l in {"mm", "millimeter", "millimeters"}:
        return depth / 1000.0
    raise ValueError(f"Unsupported depth unit {unit!r}; use 'm' or 'mm'")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """
    Guarda una estructura de datos de Python en un archivo con formato JSON en el disco local.
    
    CONCEPTOS CLAVE:
    - indent=2: Aplica saltos de línea y 2 espacios de sangría al texto generado. Aunque ocupa 
      ligeramente más bytes de disco, permite que los desarrolladores y humanos puedan leer el archivo fácilmente.
    - ensure_ascii=True: Si el diccionario tiene letras con tildes (ej. 'á') u otros caracteres especiales, 
      los formatea automáticamente a su representación estándar escapada de ASCII. Esto evita corrupción de datos 
      al transferir los archivos entre diferentes sistemas operativos o redes.
    
    Args:
        path: Ruta y nombre del archivo destino a crear/sobrescribir.
        payload: El diccionario o los datos nativos de Python que se van a guardar.
    """
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def to_builtin(value: Any) -> Any:
    """
    Filtro recursivo (sanitizador) que recorre estructuras complejas y convierte todos los objetos 
    matemáticos científicos de NumPy a tipos de datos básicos y nativos de Python.
    
    CONCEPTOS CLAVE:
    - El choque de serialización (JSON Crashing): La librería json estándar de Python está diseñada 
      para guardar enteros (int), decimales (float) y listas nativas. Si intentas pasarle un 
      arreglo matemático (np.ndarray) o incluso un decimal específico de NumPy (como np.float32), 
      el programa colapsará violentamente lanzando un TypeError ("Object of type X is not JSON serializable").
    - Recursividad estructural: La función se llama a sí misma para escanear y desarmar cada elemento anidado 
      dentro de diccionarios, listas o tuplas, buscando escondites de tipos problemáticos.
    - Extracción de nativos: Convierte los np.ndarray enteros usando `.tolist()` y transforma 
      las variables numéricas escalares (np.generic) usando `.item()` para obtener su equivalente en Python estándar.
    
    Args:
        value: El objeto original (o colección de objetos) que será inspeccionado y convertido.
    
    Returns:
        Un objeto idéntico en estructura, pero construido en su 100% con tipos de datos nativos de Python.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    return value