"""Importar un mapa de ocupacion generado por OTRA herramienta (ROS
map_server / RTAB-Map) como PersistentMap de genie_rover.

Formato soportado: el estandar de ROS `map_server` (un `nav_msgs/OccupancyGrid`
guardado a disco), que es tambien lo que exporta RTAB-Map:

    mi_mapa.yaml:
        image: mi_mapa.pgm
        resolution: 0.050000        # metros/pixel
        origin: [x0, y0, yaw0]      # pose del pixel INFERIOR-IZQUIERDO, en
                                    # el marco propio del mapa (yaw0 casi
                                    # siempre 0 en los mapas que arma ROS)
        negate: 0
        occupied_thresh: 0.65
        free_thresh: 0.196

    mi_mapa.pgm (o .png): escala de grises. Blanco = libre, negro = ocupado,
        gris = desconocido — la misma convencion que usa `map_server` de ROS
        y el export de grilla 2D de RTAB-Map (Database Viewer -> File ->
        "Export 2D occupancy grid", o publicando /rtabmap/grid_map y
        corriendo `rosrun map_server map_saver`).

Si tenes un `.db` de RTAB-Map en vez de este par de archivos, exportale
primero la grilla 2D con la propia herramienta (arriba) — este modulo NO
lee la base de datos de RTAB-Map directamente, seria traer sqlite3 + su
esquema propio para un caso que RTAB-Map ya resuelve con un boton.

LIMITACION IMPORTANTE — por que esto no es "cargar y listo":

El mapa importado queda fijo en el marco de coordenadas que tenia en
ROS/RTAB-Map. `odometry.py` (Pose) siempre arranca asumiendo que el robot
esta en el origen del marco en el que bootea. Sin relocalizacion (reconocer
"donde estoy" comparando la imagen contra el mapa — este repo NO la
implementa), la unica forma de que el mapa importado quede alineado con lo
que la camara ve de verdad es decirle a IndoorBridge, A MANO, en que pose
(x, y, yaw) de ESE mapa estas parando el robot en el instante exacto de
arrancar `indoor_bridge.py` (`memory.external_map.start_pose_map` del
config). Si el robot arranca en otro lugar o mirando para otro lado, el
mapa importado queda rotado/corrido respecto de la realidad — y el planner
tomando decisiones sobre un mapa mal alineado es PEOR que arrancar con el
mapa vacio. Marca el punto de arranque en el piso (cinta, una silla, lo que
sea) y medi su pose en el mapa una sola vez con cuidado.

Una vez cargado, el mapa importado se sigue actualizando con lo que ve
SAM-TP en vivo (mismo `PersistentMap.integrate`, sin cambios): una celda
importada que la camara vuelve a observar se mezcla con la lectura fresca
al peso de siempre (`update_weight`), y una celda importada que nunca se
vuelve a ver decae hacia "desconocido" con el tiempo (`decay_per_s`) igual
que cualquier otra — si el edificio cambio desde que se genero el mapa
(muebles movidos, puerta cerrada), esa certeza vieja se va desvaneciendo
sola en vez de quedar pegada para siempre.

Autoprueba (arma un mapa ROS sintetico en un directorio temporal, no
necesita ningun archivo real ni robot):
    python -m genie_rover.indoor.external_map
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..persistent_map import MapConfig, PersistentMap


@dataclass
class RosMapMeta:
    image_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    negate: bool
    occupied_thresh: float
    free_thresh: float


def load_ros_map_yaml(yaml_path: str) -> RosMapMeta:
    import yaml

    p = Path(yaml_path)
    data = yaml.safe_load(p.read_text())
    origin = data.get("origin", [0.0, 0.0, 0.0])
    image_rel = data["image"]
    image_path = Path(image_rel)
    if not image_path.is_absolute():
        image_path = p.parent / image_path
    return RosMapMeta(
        image_path=image_path,
        resolution=float(data["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]) if len(origin) > 2 else 0.0,
        negate=bool(data.get("negate", 0)),
        occupied_thresh=float(data.get("occupied_thresh", 0.65)),
        free_thresh=float(data.get("free_thresh", 0.196)),
    )


def ros_map_to_grids(meta: RosMapMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convierte la imagen del mapa a (value, known, world_x, world_y).

    Las cuatro son HxW. value/known usan la misma convencion que
    PersistentMap (1=transitable, 0=intransitable, valido solo donde
    known=True). world_x/world_y son las coordenadas de cada pixel en el
    marco propio del mapa (el de `origin` en el yaml).
    """
    from PIL import Image

    if not meta.image_path.is_file():
        raise FileNotFoundError(
            f"No encuentro la imagen del mapa en {meta.image_path} "
            f"(referenciada por el .yaml). Revisa que 'image:' en el yaml "
            "apunte al archivo correcto."
        )
    img = np.asarray(Image.open(meta.image_path).convert("L"), dtype=np.float32)
    h, w = img.shape

    p_occupied = (255.0 - img) / 255.0 if not meta.negate else img / 255.0

    known = (p_occupied > meta.occupied_thresh) | (p_occupied < meta.free_thresh)
    value = np.full((h, w), 0.5, dtype=np.float32)
    value[p_occupied > meta.occupied_thresh] = 0.0
    value[p_occupied < meta.free_thresh] = 1.0

    rows, cols = np.indices((h, w), dtype=np.float64)
    # Offset del pixel respecto de la esquina inferior-izquierda, en los ejes
    # PROPIOS de la imagen (x = derecha, y = arriba) antes de rotar por yaw0.
    dx_img = (cols + 0.5) * meta.resolution
    dy_img = (h - 1 - rows + 0.5) * meta.resolution

    c, s = math.cos(meta.origin_yaw), math.sin(meta.origin_yaw)
    world_x = meta.origin_x + c * dx_img - s * dy_img
    world_y = meta.origin_y + s * dx_img + c * dy_img

    return value, known, world_x.astype(np.float32), world_y.astype(np.float32)


def build_persistent_map_from_grids(
    value: np.ndarray, known: np.ndarray, world_x: np.ndarray, world_y: np.ndarray,
    resolution_m_per_px: float, margin_m: float = 1.5,
    update_weight: float = 0.45, decay_per_s: float = 0.08,
    recenter_margin_m: float = 1.5, min_confidence: float = 0.15,
) -> PersistentMap:
    """Arma un PersistentMap NUEVO ya poblado, centrado en el area del mapa
    importado (con `margin_m` de margen para no recentrar de entrada)."""
    x_min, x_max = float(world_x.min()), float(world_x.max())
    y_min, y_max = float(world_y.min()), float(world_y.max())
    size_m = max(x_max - x_min, y_max - y_min) + 2.0 * margin_m

    pmap = PersistentMap(MapConfig(
        size_m=size_m, resolution_m_per_px=resolution_m_per_px,
        update_weight=update_weight, decay_per_s=decay_per_s,
        recenter_margin_m=recenter_margin_m, min_confidence=min_confidence,
    ))
    pmap.origin_x = 0.5 * (x_min + x_max)
    pmap.origin_y = 0.5 * (y_min + y_max)

    r = pmap.cfg.resolution_m_per_px
    n = pmap.n
    f = np.rint(n / 2 - (world_x - pmap.origin_x) / r).astype(int)
    cc = np.rint(n / 2 - (world_y - pmap.origin_y) / r).astype(int)

    dentro = (f >= 0) & (f < n) & (cc >= 0) & (cc < n)
    f, cc, val, kn = f[dentro], cc[dentro], value[dentro], known[dentro]

    # Si varios pixeles del mapa importado (resolucion mas fina que la
    # nuestra) caen en la misma celda, promediamos en vez de dejar que gane
    # el ultimo — mismo criterio que PersistentMap.integrate.
    idx_conocidos = kn
    f_k, c_k, v_k = f[idx_conocidos], cc[idx_conocidos], val[idx_conocidos]
    if f_k.size:
        plano = f_k * n + c_k
        orden = np.argsort(plano)
        plano, v_k = plano[orden], v_k[orden]
        bordes = np.concatenate([[0], np.nonzero(np.diff(plano))[0] + 1, [len(plano)]])
        idx_unicos = plano[bordes[:-1]]
        medias = np.add.reduceat(v_k, bordes[:-1]) / np.diff(bordes)
        fu, cu = idx_unicos // n, idx_unicos % n
        pmap.value[fu, cu] = medias
        pmap.conf[fu, cu] = 1.0

    return pmap


def load_ros_occupancy_map(
    yaml_path: str, resolution_m_per_px: float | None = None, margin_m: float = 1.5,
    update_weight: float = 0.45, decay_per_s: float = 0.08,
    recenter_margin_m: float = 1.5, min_confidence: float = 0.15,
) -> PersistentMap:
    """Punto de entrada unico: .yaml del mapa -> PersistentMap listo para
    reemplazar el que arma Bridge.__init__ vacio.

    Si `resolution_m_per_px` es None, usa la resolucion nativa del archivo
    importado (no la de `projection.resolution_m_per_px` del config del
    rover) — lo mas comun es que un mapa ROS este a 0.05 m/px y el BEV del
    rover a 0.03: no hace falta que coincidan, PersistentMap.extract_bev ya
    resamplea al recortar la ventana que le pide el planner.
    """
    meta = load_ros_map_yaml(yaml_path)
    value, known, world_x, world_y = ros_map_to_grids(meta)
    res = float(resolution_m_per_px) if resolution_m_per_px else meta.resolution
    pmap = build_persistent_map_from_grids(
        value, known, world_x, world_y, resolution_m_per_px=res, margin_m=margin_m,
        update_weight=update_weight, decay_per_s=decay_per_s,
        recenter_margin_m=recenter_margin_m, min_confidence=min_confidence,
    )
    st = pmap.stats()
    print(f"[external_map] cargado {yaml_path}: {st['celdas_vistas']} celdas conocidas, "
          f"mapa de {pmap.cfg.size_m:.1f} m de lado a {res:.3f} m/px, "
          f"centro del mapa importado en ({pmap.origin_x:+.2f}, {pmap.origin_y:+.2f})")
    return pmap


# --------------------------------------------------------------------- pruebas

def _write_synthetic_ros_map(tmp_dir: Path) -> Path:
    """Sala rectangular de 4x3 m: paredes negras, interior blanco, una franja
    gris (desconocida) cerca de un costado. resolucion 0.05 m/px."""
    from PIL import Image

    res = 0.05
    w_m, h_m = 4.0, 3.0
    w, h = int(w_m / res), int(h_m / res)

    img = np.full((h, w), 254, dtype=np.uint8)      # blanco = libre
    img[0:2, :] = 0; img[-2:, :] = 0                # paredes arriba/abajo
    img[:, 0:2] = 0; img[:, -2:] = 0                # paredes izq/derecha
    img[:, w // 2: w // 2 + 6] = 205                # franja desconocida

    img_path = tmp_dir / "mapa_sintetico.pgm"
    Image.fromarray(img, mode="L").save(img_path)

    yaml_path = tmp_dir / "mapa_sintetico.yaml"
    yaml_path.write_text(
        f"image: {img_path.name}\n"
        f"resolution: {res}\n"
        f"origin: [-2.0, -1.5, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    return yaml_path


def _self_test() -> None:
    import tempfile

    from ..odometry import Pose

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        yaml_path = _write_synthetic_ros_map(tmp)

        print("=== load_ros_map_yaml ===")
        meta = load_ros_map_yaml(str(yaml_path))
        print(f"  {meta}")
        assert abs(meta.resolution - 0.05) < 1e-9
        assert meta.origin_x == -2.0 and meta.origin_y == -1.5

        print("\n=== ros_map_to_grids: geometria ===")
        value, known, wx, wy = ros_map_to_grids(meta)
        print(f"  forma: {value.shape}, conocidos: {int(known.sum())}/{known.size}")
        # Esquina inferior-izquierda de la imagen (ultima fila, primera
        # columna) debe caer cerca de origin_x/origin_y.
        assert abs(wx[-1, 0] - (meta.origin_x + 0.5 * meta.resolution)) < 1e-3
        assert abs(wy[-1, 0] - (meta.origin_y + 0.5 * meta.resolution)) < 1e-3
        # El centro de la sala (lejos de paredes y de la franja gris) es libre.
        centro_fila, centro_col = value.shape[0] // 2, 10
        assert known[centro_fila, centro_col] and value[centro_fila, centro_col] == 1.0
        # Una pared es intransitable (columna lejos de la franja gris, que
        # tambien tapa un pedazo de la pared superior donde se superponen).
        col_pared = value.shape[1] // 4
        assert known[0, col_pared] and value[0, col_pared] == 0.0
        # La franja gris es desconocida.
        col_franja = value.shape[1] // 2 + 3
        assert not known[centro_fila, col_franja]

        print("\n=== load_ros_occupancy_map: PersistentMap resultante ===")
        pmap = load_ros_occupancy_map(str(yaml_path), margin_m=1.0)
        print(f"  {pmap.stats()}")
        # Consultamos el centro de la sala en coordenadas del MAPA (no del
        # robot): origin=(-2,-1.5), sala de 4x3 -> centro en (0, 0).
        bev, obs = pmap.extract_bev(
            Pose(0.0, 0.0, 0.0), forward_range_m=0.5, side_range_m=0.5, out_h=10, out_w=10)
        assert obs.sum() > 0, "no debería quedar vacío justo en el centro de la sala"
        assert float(bev[obs.astype(bool)].mean()) > 0.7, "el centro de la sala debería leer transitable"

        print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
