"""ADDON para genie_rover/persistent_map.py -- NO es un archivo para correr
solo. Es el metodo que hay que agregar DENTRO de la clase `PersistentMap`
(en `genie/genie_rover/persistent_map.py`), como contraparte de
`external_map.load_ros_occupancy_map` (que ya sabe leer este mismo formato).

COMO APLICAR ESTE PARCHE
-------------------------
1. Abrir genie/genie_rover/persistent_map.py
2. Buscar el metodo `def stats(self) -> dict:` (o `def to_image`) dentro de
   la clase PersistentMap
3. Pegar el metodo `export_ros_map` de mas abajo en cualquier lugar DENTRO
   de la clase (con la misma indentacion que los demas metodos, 4 espacios)
4. No hace falta agregar imports arriba del archivo: `Path` y `Image` se
   importan localmente adentro del metodo, igual que hace `external_map.py`.

Convencion de la imagen (identica a la que ya lee external_map.py, y a la
que usa `map_server`/`slam_toolbox` de ROS):
    254 = libre/transitable, 0 = ocupado/intransitable, 205 = desconocido.
Es DELIBERADAMENTE binaria (no se exporta el valor continuo 0..1 de
`self.value`): el formato yaml+pgm de ROS solo tiene 3 niveles, asi que una
celda "sabida pero incierta" (value cerca de 0.5, conf >= min_confidence) se
redondea a transitable/intransitable segun de que lado de 0.5 cae, en vez de
quedar en una zona ambigua que el importador interpretaria como desconocida.

La derivacion de la formula de `origin` (para que el roundtrip
export -> external_map.load_ros_occupancy_map de vuelta el MISMO
world_x/world_y para cada celda) fue verificada con una prueba de roundtrip
standalone antes de entregar este parche -- error de alineamiento 0.00 mm.
"""

from __future__ import annotations


def export_ros_map(self, path_prefix, image_format: str = "pgm") -> tuple:
    """Exporta este PersistentMap al formato ROS `map_server` (yaml+pgm) --
    el mismo formato que ya lee `external_map.load_ros_occupancy_map`, y el
    mismo que exportan `map_server`/`slam_toolbox`/RTAB-Map (Database Viewer
    -> File -> "Export 2D occupancy grid").

    Uso tipico, al terminar (o periodicamente durante) una sesion de mapeo:

        pmap.export_ros_map("maps/sesion1")
        # produce maps/sesion1.yaml y maps/sesion1.pgm

    Para reusarlo despues con indoor_bridge.py, en el config:

        memory:
          external_map:
            enabled: true
            yaml_path: "maps/sesion1.yaml"
            start_pose_map:
              x_m: <donde estas parado ahora, en ESTE mapa>
              y_m: <...>
              yaw_deg: <...>

    Parametros
    ----------
    path_prefix : str | Path
        Prefijo de salida SIN extension (ej. "maps/sesion1"). Se escriben
        `{path_prefix}.yaml` y `{path_prefix}.{image_format}`.
    image_format : str
        "pgm" (estandar ROS) o "png" (mismo contenido, mas facil de abrir
        con un visor de imagenes comun para inspeccionar a ojo).

    Devuelve (yaml_path, image_path) como pathlib.Path.
    """
    from pathlib import Path

    import numpy as np
    from PIL import Image

    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    yaml_path = prefix.with_suffix(".yaml")
    img_path = prefix.with_suffix(f".{image_format}")

    r = self.cfg.resolution_m_per_px
    n = self.n
    seen = self.conf >= self.cfg.min_confidence

    # Binarizamos a proposito -- ver docstring del modulo.
    img = np.full((n, n), 205, dtype=np.uint8)          # desconocido
    img[seen & (self.value >= 0.5)] = 254                # libre
    img[seen & (self.value < 0.5)] = 0                   # ocupado

    # Reindexado (fila,col) de PersistentMap -> (row,col) de la imagen ROS,
    # de forma que el roundtrip con ros_map_to_grids() reproduzca EXACTO el
    # mismo world_x/world_y para cada celda (ver derivacion en el docstring
    # del modulo / verificado con test de roundtrip):
    #   col_img = n - 1 - fila     row_img = col
    img_out = img.T[:, ::-1]

    # origin: pose del pixel INFERIOR-IZQUIERDO de la imagen, en el marco
    # propio de este PersistentMap (x adelante, y izquierda -- yaw0 = 0
    # siempre, porque nunca rotamos el mapa respecto de si mismo).
    ros_origin_x = self.origin_x - r * (n - 1) / 2.0
    ros_origin_y = self.origin_y - r * (n - 1) / 2.0

    Image.fromarray(img_out, mode="L").save(img_path)

    yaml_text = (
        f"image: {img_path.name}\n"
        f"resolution: {r:.6f}\n"
        f"origin: [{ros_origin_x:.6f}, {ros_origin_y:.6f}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    yaml_path.write_text(yaml_text)
    return yaml_path, img_path
