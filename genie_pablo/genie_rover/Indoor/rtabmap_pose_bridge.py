"""Puente liviano hacia la correccion de pose de RTAB-Map (TF `map -> base_link`).

genie_rover normalmente calcula su propia pose por dead-reckoning.
(rueda+giro+GPS, ver `odometry.py`). Este modulo permite REEMPLAZAR esa pose,
cuadro a cuadro, por la que resulta de componer:

    map -> odom   (correccion de RTAB-Map: cierre de bucles visual)
    odom -> base_link  (odometria cruda, la misma rueda+giro de siempre)

`tf2_ros` ya hace esa composicion por vos: pedirle la transformacion
`map -> base_link` directamente devuelve el resultado ya compuesto.

Requiere que, EN PARALELO, este corriendo la sesion ROS2 de mapeo
(`Indoor_Instalacion_SDK_SLAM/ros2/mapping/rtabmap_mapping.launch.py`
(instalado como `earth-rovers-sdk/examples/ros2/mapping/`; ver `./rover_launch.sh sync-ros2`)), que es quien publica
esas TF. Si no esta corriendo (o rclpy/tf2_ros no estan instalados en este
entorno), este modulo lo dice explicitamente al construirse -- quien lo usa
(`map_session.py`) atrapa esa excepcion y sigue con dead-reckoning solo,
nunca falla en silencio ni inventa una pose.

Autoprueba: no tiene (necesita ROS2 real corriendo). Se prueba de punta a
punta corriendo `map_session.py` con `mapping.rtabmap_correction.enabled:
true` mientras `rtabmap_mapping.launch.py` esta activo.
"""

from __future__ import annotations

import math
import threading

from ..odometry import Pose


class RtabmapPoseBridge:
    def __init__(self, map_frame: str = "map", base_frame: str = "base_link",
                lookup_timeout_s: float = 0.2):
        try:
            import rclpy
            from rclpy.duration import Duration
            from rclpy.node import Node
            from tf2_ros import Buffer, TransformListener
        except ImportError as exc:
            raise RuntimeError(
                "rclpy/tf2_ros no estan instalados en este entorno de Python -- "
                "la correccion de RTAB-Map necesita correr donde SI haya un "
                "ROS2 completo (mismo venv/entorno que earth_rover_bridge.py). "
                f"Error original: {exc}"
            ) from exc

        self.map_frame = map_frame
        self.base_frame = base_frame
        self._timeout_s = float(lookup_timeout_s)

        self._rclpy = rclpy
        self._Duration = Duration
        self._owns_rclpy_init = not rclpy.ok()
        if self._owns_rclpy_init:
            rclpy.init(args=None)

        self._node = Node("genie_rtabmap_pose_bridge")
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self._node)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        self.lookups_ok = 0
        self.lookups_failed = 0

    def _spin(self) -> None:
        while not self._stop.is_set() and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.1)

    def get_corrected_pose(self) -> Pose | None:
        """Pose(x, y, theta) de la TF `map -> base_link` mas reciente, o
        None si todavia no hay una disponible (por ejemplo, RTAB-Map recien
        arranca y no publico TF todavia -- en ese caso quien llama debe
        seguir usando la odometria cruda para ese frame)."""
        try:
            tf = self._buffer.lookup_transform(
                self.map_frame, self.base_frame, self._rclpy.time.Time(),
                timeout=self._Duration(seconds=self._timeout_s),
            )
        except Exception:
            self.lookups_failed += 1
            return None

        self.lookups_ok += 1
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return Pose(t.x, t.y, yaw)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy_init and self._rclpy.ok():
            self._rclpy.shutdown()
