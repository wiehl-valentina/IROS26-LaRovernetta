#!/usr/bin/env python3
"""test_recovery_turn_latency.py — Validación de fin de giro de recovery en lazo cerrado con latencia de arranque.

Verifica:
1. Caso Corrida 2 (latencia 1.8s, objetivo 90°): el giro NO se corta prematuramente
   tras 2 comandos, sino que espera a superar la latencia y alcanza el ángulo real objetivo.
2. Criterio de corte por despeje de mapa: inhibido durante la latencia de arranque y
   habilitado únicamente tras una rotación física real significativa (>= 20°).
3. Timeout de seguridad: si el robot está bloqueado mecánicamente (0° de rotación real),
   termina limpiamente al expirar el timeout configurado.
"""

import math
import time
import types
import unittest
import numpy as np

from genie_rover.bridge import Bridge
from genie_rover.navigation import DriveCommand, PathFollower
from genie_rover.odometry import Pose
from genie_rover.persistent_map import MapConfig, PersistentMap


class SimulatedRoverOdo:
    """Odometría simulada que modela latencia de arranque de hardware
    y luego gira físicamente a una tasa real (deg_per_s)."""
    def __init__(self, startup_latency_s: float = 0.2, turn_rate_deg_s: float = 100.0):
        self.startup_latency_s = startup_latency_s
        self.turn_rate_deg_s = turn_rate_deg_s
        self.t0 = time.time()
        self._pose = Pose(0.0, 0.0, 0.0)

    @property
    def pose(self):
        return self._pose

    def update(self, _raw, *args, **kwargs):
        elapsed = time.time() - self.t0
        if elapsed > self.startup_latency_s:
            # Comenzó el movimiento físico real
            dt_motion = elapsed - self.startup_latency_s
            theta_deg = -(dt_motion * self.turn_rate_deg_s)
            self._pose = Pose(0.0, 0.0, math.radians(theta_deg))
        return self._pose


class TestRecoveryTurnLatency(unittest.TestCase):
    def setUp(self):
        self.bridge = types.SimpleNamespace()
        self.bridge.follower = PathFollower(angular_sign=-1.0, turn_speed=0.35, max_linear=0.35, max_angular=0.45)
        self.bridge.recovery_turn_speed = 0.45
        self.bridge.recovery_step_deg = 45.0
        self.bridge.recovery_deg_per_s = 100.0
        self.bridge.recovery_startup_latency_s = 0.2
        self.bridge.recovery_turn_tolerance_deg = 15.0
        self.bridge.recovery_turn_timeout_s = 5.0
        self.bridge.recovery_min_cobertura_pct = 25.0
        self.bridge.retroceso_min_libre_pct = 55.0
        self.bridge.heading_search_radius_m = 2.0
        self.bridge._stop_requested = False
        self.bridge.sent_commands = []
        self.bridge.send = lambda cmd: self.bridge.sent_commands.append(cmd)

        self.bridge.pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
        self.bridge._is_tilt_too_steep_for_recovery = lambda: (False, "")
        # Por defecto frente bloqueado para evaluar lazo cerrado puro
        self.bridge._map_free_and_coverage = lambda pose, h, r: (10.0, 50.0)

        self.bridge._girar_hacia = types.MethodType(Bridge._girar_hacia, self.bridge)
        self.bridge.client = types.SimpleNamespace(
            telemetry=lambda: types.SimpleNamespace(raw={})
        )

    def test_turn_reaches_full_target_in_closed_loop(self):
        """Con frente bloqueado en mapa, el giro se mantiene hasta alcanzar la meta real (>=75° para 90°)."""
        self.bridge.odometry = SimulatedRoverOdo(startup_latency_s=0.2, turn_rate_deg_s=100.0)

        self.bridge._girar_hacia(90.0)

        # Debe haber enviado múltiples comandos superando la latencia de arranque
        self.assertTrue(len(self.bridge.sent_commands) >= 3,
                        f"Debería enviar múltiples comandos superando latencia (envió {len(self.bridge.sent_commands)})")
        self.assertEqual(self.bridge.sent_commands[-1].reason, "fin del giro")

        girado_final_deg = abs(math.degrees(self.bridge.odometry.pose.theta))
        self.assertGreaterEqual(girado_final_deg, 75.0,
                                f"Debería haber alcanzado >= 75° reales, pero alcanzó {girado_final_deg:.1f}°")

    def test_early_map_clearing_only_after_latency_and_min_turn(self):
        """Si el mapa detecta frente libre, corta antes de 90°, pero NUNCA en la ventana de latencia (0°)."""
        # Mapa detecta frente libre
        self.bridge._map_free_and_coverage = lambda pose, h, r: (90.0, 50.0)
        self.bridge.odometry = SimulatedRoverOdo(startup_latency_s=0.2, turn_rate_deg_s=100.0)

        self.bridge._girar_hacia(90.0)

        girado_final_deg = abs(math.degrees(self.bridge.odometry.pose.theta))
        # Debe haber cortado con giro real significativo (>= 20°), no en 0°
        self.assertGreaterEqual(girado_final_deg, 20.0,
                                f"El corte anticipado por mapa debe requerir giro real >= 20°, obtuvo {girado_final_deg:.1f}°")

    def test_timeout_safety_when_mechanically_stuck(self):
        """Si el rover está trabado contra una piedra (0° de rotación real),
        debe cortar por timeout de seguridad sin colgarse."""
        self.bridge.recovery_startup_latency_s = 0.1
        self.bridge.recovery_turn_timeout_s = 0.4
        self.bridge.recovery_deg_per_s = 50.0
        self.bridge.odometry = types.SimpleNamespace(
            pose=Pose(0.0, 0.0, 0.0),
            update=lambda *a, **kw: Pose(0.0, 0.0, 0.0)
        )

        t0 = time.time()
        self.bridge._girar_hacia(90.0)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 1.5, "El timeout debe cortar en ~0.4s, no colgarse")
        self.assertEqual(self.bridge.sent_commands[-1].reason, "fin del giro")


if __name__ == "__main__":
    unittest.main()
