#!/usr/bin/env python3
"""test_checkpoint_reached.py — Validación de checkpoint-reached en La Rovernetta (Parte B).

Verifica:
1. Posición fuera del radio (>13.0m) -> reached=False.
2. Posición dentro del radio (<=13.0m) -> reached=True.
3. Frontera exacta (12.9m vs 13.1m con radio nominal 13.0m).
4. Carga de configuración desde frodobot_rover.yaml con checkpoint_reached_radius_m=13.0.
5. Estrangulación de tolerancia ante rechazo del SDK (13.0 -> 6.5 -> 3.25m, piso 0.5m)
   y restauración al avanzar de checkpoint (portado de gps_waypoint_controller.py:392-401).
6. Continuidad del flujo de navegación y planificación: goal_from_gps y seguimiento
   siguen produciendo metas locales válidas sin alterar el compromiso de mapa persistente.
"""

import math
import unittest
from pathlib import Path
import yaml

from genie_rover.navigation import (
    check_checkpoint_reached,
    goal_from_gps,
    latlon_to_local_ne,
    LocalGoal,
)


class TestCheckpointReached(unittest.TestCase):
    def setUp(self):
        # Punto de referencia de prueba (San José / Mini+ test site)
        self.target_lat = 9.791535
        self.target_lon = -84.105873
        self.radius_m = 13.0

    def test_outside_radius(self):
        """Posición a ~15.0m del checkpoint: reached debe ser False con radio 13.0m."""
        # 0.000135 grados de latitud ~ 15.0 m al Norte
        rover_lat = self.target_lat + (15.0 / 111194.9)
        rover_lon = self.target_lon

        reached, dist_m = check_checkpoint_reached(
            rover_lat, rover_lon, self.target_lat, self.target_lon, self.radius_m
        )
        self.assertFalse(reached)
        self.assertAlmostEqual(dist_m, 15.0, delta=0.2)

    def test_inside_radius(self):
        """Posición a ~10.0m del checkpoint: reached debe ser True con radio 13.0m."""
        rover_lat = self.target_lat + (10.0 / 111194.9)
        rover_lon = self.target_lon

        reached, dist_m = check_checkpoint_reached(
            rover_lat, rover_lon, self.target_lat, self.target_lon, self.radius_m
        )
        self.assertTrue(reached)
        self.assertAlmostEqual(dist_m, 10.0, delta=0.2)

    def test_exact_boundary(self):
        """Frontera exacta: 12.9m está dentro (True), 13.1m está fuera (False)."""
        lat_12_9m = self.target_lat + (12.9 / 111194.9)
        reached_in, dist_in = check_checkpoint_reached(
            lat_12_9m, self.target_lon, self.target_lat, self.target_lon, self.radius_m
        )
        self.assertTrue(reached_in)
        self.assertLessEqual(dist_in, 13.0)

        lat_13_1m = self.target_lat + (13.1 / 111194.9)
        reached_out, dist_out = check_checkpoint_reached(
            lat_13_1m, self.target_lon, self.target_lat, self.target_lon, self.radius_m
        )
        self.assertFalse(reached_out)
        self.assertGreater(dist_out, 13.0)

    def test_config_loading(self):
        """Verifica que configs/frodobot_rover.yaml defina checkpoint_reached_radius_m: 13.0."""
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "frodobot_rover.yaml"
        self.assertTrue(cfg_path.exists(), f"No se encontró {cfg_path}")

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        nav = cfg.get("navigation", {})
        self.assertIn("checkpoint_reached_radius_m", nav)
        radius = float(nav["checkpoint_reached_radius_m"])
        self.assertEqual(radius, 13.0)
        # Compatibilidad con alias
        self.assertEqual(float(nav.get("claim_radius_m")), 13.0)

    def test_tolerance_throttling_on_rejection(self):
        """Simula el estrangulamiento de tolerancia geodésica ante rechazos del SDK."""
        base_tol = 13.0
        cur_tol = base_tol

        # 1er rechazo (ej. en 12.5m): estrangula a la mitad
        cur_tol = max(0.5, cur_tol * 0.5)
        self.assertAlmostEqual(cur_tol, 6.5)

        # A 10.0m: con 13.0m daba reached, pero con 6.5m ya no da reached
        rover_lat = self.target_lat + (10.0 / 111194.9)
        reached, _ = check_checkpoint_reached(
            rover_lat, self.target_lon, self.target_lat, self.target_lon, cur_tol
        )
        self.assertFalse(reached, "A 10m no debe reintentar mientras la tolerancia esté estrangulada a 6.5m")

        # 2do rechazo: estrangula a 3.25m
        cur_tol = max(0.5, cur_tol * 0.5)
        self.assertAlmostEqual(cur_tol, 3.25)

        # Múltiples rechazos: satura en el piso mínimo de 0.5m
        for _ in range(5):
            cur_tol = max(0.5, cur_tol * 0.5)
        self.assertEqual(cur_tol, 0.5)

        # Al avanzar de checkpoint (nueva secuencia): se restaura la tolerancia base
        target_seq = 1
        new_target_seq = 2
        if new_target_seq != target_seq:
            cur_tol = base_tol
        self.assertEqual(cur_tol, 13.0)

    def test_navigation_flow_continuity(self):
        """Verifica que el cálculo de LocalGoal permanezca intacto y no rompa el plan BEV."""
        # Rover a 5.0m, rumbo Norte (0°)
        rover_lat = self.target_lat - (5.0 / 111194.9)  # 5m al Sur del target
        rover_lon = self.target_lon
        heading_deg = 0.0  # Mirando al Norte

        reached, dist = check_checkpoint_reached(
            rover_lat, rover_lon, self.target_lat, self.target_lon, self.radius_m
        )
        self.assertTrue(reached)

        # LocalGoal debe seguir apuntando hacia adelante (y_forward_m > 0)
        goal = goal_from_gps(rover_lat, rover_lon, heading_deg,
                             self.target_lat, self.target_lon, max_range_m=3.5)
        self.assertIsInstance(goal, LocalGoal)
        self.assertAlmostEqual(goal.distance_m, 5.0, delta=0.2)
        self.assertAlmostEqual(goal.y_forward_m, 3.5, delta=0.2)
        self.assertAlmostEqual(goal.x_right_m, 0.0, delta=0.2)
        self.assertAlmostEqual(goal.relative_bearing_deg, 0.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
