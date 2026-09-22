"""Test unitario de validación de huella: footprint_px=20 vs footprint_px=12.

Confirma que con footprint_px=20 el planificador detecta y descarta trayectorias
que rozan obstáculos en esquinas dentro del diámetro circunscrito del chasis (D_circ = 0.314m),
mientras que el valor subdimensionado previo (footprint_px=12) los dejaba pasar por no cubrir la diagonal.
"""

import unittest
import numpy as np

from genie_path_planner.path_selection import has_majority_of_high_cost_points, filter_paths_with_high_costs


class TestFootprintCornerClearance(unittest.TestCase):
    def test_corner_obstacle_filtering(self):
        """Un obstáculo a 8 px (13.3 cm) del eje de la trayectoria roza la esquina física del rover (radio circunscrito 15.7 cm).

        - Con footprint_px=12 (r_half=6): el obstáculo cae fuera de la ventana (+/-6 px),
          por lo que el camino se considera libre y el rover colisiona su esquina.
        - Con footprint_px=20 (r_half=10): el obstáculo cae dentro de la ventana (+/-10 px),
          filtrando el camino peligroso.
        """
        grid_size = 240
        cost_map = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Trayectoria recta vertical que avanza por el centro (columna 120)
        # desde la fila 239 hasta la fila 140 (100 puntos)
        rows = np.linspace(239, 140, 100)
        cols = np.full(100, 120.0)
        path = np.stack([rows, cols], axis=1).astype(np.float32)

        # Colocamos un obstáculo sólido (costo 1.0) en columna 128 (distancia = 8 px = 13.3 cm)
        # entre las filas 180 y 200
        cost_map[180:200, 127:130] = 1.0

        # Con footprint_px = 12 (r_half = 6):
        # La ventana llega solo hasta col = 120 + 6 = 126.
        # El obstáculo en col 127:130 queda fuera de la inspección.
        blocked_12 = has_majority_of_high_cost_points(
            raw_topdown_score=cost_map,
            candidate_path=path,
            num_points=60,
            footprint_px=12,
            threshold_points_ratio=0.05,
            threshold_cost=0.8,
        )
        self.assertFalse(
            blocked_12,
            "Con footprint_px=12, el obstáculo en esquina no es detectado (falso negativo de colisión).",
        )

        # Con footprint_px = 20 (r_half = 10):
        # La ventana llega hasta col = 120 + 10 = 130.
        # El obstáculo en col 127:130 queda dentro de la inspección y el camino se marca bloqueado.
        blocked_20 = has_majority_of_high_cost_points(
            raw_topdown_score=cost_map,
            candidate_path=path,
            num_points=60,
            footprint_px=20,
            threshold_points_ratio=0.05,
            threshold_cost=0.8,
        )
        self.assertTrue(
            blocked_20,
            "Con footprint_px=20, el obstáculo en esquina dentro de D_circ es correctamente detectado y bloqueado.",
        )

        # Verificar sobre filter_paths_with_high_costs
        filtered_12 = filter_paths_with_high_costs(
            [path], cost_map, num_points=60, footprint_px=12, threshold_points_ratio=0.05, threshold_cost=0.8
        )
        self.assertEqual(len(filtered_12), 1, "footprint=12 deja pasar el camino peligroso")

        filtered_20 = filter_paths_with_high_costs(
            [path], cost_map, num_points=60, footprint_px=20, threshold_points_ratio=0.05, threshold_cost=0.8
        )
        self.assertEqual(len(filtered_20), 0, "footprint=20 rechaza el camino peligroso")


if __name__ == "__main__":
    unittest.main()
