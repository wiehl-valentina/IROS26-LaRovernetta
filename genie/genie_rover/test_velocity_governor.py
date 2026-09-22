"""Tests unitarios para el Gobernador Dinámico de Velocidad por Latencia (P95)."""

import unittest
import numpy as np

from genie_rover.velocity_governor import GovernorConfig, VelocityGovernor


class TestVelocityGovernor(unittest.TestCase):
    def setUp(self):
        self.cfg = GovernorConfig(
            enabled=True,
            window_size=30,
            a_brake=1.5,
            cmd_latency_s=2.0,
            d_horizon_m=1.25,
            margin=1.2,
            min_speed_mps=0.10,
            max_linear_speed_mps=0.557,
            alpha_up=0.25,
        )
        self.gov = VelocityGovernor(self.cfg)

    def test_derivation_physics_formula(self):
        """Verifica la solución exacta de la ecuación cuadrática de frenado."""
        # Para t_plan_p95 = 0.2s, t_total = 2.2s
        # 1/(2*1.5) * v^2 + 2.2 * v - 1.25/1.2 = 0
        # v^2/3 + 2.2*v - 1.041667 = 0 -> v_safe ~ 0.4437 m/s
        v_safe = self.gov.compute_v_safe_raw(0.20)
        self.assertAlmostEqual(v_safe, 0.4437, places=3)

    def test_low_latency_nominal_throttle(self):
        """En baja latencia (0.2s / 5Hz), v_safe permite el acelerador nominal (0.50)."""
        # Alimentar 30 muestras de 0.20s
        for _ in range(30):
            t_p95, v_safe, throttle_lim = self.gov.update(0.20)

        self.assertAlmostEqual(t_p95, 0.20, places=2)
        self.assertAlmostEqual(v_safe, 0.4437, places=2)
        # throttle_lim = 0.4437 / 0.557 ~ 0.796
        self.assertGreater(throttle_lim, 0.75)

        # Si el comando propuesto es 0.50, no se recorta
        cmd_out = self.gov.apply_throttle_limit(0.50)
        self.assertEqual(cmd_out, 0.50)

    def test_medium_latency_scaling(self):
        """En latencia media (0.8s), v_safe reduce el acelerador por debajo de 0.70 pero encima de 0.50."""
        gov = VelocityGovernor(self.cfg)
        for _ in range(30):
            t_p95, v_safe, throttle_lim = gov.update(0.80)

        # t_total = 2.8s -> v_safe ~ 0.357 m/s -> throttle_lim ~ 0.357 / 0.557 ~ 0.64
        self.assertAlmostEqual(t_p95, 0.80, places=2)
        self.assertAlmostEqual(v_safe, 0.357, places=2)
        self.assertAlmostEqual(throttle_lim, 0.64, places=2)

    def test_high_latency_clamping(self):
        """En latencia alta (2.5s), throttle_lim cae a ~0.41, recortando un comando de 0.50."""
        gov = VelocityGovernor(self.cfg)
        for _ in range(30):
            t_p95, v_safe, throttle_lim = gov.update(2.50)

        # t_total = 4.5s -> v_safe ~ 0.228 m/s -> throttle_lim ~ 0.409
        self.assertAlmostEqual(v_safe, 0.228, places=2)
        self.assertAlmostEqual(throttle_lim, 0.409, places=2)

        # Un comando de 0.50 debe ser recortado a ~0.409
        cmd_out = gov.apply_throttle_limit(0.50)
        self.assertAlmostEqual(cmd_out, 0.409, places=2)

    def test_extreme_latency_stop_and_wait(self):
        """En latencia extrema (>8.0s), v_safe < 0.10 m/s y el gobernador corta a 0.0 (Stop & Wait)."""
        gov = VelocityGovernor(self.cfg)
        for _ in range(30):
            t_p95, v_safe, throttle_lim = gov.update(8.50)

        # t_total = 10.5s -> v_safe ~ 0.099 m/s < 0.10 m/s
        self.assertEqual(v_safe, 0.0)
        self.assertEqual(throttle_lim, 0.0)

        # Cualquier comando lineal positivo debe cortar a 0.0
        self.assertEqual(gov.apply_throttle_limit(0.50), 0.0)
        self.assertEqual(gov.apply_throttle_limit(0.20), 0.0)
        # Comandos negativos (retroceso) no deben ser bloqueados por el gobernador de avance
        self.assertEqual(gov.apply_throttle_limit(-0.30), -0.30)

    def test_single_frame_spike_immunity(self):
        """Un pico aislado de 1 frame no debe disparar oscilación completa del comando."""
        gov = VelocityGovernor(self.cfg)
        # Llenar con 29 frames rápidos a 0.20s
        for _ in range(29):
            gov.update(0.20)

        # Un pico aislado enorme de 8.0s (1 solo frame)
        t_p95, v_safe, throttle_lim = gov.update(8.0)

        # En una ventana de 30 muestras, 29 son 0.20s y 1 es 8.0s.
        # El percentil 95 (índice 29 * 0.95 = 27.55) sigue siendo 0.20s
        self.assertAlmostEqual(t_p95, 0.20, places=2)
        self.assertAlmostEqual(v_safe, 0.4437, places=2)

        # El comando nominal de 0.50 NO debe colapsar a 0
        cmd_out = gov.apply_throttle_limit(0.50)
        self.assertEqual(cmd_out, 0.50)

    def test_asymmetric_anti_oscillation_smoothing(self):
        """El freno debe actuar rápido ante degradación persistente, pero la aceleración debe ser gradual."""
        gov = VelocityGovernor(self.cfg)
        # Arrancar en régimen degradado (alta latencia persistente de 2.5s)
        for _ in range(30):
            gov.update(2.50)
        self.assertAlmostEqual(gov.filtered_throttle_limit, 0.409, places=2)

        # Si la latencia mejora de golpe a 0.20s, la ventana móvil P95 y alpha_up
        # garantizan que la recuperación no sea un salto abrupto en 1 frame:
        for _ in range(5):
            gov.update(0.20)
        # Tras 5 frames rápidos, el límite sigue firmemente contenido (no saltó a 0.80)
        self.assertLess(gov.filtered_throttle_limit, 0.50)

        # Tras 25 frames rápidos, se recupera de forma continua y suave:
        for _ in range(25):
            gov.update(0.20)
        self.assertGreater(gov.filtered_throttle_limit, 0.55)

        # Con 15 frames adicionales rápidos, converge hacia el nominal:
        for _ in range(15):
            gov.update(0.20)
        self.assertGreater(gov.filtered_throttle_limit, 0.70)

        # Ahora probamos la reactividad de frenado: si empeora de golpe (30 frames de 5.0s),
        # el gobernador frena de inmediato hacia el nuevo límite bajo
        for _ in range(30):
            gov.update(5.0)
        self.assertLess(gov.filtered_throttle_limit, 0.35)

    def test_simultaneous_bad_gps_and_high_latency_composition(self):
        """Verifica que el clamp de GpsGuard y VelocityGovernor compongan tomando estrictamente el mínimo (el más restrictivo)."""
        from genie_rover.gps_guard import GpsGuard

        guard = GpsGuard(
            degraded_max_linear=0.35,
            degraded_min_linear=0.20,
            deadband_linear=0.15,
        )
        gov = VelocityGovernor(self.cfg)

        # Caso 1: GPS Nominal (L1) + Latencia Baja (0.2s) -> Comando nominal de 0.50 pasa intacto
        guard.level = 1
        for _ in range(30):
            gov.update(0.20)
        cmd_lin = 0.50
        cmd_lin = guard.apply_throttle(cmd_lin)
        cmd_lin = gov.apply_throttle_limit(cmd_lin)
        self.assertAlmostEqual(cmd_lin, 0.50)

        # Caso 2: GPS Degradado (L2, techo 0.35) + Latencia Baja (Gov permite ~0.79)
        # GpsGuard acota a 0.35, Governor no restringe más -> out = 0.35
        guard.level = 2
        cmd_lin = 0.50
        cmd_lin = guard.apply_throttle(cmd_lin)
        cmd_lin = gov.apply_throttle_limit(cmd_lin)
        self.assertAlmostEqual(cmd_lin, 0.35)

        # Caso 3: GPS Degradado (L2, techo 0.35) + Latencia Alta (Gov acota a ~0.228 m/s -> throttle ~0.409)
        # GpsGuard acota a 0.35, Governor permite 0.409 -> el mínimo es 0.35
        for _ in range(30):
            gov.update(2.50)
        cmd_lin = 0.50
        cmd_lin = guard.apply_throttle(cmd_lin)
        cmd_lin = gov.apply_throttle_limit(cmd_lin)
        self.assertAlmostEqual(cmd_lin, 0.35)

        # Caso 4: GPS Degradado (L2, techo 0.35) + Latencia Aún Más Alta (Gov acota a ~0.17 m/s -> throttle ~0.30)
        # GpsGuard permitiría 0.35, pero Governor es más restrictivo y acota a ~0.30 -> out = ~0.30
        for _ in range(30):
            gov.update(4.0)
        self.assertLess(gov.filtered_throttle_limit, 0.35)
        cmd_lin = 0.50
        cmd_lin = guard.apply_throttle(cmd_lin)
        cmd_lin = gov.apply_throttle_limit(cmd_lin)
        self.assertAlmostEqual(cmd_lin, gov.filtered_throttle_limit)
        self.assertLess(cmd_lin, 0.35)

        # Caso 5: GPS Degradado (L2, techo 0.35) + Latencia Extrema (Gov en Stop & Wait -> 0.0)
        # Governor corta a 0.0 a pesar de que GpsGuard permitía hasta 0.35
        for _ in range(30):
            gov.update(8.5)
        cmd_lin = 0.50
        cmd_lin = guard.apply_throttle(cmd_lin)
        cmd_lin = gov.apply_throttle_limit(cmd_lin)
        self.assertEqual(cmd_lin, 0.0)

        # Caso 6: GPS Freno Emergencia (L3 -> 0.0) + Latencia Baja (Gov permite 0.80)
        # GpsGuard corta a 0.0 y el Governor no lo acelera
        guard.level = 3
        gov_nominal = VelocityGovernor(self.cfg)
        for _ in range(30):
            gov_nominal.update(0.20)
        cmd_lin = 0.50
        cmd_lin = guard.apply_throttle(cmd_lin)
        cmd_lin = gov_nominal.apply_throttle_limit(cmd_lin)
        self.assertEqual(cmd_lin, 0.0)


if __name__ == "__main__":
    unittest.main()
