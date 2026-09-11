#!/usr/bin/env python3
"""
Suite de tests unitarios offline (Brief 13 / M.5 y M.4.2)
Verificación puramente numérica para ejecución en CPU sin requerir hardware físico ni GPU.
"""

import math
import numpy as np
import pytest

from genie_path_planner.planner import traversability_to_cost


# ==============================================================================
# M.5.1 — Tests del Filtro Complementario Roll / Pitch
# ==============================================================================
class ComplementaryFilterSimulator:
    def __init__(self, alpha=0.20, gyro_noise_density=0.005, sigma_acc_deg=0.85):
        self.alpha = alpha
        self.q_gyro = gyro_noise_density
        self.sigma_acc_rad = math.radians(sigma_acc_deg)
        self.roll = 0.0
        self.pitch = 0.0
        self.sigma_tilt_rad = math.radians(1.5)

    def step(self, dt, omega_x, omega_y, roll_acc, pitch_acc, gate_open):
        if gate_open:
            # Propagación gyro + Corrección accel
            roll_pred = self.roll + omega_x * dt
            pitch_pred = self.pitch + omega_y * dt
            self.roll = (1.0 - self.alpha) * roll_pred + self.alpha * roll_acc
            self.pitch = (1.0 - self.alpha) * pitch_pred + self.alpha * pitch_acc

            sigma_pred_sq = self.sigma_tilt_rad**2 + (self.q_gyro**2) * dt
            self.sigma_tilt_rad = math.sqrt(
                (1.0 - self.alpha)**2 * sigma_pred_sq + (self.alpha**2) * (self.sigma_acc_rad**2)
            )
        else:
            # Dead-reckoning angular con gyro
            self.roll += omega_x * dt
            self.pitch += omega_y * dt
            self.sigma_tilt_rad = math.sqrt(self.sigma_tilt_rad**2 + (self.q_gyro**2) * dt)

        return self.roll, self.pitch, self.sigma_tilt_rad


def test_complementary_filter_static_convergence():
    """Rover inclinado estático: el filtro debe converger al ángulo real y la incertidumbre reducirse."""
    filt = ComplementaryFilterSimulator(alpha=0.20)
    target_pitch = math.radians(10.0)
    dt = 2.0  # cadencia del SDK FrodoBots

    for _ in range(25):  # 50 segundos de convergencia
        roll, pitch, sigma = filt.step(dt, omega_x=0.0, omega_y=0.0, roll_acc=0.0, pitch_acc=target_pitch, gate_open=True)

    assert math.isclose(pitch, target_pitch, abs_tol=math.radians(0.2))
    assert math.isclose(roll, 0.0, abs_tol=math.radians(0.1))
    assert sigma < math.radians(1.0)  # La incertidumbre debe bajar


def test_complementary_filter_dynamic_ramp():
    """Rampa de inclinación creciente con gate abierto: debe seguir la rampa continuamente."""
    filt = ComplementaryFilterSimulator(alpha=0.20)
    dt = 2.0
    slope_rad_per_s = math.radians(0.5)  # rampa subiendo a 0.5 deg/s

    pitch_acc = 0.0
    for _ in range(15):
        pitch_acc += slope_rad_per_s * dt
        omega_y = slope_rad_per_s  # el gyro registra la velocidad angular
        roll, pitch, _ = filt.step(dt, omega_x=0.0, omega_y=omega_y, roll_acc=0.0, pitch_acc=pitch_acc, gate_open=True)

    assert math.isclose(pitch, pitch_acc, abs_tol=math.radians(0.5))


def test_complementary_filter_gate_closed_dead_reckoning():
    """Gate cerrado durante 30s con gyro en cero: ángulo se mantiene y la incertidumbre crece."""
    filt = ComplementaryFilterSimulator(alpha=0.20)
    filt.pitch = math.radians(8.0)
    initial_sigma = filt.sigma_tilt_rad
    dt = 2.0

    for _ in range(15):  # 30 segundos
        roll, pitch, sigma = filt.step(dt, omega_x=0.0, omega_y=0.0, roll_acc=math.radians(0.0), pitch_acc=math.radians(0.0), gate_open=False)

    assert math.isclose(pitch, math.radians(8.0), abs_tol=1e-5)  # No se resetea a 0
    assert sigma > initial_sigma  # La incertidumbre creció por deriva


def test_complementary_filter_gate_closed_gyro_integration():
    """Gate cerrado con gyro rotando: debe integrar omega * dt correctamente."""
    filt = ComplementaryFilterSimulator(alpha=0.20)
    filt.roll = 0.0
    dt = 2.0
    omega_x = math.radians(1.0)  # 1 deg/s

    for _ in range(5):  # 10 segundos -> debe rotar 10 grados
        roll, pitch, _ = filt.step(dt, omega_x=omega_x, omega_y=0.0, roll_acc=0.0, pitch_acc=0.0, gate_open=False)

    assert math.isclose(roll, math.radians(10.0), abs_tol=math.radians(0.1))


def test_complementary_filter_gate_transition_continuity():
    """Transición de gate cerrado a abierto: sin discontinuidades instantáneas."""
    filt = ComplementaryFilterSimulator(alpha=0.20)
    filt.pitch = math.radians(10.0)
    dt = 2.0

    # 10s en gate cerrado
    for _ in range(5):
        filt.step(dt, omega_x=0.0, omega_y=0.0, roll_acc=0.0, pitch_acc=0.0, gate_open=False)

    pitch_before_open = filt.pitch
    # Se abre el gate con lectura de acelerómetro en 9.0 deg
    roll_after, pitch_after, _ = filt.step(dt, omega_x=0.0, omega_y=0.0, roll_acc=0.0, pitch_acc=math.radians(9.0), gate_open=True)

    delta = abs(pitch_after - pitch_before_open)
    # Con alpha = 0.20, delta = 0.20 * (10 - 9) = 0.2 deg, nunca un salto de 10 deg a 0 deg
    assert delta <= math.radians(0.3)


# ==============================================================================
# M.5.2 — Tests del Gobernador Dinámico de Velocidad
# ==============================================================================
def compute_v_safe(t_plan_s, t_rtt=0.061, t_delay=0.080, d_horizon=4.00, a_brake=1.5, margin=1.5, min_effective_speed=0.15):
    b_term = max(0.0, float(t_plan_s)) + t_rtt + t_delay
    discrim = (b_term ** 2) + (2.0 * float(d_horizon)) / (margin * a_brake)
    if discrim > 0.0:
        v_safe = a_brake * (math.sqrt(discrim) - b_term)
    else:
        v_safe = 0.0

    if v_safe < min_effective_speed:
        v_safe = 0.0
    return float(v_safe)


def test_velocity_governor_table_values():
    """Verificar valores de la tabla teórica de frenado corregida."""
    # Para d_horizon=4.00m (forward_range), margin=1.5, a_brake=1.5, t_rtt=0.061, t_delay=0.080 (b = t_plan + 0.141):
    # 0.28s -> 2.27 m/s, 1.0s -> 1.59 m/s, 2.0s -> 1.07 m/s, 6.4s -> 0.40 m/s, 8.13s -> 0.32 m/s, 11.2s -> 0.23 m/s
    assert math.isclose(compute_v_safe(0.28), 2.27, abs_tol=0.02)
    assert math.isclose(compute_v_safe(1.00), 1.59, abs_tol=0.02)
    assert math.isclose(compute_v_safe(2.00), 1.07, abs_tol=0.02)
    assert math.isclose(compute_v_safe(6.40), 0.40, abs_tol=0.02)
    assert math.isclose(compute_v_safe(8.13), 0.32, abs_tol=0.02)
    assert math.isclose(compute_v_safe(11.20), 0.23, abs_tol=0.02)


def test_velocity_governor_floor_cutoff():
    """Si v_safe cae por debajo de 0.15 m/s, debe retornar 0.0 m/s (Stop & Wait)."""
    # Para d_horizon=4.00m, a t_plan = 18.0s, v_safe continuo sería ~0.146 m/s < 0.15 m/s -> 0.0 m/s
    assert compute_v_safe(18.0) == 0.0
    assert compute_v_safe(25.0) == 0.0


def test_velocity_governor_p95_spike_reaction():
    """El gobernador con P95 reacciona de inmediato ante un pico de latencia."""
    times = [280.0] * 9 + [5000.0]  # 9 ciclos nominales y 1 pico de 5s
    p95_lat_s = float(np.percentile(times, 95)) / 1000.0
    mean_lat_s = float(np.mean(times)) / 1000.0

    v_safe_p95 = compute_v_safe(p95_lat_s)
    v_safe_mean = compute_v_safe(mean_lat_s)

    assert p95_lat_s > mean_lat_s
    assert v_safe_p95 < v_safe_mean
    assert v_safe_p95 <= 0.85


def test_velocity_governor_degenerate_latencies():
    """Latencia cero o negativa no debe romper ni arrojar excepciones."""
    v_zero = compute_v_safe(0.0)
    v_neg = compute_v_safe(-1.0)
    assert v_zero > 1.50
    assert v_neg == v_zero


def speed_to_throttle(v_mps: float, max_linear_speed_mps: float = 1.111) -> float:
    """Convierte una velocidad física (m/s) a acelerador normalizado [0.0, 1.0]."""
    if v_mps <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, v_mps / max(max_linear_speed_mps, 0.01))))


def test_fail_safe_velocity_governor_states():
    """Verificar los casos del gobernador fail-safe con acelerador normalizado (Brief 14/15/18)."""
    forward_throttle = 0.40
    geodesic_fallback_throttle = 0.20

    def resolve_effective_throttle(safe_limit_last_rx, safe_limit_mps, age_s, path_following_enabled, require_governor=True):
        safe_throttle_limit = speed_to_throttle(safe_limit_mps)
        if safe_limit_last_rx is None:
            if require_governor:
                # Caso 1a: Esperado pero nunca recibido -> acelerador 0.0
                return 0.0
            else:
                # Caso 1b: No requerido (modo geodésico puro) -> acelerador conservador
                return max(0.0, min(geodesic_fallback_throttle, forward_throttle))
        if age_s <= 3.0:
            # Caso 2: Recibido y vigente -> clampear en dominio acelerador
            return max(0.0, min(forward_throttle, safe_throttle_limit))
        # Caso 3: Expirado (>3.0s)
        if path_following_enabled:
            return 0.0
        return max(0.0, min(geodesic_fallback_throttle, forward_throttle))

    # 1a. Nunca recibido y require_governor=True -> 0.0
    assert resolve_effective_throttle(None, 0.0, 0.0, True, require_governor=True) == 0.0
    assert resolve_effective_throttle(None, 0.0, 0.0, False, require_governor=True) == 0.0

    # 1b. Nunca recibido y require_governor=False (modo geodésico puro) -> 0.20
    assert math.isclose(resolve_effective_throttle(None, 0.0, 0.0, False, require_governor=False), 0.20)

    # 2. Recibido y vigente:
    # 2a. Operación nominal GPU: v_safe = 2.29 m/s -> safe_throttle = 1.0 -> effective = forward_throttle (0.40)
    assert math.isclose(resolve_effective_throttle(True, 2.29, 0.5, True), 0.40)
    # 2b. Degradado: v_safe = 0.30 m/s -> safe_throttle = 0.30/1.111 = 0.270 -> effective = 0.270 < 0.40
    assert math.isclose(resolve_effective_throttle(True, 0.30, 0.5, True), 0.30 / 1.111, abs_tol=1e-3)
    # 2c. Stop & Wait: v_safe = 0.0 m/s -> safe_throttle = 0.0 -> effective = 0.0
    assert resolve_effective_throttle(True, 0.0, 0.5, True) == 0.0

    # 3. Recibido pero expirado (4.0s) con path following activo -> 0.0
    assert resolve_effective_throttle(True, 2.29, 4.0, True) == 0.0

    # 4. Recibido pero expirado (4.0s) en modo geodésico puro -> 0.20
    assert math.isclose(resolve_effective_throttle(True, 2.29, 4.0, False), 0.20)


def test_heading_stale_guard():
    """Verificar la guarda de heading stale (Brief 14 / N.2)."""
    max_stale_s = 2.0

    def heading_is_fresh(last_rx, age_s):
        if last_rx is None:
            return False
        return age_s <= max_stale_s

    # Nunca recibido
    assert heading_is_fresh(None, 0.0) is False
    # Vigente (0.5s <= 2.0s)
    assert heading_is_fresh(True, 0.5) is True
    # En el borde (2.0s)
    assert heading_is_fresh(True, 2.0) is True
    # Expirado (2.1s > 2.0s)
    assert heading_is_fresh(True, 2.1) is False


# ==============================================================================
# M.5.3 — Tests de Footprint Derivado
# ==============================================================================
def derive_footprint_px(robot_length_m=0.250, robot_width_m=0.190, resolution=0.03, grid_size=240, bev_h=80, margin=1.05):
    d_circ = math.sqrt(robot_length_m**2 + robot_width_m**2)
    fp = math.ceil((d_circ / resolution) * (grid_size / bev_h) * margin)
    return int(fp)


def test_derived_footprint_px_calculations():
    """Verificar derivación matemática de footprint_px para distintas resoluciones."""
    assert derive_footprint_px(0.250, 0.190, 0.03, 240, 80, 1.05) == 33
    assert derive_footprint_px(0.250, 0.190, 0.05, 240, 56, 1.05) == 29
    assert derive_footprint_px(0.250, 0.190, 0.03, 160, 80, 1.05) == 22
    # Configuración oficial Earth Rover Mini+ (Brief 6 / Brief 20):
    # 4.0m range @ 0.03 m/px -> bev_h = 134, grid_size = 240, margin = 1.05 -> footprint_px = 20
    assert derive_footprint_px(0.250, 0.190, 0.03, 240, 134, 1.05) == 20


def test_bev_planner_standalone_compute_footprint_px():
    """Verificar función standalone compute_footprint_px exportada por bev_planner_node (T.1.4/T.1.5)."""
    from er_planning.bev_planner_node import compute_footprint_px

    fp, d_circ = compute_footprint_px(0.250, 0.190, 0.03, 240, 134, 1.05)
    assert math.isclose(d_circ, 0.314, abs_tol=0.001)
    assert fp == 20


def test_bev_planner_node_smoke_instantiation():
    """Smoke test (Brief 20 / T.1.5): Instanciar BEVPlannerNode completo y verificar construcción sin NameError."""
    import rclpy
    from unittest.mock import patch, MagicMock
    from er_planning.bev_planner_node import BEVPlannerNode

    if not rclpy.ok():
        rclpy.init()

    with patch("rover_traversability.predictor.TraversabilityPredictor") as mock_tp:
        mock_inst = MagicMock()
        mock_inst.device = "cpu"
        mock_tp.return_value = mock_inst

        node = BEVPlannerNode()
        try:
            assert node._planner_cfg.footprint_px == 20
            assert node._planner_cfg.grid_size == 240
            assert node._candidate_path_bank is not None
            assert len(node._candidate_path_bank) == 277
            assert math.isclose(node.forward_range, 4.0)
            assert math.isclose(node.side_range, 2.0)
            assert math.isclose(node.bev_resolution, 0.03)
        finally:
            node.destroy_node()
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


# ==============================================================================
# M.5.4 — Test de Validación de Isotropía
# ==============================================================================
def validate_isotropy(bev_h, bev_w):
    if bev_h != bev_w:
        raise ValueError(
            f"La grilla BEV debe ser cuadrada para garantizar isotropía en GeNIE. "
            f"Dimensiones recibidas: bev_h={bev_h}, bev_w={bev_w}."
        )
    return True


def test_isotropy_validation():
    """Confirmar que grillas rectangulares disparan ValueError y cuadradas pasan."""
    assert validate_isotropy(80, 80) is True
    assert validate_isotropy(134, 134) is True

    with pytest.raises(ValueError, match="La grilla BEV debe ser cuadrada"):
        validate_isotropy(56, 72)

    with pytest.raises(ValueError, match="La grilla BEV debe ser cuadrada"):
        validate_isotropy(80, 120)


# ==============================================================================
# M.5.5 — Tests de Compensación de Tilt en Compás Magnético
# ==============================================================================
def compute_tilt_compensated_heading(mx, my, mz, roll_rad, pitch_rad):
    sin_phi = math.sin(roll_rad)
    cos_phi = math.cos(roll_rad)
    sin_theta = math.sin(pitch_rad)
    cos_theta = math.cos(pitch_rad)

    bx = mx * cos_theta + my * sin_phi * sin_theta + mz * cos_phi * sin_theta
    by = my * cos_phi - mz * sin_phi
    yaw = math.atan2(-by, bx)
    return (math.degrees(yaw) + 360.0) % 360.0


def test_tilt_compensation_synthetic():
    """Verificar recuperación exacta de heading en plano y con inclinaciones de 10 deg."""
    b_mag = 0.5
    inc = math.radians(60.0)
    true_heading_deg = 45.0
    psi_rad = math.radians(true_heading_deg)

    bx_h = b_mag * math.cos(inc) * math.cos(psi_rad)
    by_h = -b_mag * math.cos(inc) * math.sin(psi_rad)
    bz_h = b_mag * math.sin(inc)

    # Caso 1: Plano (roll = 0, pitch = 0)
    h_flat = compute_tilt_compensated_heading(bx_h, by_h, bz_h, roll_rad=0.0, pitch_rad=0.0)
    assert math.isclose(h_flat, true_heading_deg, abs_tol=1e-4)

    # Caso 2: 10 deg Pitch
    pitch_10 = math.radians(10.0)
    mx_pitch = bx_h * math.cos(pitch_10) - bz_h * math.sin(pitch_10)
    my_pitch = by_h
    mz_pitch = bx_h * math.sin(pitch_10) + bz_h * math.cos(pitch_10)

    h_uncompensated = (math.degrees(math.atan2(-my_pitch, mx_pitch)) + 360.0) % 360.0
    assert abs(h_uncompensated - true_heading_deg) > 10.0

    h_compensated_pitch = compute_tilt_compensated_heading(mx_pitch, my_pitch, mz_pitch, roll_rad=0.0, pitch_rad=pitch_10)
    assert math.isclose(h_compensated_pitch, true_heading_deg, abs_tol=1e-4)

    # Caso 3: 10 deg Roll
    roll_10 = math.radians(10.0)
    mx_roll = bx_h
    my_roll = by_h * math.cos(roll_10) + bz_h * math.sin(roll_10)
    mz_roll = -by_h * math.sin(roll_10) + bz_h * math.cos(roll_10)

    h_compensated_roll = compute_tilt_compensated_heading(mx_roll, my_roll, mz_roll, roll_rad=roll_10, pitch_rad=0.0)
    assert math.isclose(h_compensated_roll, true_heading_deg, abs_tol=1e-4)


# ==============================================================================
# M.4.2 — Test del Canal de Confianza
# ==============================================================================
def test_confidence_channel_interpolation():
    """Verificar canal de confianza en traversability_to_cost."""
    trav = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)
    conf = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    unknown_cost = 0.20

    cost_map = traversability_to_cost(trav, unknown_cost=unknown_cost, confidence=conf)

    # Celda 0: 100% observada -> cost = 1.0 - 0.9 = 0.10
    assert math.isclose(float(cost_map[0, 0]), 0.10, abs_tol=1e-5)

    # Celda 1: 0% observada -> cost = unknown_cost = 0.20
    assert math.isclose(float(cost_map[0, 1]), unknown_cost, abs_tol=1e-5)

    # Celda 2: 50% observada -> cost = 0.20 + 0.5 * (0.10 - 0.20) = 0.15
    assert math.isclose(float(cost_map[0, 2]), 0.15, abs_tol=1e-5)


# ==============================================================================
# S.1.4 — Test de Conversión Cinemática v_safe -> Acelerador Normalizado
# ==============================================================================
def test_speed_to_throttle_conversion():
    """Verificar la función de mapeo dimensional speed_to_throttle (Brief 18 / R.1.3 & Brief 19 / S.1.4)."""
    v_max = 1.111

    # 1. Velocidad cero -> 0.0
    assert speed_to_throttle(0.0, max_linear_speed_mps=v_max) == 0.0

    # 2. Velocidad nominal por debajo del máximo (ej. 0.44 m/s) -> fracción lineal
    t_mid = speed_to_throttle(0.4444, max_linear_speed_mps=v_max)
    assert math.isclose(t_mid, 0.4444 / 1.111, abs_tol=1e-4)
    assert 0.0 < t_mid < 1.0

    # 3. Velocidad exactamente en el máximo nominal (1.111 m/s) -> 1.0
    assert math.isclose(speed_to_throttle(1.111, max_linear_speed_mps=v_max), 1.0, abs_tol=1e-5)

    # 4. Velocidad por encima del máximo (ej. v_safe teórica en GPU = 2.36 m/s) -> saturación estricta en 1.0
    assert speed_to_throttle(2.36, max_linear_speed_mps=v_max) == 1.0
    assert speed_to_throttle(10.0, max_linear_speed_mps=v_max) == 1.0

    # 5. Velocidad negativa -> 0.0
    assert speed_to_throttle(-0.5, max_linear_speed_mps=v_max) == 0.0
    assert speed_to_throttle(-10.0, max_linear_speed_mps=v_max) == 0.0


# ==============================================================================
# S.2 — Test de la Guarda de Retención de GPS en Mission Manager
# ==============================================================================
def test_gps_retention_guard():
    """Verificar la guarda de retención de GPS ante micro-cortes transitorios (Brief 19 / S.2)."""
    def resolve_effective_gps(current_gps, last_valid_gps, last_valid_time_s, now_s, max_retention_s=10.0):
        if current_gps is not None:
            lat, lon = current_gps
            return lat, lon, 0.0

        if last_valid_gps is not None and last_valid_time_s is not None:
            age_s = now_s - last_valid_time_s
            if age_s <= max_retention_s:
                lat, lon = last_valid_gps
                return lat, lon, age_s

        return None, None, None

    # Caso 1: GPS válido presente (lat=19.4326, lon=-99.1332)
    lat, lon, age = resolve_effective_gps((19.4326, -99.1332), (19.4320, -99.1330), 100.0, 105.0)
    assert lat == 19.4326 and lon == -99.1332 and age == 0.0

    # Caso 2: GPS ausente (corte transitorio), última muestra válida a los 5.0 s (age < 10.0 s)
    lat, lon, age = resolve_effective_gps(None, (19.4326, -99.1332), 100.0, 105.0, max_retention_s=10.0)
    assert lat == 19.4326 and lon == -99.1332
    assert math.isclose(age, 5.0, abs_tol=1e-5)

    # Caso 3: GPS ausente, última muestra válida a los 15.0 s (age > 10.0 s) -> Rechazo
    lat, lon, age = resolve_effective_gps(None, (19.4326, -99.1332), 100.0, 115.0, max_retention_s=10.0)
    assert lat is None and lon is None and age is None

    # Caso 4: GPS ausente y nunca se recibió coordenada previa -> Rechazo
    lat, lon, age = resolve_effective_gps(None, None, None, 105.0, max_retention_s=10.0)
    assert lat is None and lon is None and age is None
