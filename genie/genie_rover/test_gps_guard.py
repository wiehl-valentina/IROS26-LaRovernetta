"""test_gps_guard.py — Validación unitaria de GpsGuard (Fase 1: La Rovernetta).

Verifica:
1. Deduplicación por timestamp y coordenadas (Paso 1.1).
2. Salto aislado (1-2 fixes malos): muestra descartada, Nivel 1, rover sigue normal (Paso 1.4).
3. GPS malo sostenido (>=3 fixes malos seguidos): Nivel 2 Modo Degradado, reducción de velocidad
   a degraded_linear_scale=0.6, navegación por dead-reckoning y BLOQUEO de checkpoints (Paso 1.4).
4. GPS malo sostenido sin ancla de rumbo > 15s: Nivel 3 Parada de Emergencia (throttle_scale=0.0).
5. Recuperación limpia a Nivel 1 tras restablecimiento de fix GNSS válido.
6. Bloqueo de reclamo de checkpoint cuando un salto espurio 'mete' al rover dentro del radio.
"""

import math
import types
import pytest

from genie_rover.gps_guard import GpsGuard, local_ne_to_latlon
from genie_rover.navigation import check_checkpoint_reached, latlon_to_local_ne


def make_telem(lat, lon, ts, fix_q=4, sats=45.0, hdop=0.012, speed=1.95):
    return types.SimpleNamespace(
        latitude=lat,
        longitude=lon,
        timestamp=ts,
        gps_timestamp=ts,
        fix_quality=fix_q,
        gps_signal=sats,
        hdop=hdop,
        speed=speed,
        orientation=0.0,
        raw={},
    )


class MockPose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta


def test_gps_guard_deduplication():
    """Paso 1.1: Deduplicación por timestamp del GPS y frecuencia de 1 Hz."""
    guard = GpsGuard(min_fix_interval_s=0.8)

    lat0, lon0 = -34.921400, -57.954400
    t0 = 100.0

    # 1. Primer fix válido: ancla inicial
    telem0 = make_telem(lat0, lon0, t0)
    st0 = guard.update(telem0, odom_pose=MockPose(0, 0), heading_deg=0.0, now=t0)
    assert st0.is_fix_new is True
    assert st0.is_fix_valid is True
    assert st0.level == 1

    # 2. Misma consulta 100ms después (fast polling a 10 Hz) con mismo timestamp
    telem_dup = make_telem(lat0, lon0, t0)
    st_dup = guard.update(telem_dup, odom_pose=MockPose(0, 0), heading_deg=0.0, now=t0 + 0.1)
    assert st_dup.is_fix_new is False, "Mismo timestamp debe ser marcado como no nuevo (duplicado)"
    assert guard.consecutive_bad_fixes == 0, "Un duplicado no debe contarse como fix malo"

    # 3. Consulta a 200ms con timestamp avanzado pero coordenadas idénticas y dt < 0.8s
    telem_fast = make_telem(lat0, lon0, t0 + 0.2)
    st_fast = guard.update(telem_fast, odom_pose=MockPose(0, 0), heading_deg=0.0, now=t0 + 0.2)
    assert st_fast.is_fix_new is False, "Mismas coordenadas con dt < 0.8s debe ser marcado como no nuevo"
    assert guard.consecutive_bad_fixes == 0


def test_gps_guard_nominal_navigation():
    """Navegación nominal a ~0.5 m/s con fixes de 1 Hz."""
    guard = GpsGuard()
    lat0, lon0 = -34.921400, -57.954400
    t = 100.0

    # Primer fix
    guard.update(make_telem(lat0, lon0, t), odom_pose=MockPose(0, 0), heading_deg=0.0, now=t)

    # Avanzar 5 pasos de 1 segundo a 0.54 m/s rumbo Norte (heading=0°)
    for i in range(1, 6):
        t += 1.0
        dist_m = 0.54 * i
        lat_step, lon_step = local_ne_to_latlon(lat0, lon0, north=dist_m, east=0.0)
        pose_step = MockPose(x=dist_m, y=0.0, theta=0.0)

        st = guard.update(
            make_telem(lat_step, lon_step, t),
            odom_pose=pose_step,
            heading_deg=0.0,
            has_heading_anchor=True,
            now=t,
        )

        assert st.level == 1
        assert st.is_fix_valid is True
        assert st.can_claim_checkpoints is True
        assert st.throttle_scale == 1.0
        assert guard.consecutive_bad_fixes == 0


def test_gps_guard_isolated_jump():
    """Paso 1.2 y 1.4: Salto aislado (1 fix malo) se descarta, rover sigue normal en Nivel 1."""
    guard = GpsGuard(bad_fix_consecutive_thresh=3)
    lat0, lon0 = -34.921400, -57.954400
    t = 100.0

    # Primer fix
    guard.update(make_telem(lat0, lon0, t), odom_pose=MockPose(0, 0), heading_deg=0.0, now=t)

    # Paso 1 nominal: avance de 0.5m
    t += 1.0
    lat1, lon1 = local_ne_to_latlon(lat0, lon0, north=0.5, east=0.0)
    guard.update(make_telem(lat1, lon1, t), odom_pose=MockPose(0.5, 0), heading_deg=0.0, now=t)

    # Paso 2: Salto espurio de +8.0 metros al norte (física permite ~1.11*1 + 1.5 = 2.61m)
    t += 1.0
    lat_jump, lon_jump = local_ne_to_latlon(lat0, lon0, north=0.5 + 8.0, east=0.0)
    st_jump = guard.update(
        make_telem(lat_jump, lon_jump, t),
        odom_pose=MockPose(1.0, 0),  # Odometría solo avanzó 0.5m más
        heading_deg=0.0,
        now=t,
    )

    assert st_jump.level == 1, "Un salto aislado debe mantener Nivel 1"
    assert st_jump.is_fix_valid is False, "El salto debe ser invalidado"
    assert guard.consecutive_bad_fixes == 1
    assert st_jump.can_claim_checkpoints is False, "No se puede reclamar checkpoint sobre fix corrupto"
    assert st_jump.throttle_scale == 1.0, "En salto aislado el acelerador no se penaliza"

    # La posición efectiva debe ser la predicha por dead-reckoning (~1.0m norte de lat0), NO el salto de 8.5m
    n_eff, e_eff = latlon_to_local_ne(lat0, lon0, st_jump.effective_lat, st_jump.effective_lon)
    assert abs(n_eff - 1.0) < 0.1, f"Posición efectiva debió ser ~1.0m, dio {n_eff}m"

    # Paso 3: Fix válido recuperado a +1.5m reales
    t += 1.0
    lat3, lon3 = local_ne_to_latlon(lat0, lon0, north=1.5, east=0.0)
    st3 = guard.update(
        make_telem(lat3, lon3, t),
        odom_pose=MockPose(1.5, 0),
        heading_deg=0.0,
        now=t,
    )
    assert st3.level == 1
    assert st3.is_fix_valid is True
    assert guard.consecutive_bad_fixes == 0, "Fix válido debe reiniciar el contador de fallas"
    assert st3.can_claim_checkpoints is True


def test_gps_guard_sustained_bad_gps_degraded_mode():
    """Paso 1.4: GPS malo sostenido (>=3 fixes malos) entra a Nivel 2 (Modo Degradado)."""
    guard = GpsGuard(bad_fix_consecutive_thresh=3, degraded_linear_scale=0.6)
    lat0, lon0 = -34.921400, -57.954400
    t = 100.0

    # Primer fix
    guard.update(make_telem(lat0, lon0, t), odom_pose=MockPose(0, 0), heading_deg=0.0, now=t)

    # 3 saltos consecutivos de +10m cada uno
    odom_x = 0.0
    for i in range(1, 4):
        t += 1.0
        odom_x += 0.5
        lat_bad, lon_bad = local_ne_to_latlon(lat0, lon0, north=odom_x + 10.0, east=0.0)
        st = guard.update(
            make_telem(lat_bad, lon_bad, t),
            odom_pose=MockPose(odom_x, 0),
            heading_deg=0.0,
            has_heading_anchor=True,
            now=t,
        )

    assert guard.consecutive_bad_fixes == 3
    assert st.level == 2, "Con 3 fixes malos seguidos debe escalar a Nivel 2 (Modo Degradado)"
    assert st.can_claim_checkpoints is False, "En Modo Degradado el reclamo de checkpoints está BLOQUEADO"
    assert math.isclose(st.throttle_scale, 0.6, abs_tol=1e-3), "En Nivel 2 el acelerador debe reducirse a 0.6"
    # Verificación de techo y piso de acelerador absoluto sobre zona muerta (~0.15):
    assert guard.apply_throttle(0.40) == 0.25, "Comando alto debe acotarse al techo degraded_max_linear=0.25"
    assert guard.apply_throttle(0.16) == 0.20, "Comando sobre zona muerta (>0.15) debe subir al piso degraded_min_linear=0.20"
    assert guard.apply_throttle(0.08) == 0.08, "Maniobra lenta (<=0.15) NO debe acelerarse al piso"
    assert guard.apply_throttle(0.22) == 0.22, "Comando dentro de [min, max] se conserva"
    assert guard.apply_throttle(0.0) == 0.0, "Comando nulo se conserva en cero"

    # Posición efectiva debe seguir la odometría (1.5m norte), ignorando los 11.5m reportados por el GPS
    n_eff, _ = latlon_to_local_ne(lat0, lon0, st.effective_lat, st.effective_lon)
    assert abs(n_eff - 1.5) < 0.1, f"Posición efectiva debió ser dead-reckoning ~1.5m, dio {n_eff}m"


def test_checkpoint_claim_blocked_by_guard():
    """Verifica que un salto espurio dentro del radio de checkpoint es bloqueado y no se reclama."""
    guard = GpsGuard(bad_fix_consecutive_thresh=3)
    lat0, lon0 = -34.921400, -57.954400
    t = 100.0

    # Robot está en (lat0, lon0), checkpoint está a 10.0m al Norte
    lat_target, lon_target = local_ne_to_latlon(lat0, lon0, north=10.0, east=0.0)
    cp_radius_m = 5.0

    # Inicializar con fix en origen
    guard.update(make_telem(lat0, lon0, t), odom_pose=MockPose(0, 0), heading_deg=0.0, now=t)

    # Simular salto corrupto que 'teletransporta' el GPS a 9.0m al norte (a 1.0m del checkpoint, dentro de 5.0m)
    t += 1.0
    lat_jump, lon_jump = local_ne_to_latlon(lat0, lon0, north=9.0, east=0.0)
    st_jump = guard.update(
        make_telem(lat_jump, lon_jump, t),
        odom_pose=MockPose(0.2, 0),  # Odometría real solo avanzó 0.2m
        heading_deg=0.0,
        now=t,
    )

    # Con la posición cruda del GPS daría reached=True
    reached_crudo, dist_crudo = check_checkpoint_reached(lat_jump, lon_jump, lat_target, lon_target, cp_radius_m)
    assert reached_crudo is True and dist_crudo < cp_radius_m

    # Pero con GpsGuard:
    # 1. effective_lat es la predicha (~0.2m de lat0), por lo que dist_eff ~ 9.8m > 5.0m
    reached_eff, dist_eff = check_checkpoint_reached(st_jump.effective_lat, st_jump.effective_lon, lat_target, lon_target, cp_radius_m)
    assert reached_eff is False and dist_eff > cp_radius_m

    # 2. can_claim_checkpoints es explícitamente False
    assert st_jump.can_claim_checkpoints is False, "GpsGuard debe bloquear el reclamo"


def test_gps_guard_level3_emergency_stop():
    """Paso 1.4: Nivel 2 + sin ancla de rumbo > 15s escala a Nivel 3 (Parada de emergencia)."""
    guard = GpsGuard(bad_fix_consecutive_thresh=3, time_without_anchor_thresh_s=15.0)
    lat0, lon0 = -34.921400, -57.954400
    t = 100.0

    # Inicializar
    guard.update(make_telem(lat0, lon0, t), odom_pose=MockPose(0, 0), heading_deg=0.0, now=t)

    # 3 fixes malos consecutivos para entrar a Nivel 2
    for i in range(1, 4):
        t += 1.0
        lat_bad, lon_bad = local_ne_to_latlon(lat0, lon0, north=20.0 * i, east=0.0)
        st = guard.update(
            make_telem(lat_bad, lon_bad, t),
            odom_pose=MockPose(0.5 * i, 0),
            heading_deg=0.0,
            has_heading_anchor=True,
            now=t,
        )
    assert st.level == 2

    # Ahora se pierde el ancla de rumbo durante 10 segundos (< 15s)
    t += 10.0
    st_10s = guard.update(
        make_telem(lat_bad, lon_bad, t),
        odom_pose=MockPose(2.0, 0),
        heading_deg=0.0,
        has_heading_anchor=False,
        now=t,
    )
    assert st_10s.level == 2, "A 10s sin ancla debe seguir en Nivel 2"
    assert st_10s.throttle_scale == 0.6

    # Pasan 6 segundos más (total 16s sin ancla > 15.0s umbral)
    t += 6.0
    st_16s = guard.update(
        make_telem(lat_bad, lon_bad, t),
        odom_pose=MockPose(2.2, 0),
        heading_deg=0.0,
        has_heading_anchor=False,
        now=t,
    )
    assert st_16s.level == 3, "Superados los 15s sin ancla debe escalar a Nivel 3"
    assert st_16s.throttle_scale == 0.0, "Nivel 3 debe forzar acelerador a 0.0 (freno de emergencia)"
    assert guard.apply_throttle(0.40) == 0.0, "En Nivel 3 apply_throttle debe cortar acelerador a 0.0"
    assert st_16s.can_claim_checkpoints is False

    # Recuperación: llega un fix válido con ancla restablecida
    t += 1.0
    lat_good, lon_good = local_ne_to_latlon(lat0, lon0, north=2.5, east=0.0)
    st_rec = guard.update(
        make_telem(lat_good, lon_good, t),
        odom_pose=MockPose(2.5, 0),
        heading_deg=0.0,
        has_heading_anchor=True,
        now=t,
    )
    assert st_rec.level == 1, "Debe recuperarse a Nivel 1 inmediatamente tras fix bueno"
    assert st_rec.throttle_scale == 1.0
    assert guard.apply_throttle(0.40) == 0.40, "En Nivel 1 apply_throttle mantiene comando sin limitar"
    assert st_rec.can_claim_checkpoints is True
    assert guard.consecutive_bad_fixes == 0


def test_gps_guard_fix_quality_rejection():
    """Verifica que fix_quality=0 (sin fix NMEA) sea rechazado por GpsGuard."""
    guard = GpsGuard(min_fix_quality=1)
    lat0, lon0 = -34.921400, -57.954400
    t = 100.0

    # Inicializar con fix válido
    guard.update(make_telem(lat0, lon0, t, fix_q=4), odom_pose=MockPose(0, 0), heading_deg=0.0, now=t)

    # Telemetría con fix_quality=0 (código oficial de sin fix de la placa)
    t += 1.0
    lat1, lon1 = local_ne_to_latlon(lat0, lon0, north=0.5, east=0.0)
    st = guard.update(
        make_telem(lat1, lon1, t, fix_q=0),
        odom_pose=MockPose(0.5, 0),
        heading_deg=0.0,
        now=t,
    )

    assert st.is_fix_valid is False, "fix_quality=0 debe ser rechazado"
    assert guard.consecutive_bad_fixes == 1
    assert "fix_quality_insuficiente" in st.reason
