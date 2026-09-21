"""Módulo de Guarda contra GPS malo (Fase 1: La Rovernetta).

Proporciona protección integral en tres niveles ante datos de GPS corruptos,
duplicados o con saltos espurios derivados de multicamino o degradación de señal.

Niveles de protección:
  * Nivel 1: Nominal / Salto aislado (1-2 fixes malos). La muestra corrupta se descarta.
             El rover sigue navegando por Dead-Reckoning (odometría + compás/EKF).
  * Nivel 2: Modo Degradado (>=3 fixes malos consecutivos). Navega por posición
             predicha de odometría, reduce acelerador (degraded_linear_scale),
             vacía la ventana de curso GNSS y BLOQUEA el reclamo de checkpoints
             para evitar penalizaciones por estrangulamiento de radio (13m -> 6.5m -> 3.25m).
  * Nivel 3: Parada de Emergencia (Modo degradado + sin ancla de rumbo > time_without_anchor_thresh_s).
             Frena motores a 0 hasta recuperar fix GNSS confiable.

Derivaciones físicas:
  * Salto GNSS: d_max(Δt) = v_max_phys * Δt + noise_margin.
    Con v_max_phys = 1.111 m/s (4 km/h / 3.6) y noise_margin = 1.5 m (2-sigma GNSS nominal),
    para Δt = 1.0s: d_max = 2.61 m.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .navigation import latlon_to_local_ne


def local_ne_to_latlon(lat_ref: float, lon_ref: float,
                       north: float, east: float) -> Tuple[float, float]:
    """Convierte desplazamiento local (Norte, Este) en metros a latitud/longitud."""
    lat = lat_ref + math.degrees(north / 6371000.0)
    lon = lon_ref + math.degrees(east / (6371000.0 * math.cos(math.radians(lat_ref))))
    return lat, lon


@dataclass
class GpsGuardStatus:
    """Estado y veredicto retornado por GpsGuard en cada ciclo."""
    level: int                       # 1: Normal / Salto aislado, 2: Degradado, 3: Frenado emergencia
    is_fix_new: bool                 # True si la muestra es un fix nuevo (no duplicado por tasa de consulta)
    is_fix_valid: bool               # True si la muestra pasó los filtros y no es salto
    effective_lat: float             # Latitud efectiva a usar (GPS real o predicha por dead-reckoning)
    effective_lon: float             # Longitud efectiva a usar (GPS real o predicha por dead-reckoning)
    can_claim_checkpoints: bool      # True solo en Nivel 1 con fix no espurio
    throttle_scale: float            # Factor de escala de velocidad (1.0 nominal, 0.6 degradado, 0.0 stop)
    consecutive_bad_fixes: int       # Contador de fixes malos seguidos
    time_without_anchor_s: float     # Tiempo transcurrido sin ancla de rumbo
    reason: str                      # Causa explicativa de rechazo o modo actual


class GpsGuard:
    """Guarda contra GPS malo para navegación y checkpoints."""

    def __init__(
        self,
        enabled: bool = True,
        v_max_phys_m_s: float = 1.111,
        gps_jump_noise_margin_m: float = 1.5,
        bad_fix_consecutive_thresh: int = 3,
        degraded_linear_scale: float = 0.6,
        degraded_max_linear: float = 0.25,
        degraded_min_linear: float = 0.20,
        deadband_linear: float = 0.15,
        time_without_anchor_thresh_s: float = 15.0,
        min_fix_interval_s: float = 0.8,
        min_fix_quality: int = 1,
        hdop_reject: float = 0.080,
        **kwargs,
    ):
        self.enabled = bool(enabled)
        self.v_max_phys_m_s = float(v_max_phys_m_s)
        self.gps_jump_noise_margin_m = float(gps_jump_noise_margin_m)
        self.bad_fix_consecutive_thresh = int(bad_fix_consecutive_thresh)
        self.degraded_linear_scale = float(degraded_linear_scale)
        self.degraded_max_linear = float(degraded_max_linear)
        self.degraded_min_linear = float(degraded_min_linear)
        self.deadband_linear = float(deadband_linear)
        self.time_without_anchor_thresh_s = float(time_without_anchor_thresh_s)
        self.min_fix_interval_s = float(min_fix_interval_s)
        self.min_fix_quality = int(min_fix_quality)
        self.hdop_reject = float(hdop_reject)

        # Estado interno
        self.level: int = 1
        self.consecutive_bad_fixes: int = 0
        self.last_valid_lat: Optional[float] = None
        self.last_valid_lon: Optional[float] = None
        self.last_valid_time: Optional[float] = None
        self.last_valid_stamp: Optional[float] = None

        # Posición predicha por dead-reckoning respecto al último fix válido
        self.accum_north: float = 0.0
        self.accum_east: float = 0.0
        self.pred_lat: Optional[float] = None
        self.pred_lon: Optional[float] = None
        self.last_odom_pose: Optional[Tuple[float, float, float]] = None

        # Rastreo de ancla de rumbo
        self.last_anchor_time: float = time.time()

    def update(
        self,
        telem: Any,
        odom_pose: Any = None,
        heading_deg: Optional[float] = None,
        has_heading_anchor: bool = True,
        now: Optional[float] = None,
    ) -> GpsGuardStatus:
        """Actualiza el estado de la guarda evaluando telemetría y odometría."""
        if now is None:
            now = time.time()

        # Si está deshabilitado por configuración, bypass directo
        if not self.enabled:
            lat = float(getattr(telem, "latitude", 0.0))
            lon = float(getattr(telem, "longitude", 0.0))
            return GpsGuardStatus(
                level=1,
                is_fix_new=True,
                is_fix_valid=True,
                effective_lat=lat,
                effective_lon=lon,
                can_claim_checkpoints=True,
                throttle_scale=1.0,
                consecutive_bad_fixes=0,
                time_without_anchor_s=0.0,
                reason="gps_guard_deshabilitado",
            )

        # -------------------------------------------------------------
        # 1. Integración de Dead-Reckoning (odometría + rumbo)
        # -------------------------------------------------------------
        if odom_pose is not None:
            ox = getattr(odom_pose, "x", 0.0)
            oy = getattr(odom_pose, "y", 0.0)
            ot = getattr(odom_pose, "theta", 0.0)
            if self.last_odom_pose is not None:
                dx = ox - self.last_odom_pose[0]
                dy = oy - self.last_odom_pose[1]
                ds = math.hypot(dx, dy)
                if ds > 0.0:
                    # Usar rumbo absoluto en grados (0=Norte, 90=Este)
                    psi_deg = heading_deg if heading_deg is not None else math.degrees(ot)
                    psi_rad = math.radians(psi_deg)
                    dn = ds * math.cos(psi_rad)
                    de = ds * math.sin(psi_rad)
                    self.accum_north += dn
                    self.accum_east += de
                    if self.last_valid_lat is not None and self.last_valid_lon is not None:
                        self.pred_lat, self.pred_lon = local_ne_to_latlon(
                            self.last_valid_lat, self.last_valid_lon,
                            self.accum_north, self.accum_east
                        )
            self.last_odom_pose = (ox, oy, ot)

        # -------------------------------------------------------------
        # 2. Rastreo de ancla de rumbo
        # -------------------------------------------------------------
        if has_heading_anchor:
            self.last_anchor_time = now
            time_without_anchor = 0.0
        else:
            time_without_anchor = now - self.last_anchor_time

        # -------------------------------------------------------------
        # 3. Extracción de coordenadas y timestamps del fix
        # -------------------------------------------------------------
        lat = float(getattr(telem, "latitude", 0.0))
        lon = float(getattr(telem, "longitude", 0.0))
        gps_ts = getattr(telem, "gps_timestamp", None)
        if gps_ts is None:
            gps_ts = getattr(telem, "timestamp", now)
        gps_ts = float(gps_ts)

        # Validación básica de calidad GNSS
        is_quality_ok = True
        reject_reason = ""

        if abs(lat) > 90.0 or abs(lon) > 180.0 or (lat == 0.0 and lon == 0.0):
            is_quality_ok = False
            reject_reason = "coordenadas_invalidas_o_nulas"
        else:
            fix_q = getattr(telem, "fix_quality", None)
            if fix_q is not None and int(fix_q) < self.min_fix_quality:
                is_quality_ok = False
                reject_reason = f"fix_quality_insuficiente ({fix_q} < {self.min_fix_quality})"

            gps_sig = getattr(telem, "gps_signal", None)
            if gps_sig is not None and float(gps_sig) <= 0.0:
                is_quality_ok = False
                reject_reason = "gps_signal_nula"

            hdop = getattr(telem, "hdop", None)
            if hdop is not None and float(hdop) > self.hdop_reject and float(hdop) > 0.0:
                is_quality_ok = False
                reject_reason = f"hdop_elevado ({hdop} > {self.hdop_reject})"

        # -------------------------------------------------------------
        # 4. Deduplicación por timestamp y coordenadas (Paso 1.1)
        # -------------------------------------------------------------
        if self.last_valid_stamp is not None:
            dt_ts = gps_ts - self.last_valid_stamp
            # Muestra repetida por tasa de consulta (timestamp idéntico o atrasado)
            if dt_ts <= 0.0:
                eff_lat = self.pred_lat if self.pred_lat is not None else self.last_valid_lat
                eff_lon = self.pred_lon if self.pred_lon is not None else self.last_valid_lon
                return GpsGuardStatus(
                    level=self.level,
                    is_fix_new=False,
                    is_fix_valid=True,
                    effective_lat=eff_lat,
                    effective_lon=eff_lon,
                    can_claim_checkpoints=(self.level == 1),
                    throttle_scale=self._get_throttle_scale(time_without_anchor),
                    consecutive_bad_fixes=self.consecutive_bad_fixes,
                    time_without_anchor_s=time_without_anchor,
                    reason="fix_duplicado_por_timestamp",
                )

            # Muestra repetida del receptor de 1 Hz antes de completar el intervalo mínimo
            if (self.last_valid_lat is not None and
                    lat == self.last_valid_lat and lon == self.last_valid_lon and
                    dt_ts < self.min_fix_interval_s):
                eff_lat = self.pred_lat if self.pred_lat is not None else self.last_valid_lat
                eff_lon = self.pred_lon if self.pred_lon is not None else self.last_valid_lon
                return GpsGuardStatus(
                    level=self.level,
                    is_fix_new=False,
                    is_fix_valid=True,
                    effective_lat=eff_lat,
                    effective_lon=eff_lon,
                    can_claim_checkpoints=(self.level == 1),
                    throttle_scale=self._get_throttle_scale(time_without_anchor),
                    consecutive_bad_fixes=self.consecutive_bad_fixes,
                    time_without_anchor_s=time_without_anchor,
                    reason="fix_duplicado_coordenadas_1hz",
                )

        # -------------------------------------------------------------
        # 5. Detección de salto físico (Paso 1.2)
        # -------------------------------------------------------------
        is_bad = False
        reason = ""

        if self.last_valid_lat is None or self.last_valid_lon is None:
            # Inicialización del primer fix
            if is_quality_ok:
                self._accept_valid_fix(lat, lon, gps_ts, now)
                return GpsGuardStatus(
                    level=1,
                    is_fix_new=True,
                    is_fix_valid=True,
                    effective_lat=lat,
                    effective_lon=lon,
                    can_claim_checkpoints=True,
                    throttle_scale=1.0,
                    consecutive_bad_fixes=0,
                    time_without_anchor_s=time_without_anchor,
                    reason="primer_fix_anclado",
                )
            else:
                is_bad = True
                reason = reject_reason
        else:
            if not is_quality_ok:
                is_bad = True
                reason = reject_reason
            else:
                # Intervalo transcurrido desde el último fix válido
                dt_fix = gps_ts - self.last_valid_stamp if self.last_valid_stamp else 1.0
                if dt_fix <= 0.0:
                    dt_fix = 1.0

                d_max_allowed = self.v_max_phys_m_s * dt_fix + self.gps_jump_noise_margin_m

                # Desplazamiento medido por GNSS desde el último ancla
                n_meas, e_meas = latlon_to_local_ne(
                    self.last_valid_lat, self.last_valid_lon, lat, lon
                )

                # Distancia entre el fix recibido y la posición predicha por odometría
                d_err = math.hypot(n_meas - self.accum_north, e_meas - self.accum_east)

                if d_err > d_max_allowed:
                    is_bad = True
                    reason = (
                        f"salto_detectado (err={d_err:.2f}m > d_max={d_max_allowed:.2f}m "
                        f"en dt={dt_fix:.2f}s)"
                    )

        # -------------------------------------------------------------
        # 6. Comportamiento escalonado (Paso 1.4)
        # -------------------------------------------------------------
        if is_bad:
            self.consecutive_bad_fixes += 1

            if self.consecutive_bad_fixes < self.bad_fix_consecutive_thresh:
                # Nivel 1: Salto aislado (1-2 fixes malos)
                self.level = 1
                print(
                    f"[GPS_GUARD] Salto/fix malo aislado detectado: {reason}. "
                    f"Fix descartado ({self.consecutive_bad_fixes}/{self.bad_fix_consecutive_thresh})."
                )
            else:
                # Nivel 2 o 3: GPS malo sostenido (>=3 fixes malos seguidos)
                if self.level < 2:
                    print(
                        f"[GPS_GUARD] NIVEL 1 -> NIVEL 2: GPS malo sostenido "
                        f"({self.consecutive_bad_fixes} fixes malos seguidos). "
                        f"Entrando en MODO DEGRADADO (bloqueo de checkpoints, "
                        f"acelerador reducido x{self.degraded_linear_scale}, "
                        "navegación por dead-reckoning)."
                    )
                self.level = 2

                # Verificar escalado a Nivel 3 (sin ancla de rumbo por tiempo prolongado)
                if time_without_anchor > self.time_without_anchor_thresh_s:
                    if self.level < 3:
                        print(
                            f"[GPS_GUARD] NIVEL 2 -> NIVEL 3: Modo degradado sin ancla de rumbo "
                            f"por {time_without_anchor:.1f}s > {self.time_without_anchor_thresh_s:.1f}s. "
                            "FRENADO DE EMERGENCIA HASTA RECUPERAR GPS."
                        )
                    self.level = 3

            eff_lat = self.pred_lat if self.pred_lat is not None else self.last_valid_lat
            eff_lon = self.pred_lon if self.pred_lon is not None else self.last_valid_lon

            return GpsGuardStatus(
                level=self.level,
                is_fix_new=True,
                is_fix_valid=False,
                effective_lat=eff_lat if eff_lat is not None else lat,
                effective_lon=eff_lon if eff_lon is not None else lon,
                can_claim_checkpoints=False,  # Reclamo BLOQUEADO en fixes malos o modo degradado
                throttle_scale=self._get_throttle_scale(time_without_anchor),
                consecutive_bad_fixes=self.consecutive_bad_fixes,
                time_without_anchor_s=time_without_anchor,
                reason=reason,
            )

        else:
            # Fix válido recibido: recuperación a Nivel 1
            if self.level > 1:
                print(
                    f"[GPS_GUARD] NIVEL {self.level} -> NIVEL 1: Fix GPS válido recuperado "
                    f"({lat:.7f}, {lon:.7f}). Restaurando navegación nominal y reclamo de checkpoints."
                )
            self.level = 1
            self.consecutive_bad_fixes = 0
            self._accept_valid_fix(lat, lon, gps_ts, now)

            return GpsGuardStatus(
                level=1,
                is_fix_new=True,
                is_fix_valid=True,
                effective_lat=lat,
                effective_lon=lon,
                can_claim_checkpoints=True,
                throttle_scale=1.0,
                consecutive_bad_fixes=0,
                time_without_anchor_s=time_without_anchor,
                reason="fix_nominal_valido",
            )

    def _accept_valid_fix(self, lat: float, lon: float, gps_ts: float, now: float) -> None:
        """Reinicia el anclaje y resetea la acumulación de dead-reckoning."""
        self.last_valid_lat = lat
        self.last_valid_lon = lon
        self.last_valid_stamp = gps_ts
        self.last_valid_time = now
        self.accum_north = 0.0
        self.accum_east = 0.0
        self.pred_lat = lat
        self.pred_lon = lon

    def _get_throttle_scale(self, time_without_anchor: float) -> float:
        """Determina la escala de acelerador según el nivel actual y ancla."""
        if self.level == 3 or (self.level == 2 and time_without_anchor > self.time_without_anchor_thresh_s):
            return 0.0
        elif self.level == 2:
            return self.degraded_linear_scale
        return 1.0

    def apply_throttle(self, cmd_linear: float) -> float:
        """Aplica el techo y piso al acelerador absoluto lineal según el nivel de la guarda.

        En Nivel 1 (Nominal): Mantiene el comando sin modificación.
        En Nivel 2 (Modo Degradado): Aplica un techo absoluto (degraded_max_linear)
            para limitar el avance a ciegas por dead-reckoning. El piso mínimo
            (degraded_min_linear) se aplica ÚNICAMENTE si el comando original ya estaba
            por encima de la zona muerta de fricción de los motores (deadband_linear ~0.15),
            evitando acelerar maniobras lentas o de ajuste fino del planner.
        En Nivel 3 (Freno de Emergencia): Corta el acelerador a 0.0.
        """
        if self.level == 3:
            return 0.0
        if self.level == 2:
            if cmd_linear > self.deadband_linear:
                return min(self.degraded_max_linear, max(self.degraded_min_linear, cmd_linear))
            elif cmd_linear > 0.0:
                return min(self.degraded_max_linear, cmd_linear)
            elif cmd_linear < -self.deadband_linear:
                return max(-self.degraded_max_linear, min(-self.degraded_min_linear, cmd_linear))
            elif cmd_linear < 0.0:
                return max(-self.degraded_max_linear, cmd_linear)
        return cmd_linear

