"""Odometria: donde esta el robot y hacia donde mira, en cada instante.

Es la base del mapa persistente. Sin saber cuanto se movio el rover entre dos
frames, no se pueden alinear dos observaciones del BEV.

Tres fuentes, cada una cubriendo la debilidad de las otras:

    giroscopo   ~50 Hz   rotacion instantanea, precisa a corto plazo,
                         pero acumula deriva
    ruedas      ~50 Hz   avance recorrido; patina y depende de calibracion
    GPS RTK     ~1 Hz    posicion absoluta sin deriva, pero lento y no
                         dice nada cuando el robot esta quieto

La rotacion sale del giroscopo (no tiene ambiguedad de signo, a diferencia de
las ruedas en este robot). El avance sale de las ruedas. El GPS corrige la
deriva acumulada cuando el desplazamiento es lo bastante grande como para
superar su ruido.

Marco de referencia: x adelante, y izquierda, theta antihorario desde el eje x
inicial. El origen es donde estaba el robot al arrancar.

Autoprueba (no necesita robot):
    python -m genie_rover.odometry
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .navigation import latlon_to_local_ne, wrap_deg, gps_quality

RPM_A_RAD_S = 2.0 * math.pi / 60.0


@dataclass
class Pose:
    """Pose en el plano. x adelante, y izquierda, theta en radianes."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_matrix(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s, self.x],
                         [s,  c, self.y],
                         [0., 0., 1.]])

    def relative_to(self, other: "Pose") -> "Pose":
        """Esta pose expresada en el marco de 'other'."""
        dx, dy = self.x - other.x, self.y - other.y
        c, s = math.cos(-other.theta), math.sin(-other.theta)
        return Pose(c * dx - s * dy, s * dx + c * dy,
                    wrap_rad(self.theta - other.theta))


def wrap_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class OdometryConfig:
    wheel_radius_m: float = 0.045
    track_width_m: float = 0.15
    left_rpm_indices: tuple[int, ...] = (0, 2)
    right_rpm_indices: tuple[int, ...] = (1, 3)
    rotation_sign: float = -1.0
    use_gyro_for_rotation: bool = True
    gps_correction: bool = True
    min_gps_displacement_m: float = 1.0
    # Cuanto se confia en el GPS al corregir. 0 = ignorarlo, 1 = saltar a el.
    gps_blend: float = 0.25
    hdop_nominal: float = 0.025
    hdop_reject: float = 0.080
    # Eje del giroscopo que mide guiñada (rotacion en el plano). Se determina
    # empiricamente con calibrate_gyro_axis().
    gyro_yaw_index: int = 2
    gyro_sign: float = 1.0
    # Bias del eje de guiñada en grados/segundo. Medido en reposo con el rover
    # quieto (speed=0, vibration=0) durante 60 s el 2026-09-08 (media 1.2784 dps).
    gyro_yaw_bias_dps: float = 1.2784
    # Las lecturas por debajo de esto se toman como ruido y se descartan, para
    # que el robot quieto no acumule deriva.
    gyro_deadband_dps: float = 0.5
    # Re-anclaje estructural de pose.theta hacia el heading EKF (UDP).
    ekf_heading_correction: bool = True
    ekf_heading_max_age_s: float = 1.5   # mismo criterio que ekf_staleness_s de navigation.py
    heading_blend: float = 0.3           # cuanto se acerca al EKF por ciclo; no saltar de golpe
    # Gating del acelerometro para estimacion de roll/pitch (doble condicion respecto a 1g)
    accel_gate_norm_tol: float = 0.08
    accel_gate_std_tol: float = 0.06
    # Biases residuales de giroscopios X e Y (dps) medidos en reposo el 2026-09-08
    gyro_bias_x_dps: float = 0.0831
    gyro_bias_y_dps: float = -0.0098
    # Habilitar proyeccion 3D del giroscopo al plano horizontal (eje de gravedad)
    use_tilt_projection: bool = True
    # Umbrales de degradacion de confianza EKF por inclinacion (en grados)
    tilt_blend_start_deg: float = 5.0
    tilt_blend_max_deg: float = 20.0
    tilt_max_staleness_s: float = 5.0
    max_dt_s: float = 0.5





# --------------------------------------------------- estimacion roll/pitch y tilt

def estimate_roll_pitch(accels: list,
                        gate_norm_tol: float = 0.08,
                        gate_std_tol: float = 0.06) -> tuple[float, float, float] | None:
    """Estima roll y pitch por gravedad, y la norma de |a| para el gating.

    Devuelve (roll_rad, pitch_rad, norma_g) o None si el gate no pasa
    (aceleración dominada por movimiento o vibracion excesiva, no por gravedad pura).

    Criterio de doble gate:
      1. |media(|a|) - 1.0g| < gate_norm_tol (por defecto 0.08g)
      2. std(|a|) < gate_std_tol (por defecto 0.06g)

    Convenciones:
      * ax adelante (+X), ay izquierda (+Y), az arriba (+Z).
      * roll: rotacion en radianes sobre eje X (positivo = chasis cae a la derecha, ay > 0).
      * pitch: rotacion en radianes sobre eje Y (positivo = morro hacia arriba, ax < 0).
    """
    if not accels:
        return None

    norms = []
    rolls = []
    pitches = []

    for s in accels:
        if len(s) < 3:
            continue
        ax = float(s[0])
        ay = float(s[1])
        az = float(s[2])
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        norms.append(norm)
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        rolls.append(roll)
        pitches.append(pitch)

    if not norms:
        return None

    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms)) if len(norms) > 1 else 0.0

    if abs(mean_norm - 1.0) >= gate_norm_tol or std_norm >= gate_std_tol:
        return None

    median_roll = float(np.median(rolls))
    median_pitch = float(np.median(pitches))
    return median_roll, median_pitch, mean_norm


def tilt_confidence_factor(pitch_rad: float | None, roll_rad: float | None,
                           tilt_start_deg: float = 5.0,
                           tilt_max_deg: float = 20.0) -> float:
    """Calcula el factor de degradacion de confianza (0.0 a 1.0) segun la inclinacion.

    Derivacion matematica:
      El error de rumbo en magnetometros sin compensacion tridimensional de tilt crece
      fuertemente con el angulo de inclinacion respecto al horizonte:
        * 5° de tilt  -> ~8.6° de error de heading
        * 10° de tilt -> ~16.7° de error de heading
        * 18° de tilt -> ~28.2° de error de heading
      Por debajo de tilt_start_deg (5°), el error es despreciable o comparable al ruido base
      del sensor, manteniendo confianza plena (factor = 1.0).
      Por encima de tilt_max_deg (20°), el error supera los 30° y compromete completamente
      el heading absoluto, por lo que la confianza se anula (factor = 0.0).
      Entre 5° y 20°, se aplica una transicion suave C^1 de medio coseno:
        u = (tilt_deg - tilt_start_deg) / (tilt_max_deg - tilt_start_deg)
        factor = 0.5 * (1 + cos(pi * u))
    """
    if pitch_rad is None or roll_rad is None:
        return 1.0
    cos_tilt = math.cos(roll_rad) * math.cos(pitch_rad)
    cos_tilt = max(-1.0, min(1.0, cos_tilt))
    tilt_deg = math.degrees(math.acos(cos_tilt))

    if tilt_deg <= tilt_start_deg:
        return 1.0
    if tilt_deg >= tilt_max_deg:
        return 0.0
    u = (tilt_deg - tilt_start_deg) / (tilt_max_deg - tilt_start_deg)
    return float(0.5 * (1.0 + math.cos(math.pi * u)))


class Odometry:
    """Integra telemetria en una pose. Alimentar con update() en cada ciclo."""

    def __init__(self, cfg: OdometryConfig):
        self.cfg = cfg
        self.pose = Pose()
        self._last_gyro_t: float | None = None
        self._last_wheel_t: float | None = None
        self._origin_latlon: tuple[float, float] | None = None
        self._origin_heading: float | None = None
        self._last_gps_pose: Pose | None = None
        self._last_gps_xy: tuple[float, float] | None = None
        self.gps_corrections = 0
        self.heading_corrections = 0
        self.distance_travelled = 0.0
        self.samples_integrated = 0
        self.last_roll: float | None = None
        self.last_pitch: float | None = None
        self.last_accel_norm: float | None = None
        self.tilt_gate_open: bool = False
        self.last_tilt_time: float | None = None
        self.last_blend_effective: float = cfg.heading_blend


    # ------------------------------------------------------------------ ruedas

    def _wheel_series(self, rpms: list) -> list[tuple[float, float]]:
        """Convierte el lote de rpm en [(timestamp, v_lineal_m_s), ...].

        /data devuelve unas 5 muestras por llamada, a ~50 Hz. Hay que usarlas
        TODAS: quedarse solo con la ultima e integrarla sobre el intervalo
        entero subestima muchisimo el movimiento.
        """
        salida = []
        r = self.cfg.wheel_radius_m
        for fila in rpms:
            if len(fila) < 5:
                continue
            vals = [float(v) for v in fila[:4]]
            t = float(fila[4])
            izq = float(np.mean([vals[i] for i in self.cfg.left_rpm_indices]))
            der = float(np.mean([vals[i] for i in self.cfg.right_rpm_indices]))
            v_izq = izq * RPM_A_RAD_S * r
            v_der = der * RPM_A_RAD_S * r
            salida.append((t, 0.5 * (v_izq + v_der)))
        salida.sort(key=lambda p: p[0])
        return salida

    def _wheel_omega_series(self, rpms: list) -> list[tuple[float, float]]:
        """Rotacion derivada de las ruedas, para comparar contra el gyro."""
        salida = []
        r = self.cfg.wheel_radius_m
        for fila in rpms:
            if len(fila) < 5:
                continue
            vals = [float(v) for v in fila[:4]]
            t = float(fila[4])
            izq = float(np.mean([vals[i] for i in self.cfg.left_rpm_indices]))
            der = float(np.mean([vals[i] for i in self.cfg.right_rpm_indices]))
            v_izq = izq * RPM_A_RAD_S * r
            v_der = der * RPM_A_RAD_S * r
            omega = self.cfg.rotation_sign * (v_der - v_izq) / self.cfg.track_width_m
            salida.append((t, omega))
        salida.sort(key=lambda p: p[0])
        return salida

    # --------------------------------------------------------------- giroscopo

    def _gyro_series(self, gyros: list, roll: float | None = None, pitch: float | None = None) -> list[tuple[float, float]]:
        """Convierte el lote del giroscopo en [(timestamp, omega_rad_s), ...].

        Si roll y pitch estan disponibles (gate de acelerometro abierto), proyecta
        el vector 3D de velocidad angular al plano horizontal (eje vertical de gravedad)
        antes de extraer la tasa de guiñada:
          omega_level = (gx * sin(pitch) - gy * cos(pitch)*sin(roll) + gz * cos(pitch)*cos(roll)) * gyro_sign

        Si el gate no pasa (aceleracion no dominada por gravedad), usa el fallback
        a eje Z crudo sin correccion de inclinacion, aceptando el acoplamiento cruzado transitorio.
        """
        salida = []
        i = self.cfg.gyro_yaw_index
        bias_z = self.cfg.gyro_yaw_bias_dps
        bias_x = self.cfg.gyro_bias_x_dps
        bias_y = self.cfg.gyro_bias_y_dps
        use_proj = self.cfg.use_tilt_projection and (roll is not None) and (pitch is not None)

        if use_proj:
            s_pitch = math.sin(pitch)
            c_pitch = math.cos(pitch)
            s_roll = math.sin(roll)
            c_roll = math.cos(roll)

        for fila in gyros:
            if len(fila) <= max(i, 3):
                continue
            t = float(fila[-1])

            if use_proj:
                gx = float(fila[0]) - bias_x
                gy = float(fila[1]) - bias_y
                gz = float(fila[2]) - bias_z
                dps = (gx * s_pitch - gy * c_pitch * s_roll + gz * c_pitch * c_roll) * self.cfg.gyro_sign
            else:
                dps = (float(fila[i]) - bias_z) * self.cfg.gyro_sign

            if abs(dps) < self.cfg.gyro_deadband_dps:
                dps = 0.0
            salida.append((t, math.radians(dps)))
        salida.sort(key=lambda p: p[0])
        return salida



    @staticmethod
    def _interpolar(serie: list[tuple[float, float]], t: float) -> float:
        """Valor de la serie en t, con interpolacion lineal y extremos fijos."""
        if not serie:
            return 0.0
        if t <= serie[0][0]:
            return serie[0][1]
        if t >= serie[-1][0]:
            return serie[-1][1]
        for k in range(1, len(serie)):
            t1, v1 = serie[k]
            if t <= t1:
                t0, v0 = serie[k - 1]
                if t1 <= t0:
                    return v1
                a = (t - t0) / (t1 - t0)
                return v0 + a * (v1 - v0)
        return serie[-1][1]

    # ------------------------------------------------------------------ update

    def update(self, telemetry_raw: dict,
               ekf_heading: float | None = None,
               ekf_timestamp: float | None = None,
               now: float | None = None) -> Pose:
        """Integra un lote de /data. Devuelve la pose actualizada.

        Cada llamada trae varias muestras: se integran una por una, usando el
        intervalo real entre timestamps consecutivos.
        """
        tilt_res = estimate_roll_pitch(
            telemetry_raw.get("accels", []),
            gate_norm_tol=self.cfg.accel_gate_norm_tol,
            gate_std_tol=self.cfg.accel_gate_std_tol,
        )
        if tilt_res is not None:
            self.last_roll, self.last_pitch, self.last_accel_norm = tilt_res
            self.tilt_gate_open = True
            self.last_tilt_time = now if now is not None else time.time()
        else:
            self.tilt_gate_open = False
            accels = telemetry_raw.get("accels", [])
            if accels:
                norms = [math.sqrt(float(r[0])**2 + float(r[1])**2 + float(r[2])**2) for r in accels if len(r) >= 3]
                self.last_accel_norm = float(np.mean(norms)) if norms else None

        serie_v = self._wheel_series(telemetry_raw.get("rpms", []))
        serie_w_gyro = self._gyro_series(
            telemetry_raw.get("gyros", []),
            roll=self.last_roll if self.tilt_gate_open else None,
            pitch=self.last_pitch if self.tilt_gate_open else None,
        )
        serie_w_ruedas = self._wheel_omega_series(telemetry_raw.get("rpms", []))
        serie_w = serie_w_gyro if self.cfg.use_gyro_for_rotation else serie_w_ruedas


        # Instantes a integrar: la union de ambos relojes, para no perder nada.
        tiempos = sorted({t for t, _ in serie_v} | {t for t, _ in serie_w})
        if not tiempos:
            if self.cfg.gps_correction:
                self._maybe_correct_with_gps(telemetry_raw)
            if self.cfg.ekf_heading_correction:
                ekf_h = ekf_heading if ekf_heading is not None else telemetry_raw.get("ekf_heading")
                ekf_t = ekf_timestamp if ekf_timestamp is not None else telemetry_raw.get("ekf_heading_time", telemetry_raw.get("ekf_timestamp"))
                self._maybe_correct_with_ekf_heading(ekf_h, ekf_t, now=now)
            return self.pose

        for t in tiempos:
            if t <= 0:
                continue
            prev = self._last_gyro_t
            if prev is None:
                self._last_gyro_t = t
                continue
            dt = t - prev
            if dt <= 0:
                continue
            if dt > self.cfg.max_dt_s:
                # Hubo un hueco (reconexion, freeze del video). No inventamos
                # movimiento: reanclamos el reloj y seguimos.
                self._last_gyro_t = t
                continue
            self._last_gyro_t = t

            v = self._interpolar(serie_v, t)
            w = self._interpolar(serie_w, t)

            th_medio = self.pose.theta + 0.5 * w * dt
            self.pose.x += v * dt * math.cos(th_medio)
            self.pose.y += v * dt * math.sin(th_medio)
            self.pose.theta = wrap_rad(self.pose.theta + w * dt)
            self.distance_travelled += abs(v) * dt
            self.samples_integrated += 1

        if self.cfg.gps_correction:
            self._maybe_correct_with_gps(telemetry_raw)
        if self.cfg.ekf_heading_correction:
            ekf_h = ekf_heading if ekf_heading is not None else telemetry_raw.get("ekf_heading")
            ekf_t = ekf_timestamp if ekf_timestamp is not None else telemetry_raw.get("ekf_heading_time", telemetry_raw.get("ekf_timestamp"))
            self._maybe_correct_with_ekf_heading(ekf_h, ekf_t, now=now)
        return self.pose

    # --------------------------------------------------------------------- GPS

    def _maybe_correct_with_gps(self, raw: dict) -> None:
        lat = raw.get("latitude")
        lon = raw.get("longitude")
        if lat is None or lon is None:
            return
        lat, lon = float(lat), float(lon)
        # (1000, 1000) es el codigo de "sin fix" del SDK
        if abs(lat) > 90 or abs(lon) > 180:
            return

        hdop = raw.get("hdop")
        fix_q = raw.get("fix_quality")
        gps_sig = raw.get("gps_signal")
        quality = gps_quality(
            hdop=float(hdop) if hdop is not None else None,
            fix_quality=int(fix_q) if fix_q is not None else None,
            gps_signal=float(gps_sig) if gps_sig is not None else None,
            hdop_nominal=self.cfg.hdop_nominal,
            hdop_reject=self.cfg.hdop_reject,
        )
        if quality <= 0.0:
            # Descartar fix GPS degradado o sin fix
            return

        if self._origin_latlon is None:
            self._origin_latlon = (lat, lon)
            self._last_gps_xy = (0.0, 0.0)
            self._last_gps_pose = Pose(self.pose.x, self.pose.y, self.pose.theta)
            return

        norte, este = latlon_to_local_ne(*self._origin_latlon, lat, lon)
        # El marco del mundo se ancla al rumbo inicial del robot, asi que
        # guardamos el desplazamiento GPS en su propio marco norte-este y solo
        # lo usamos para medir DISTANCIAS, no direcciones absolutas.
        gps_xy = (norte, este)

        if self._last_gps_xy is None or self._last_gps_pose is None:
            self._last_gps_xy, self._last_gps_pose = gps_xy, Pose(
                self.pose.x, self.pose.y, self.pose.theta)
            return

        d_gps = math.hypot(gps_xy[0] - self._last_gps_xy[0],
                           gps_xy[1] - self._last_gps_xy[1])
        if d_gps < self.cfg.min_gps_displacement_m:
            return

        d_odo = math.hypot(self.pose.x - self._last_gps_pose.x,
                           self.pose.y - self._last_gps_pose.y)
        if d_odo < 1e-3:
            self._last_gps_xy, self._last_gps_pose = gps_xy, Pose(
                self.pose.x, self.pose.y, self.pose.theta)
            return

        # Las ruedas patinan y tienden a sobreestimar. Corregimos la ESCALA del
        # tramo recorrido, que es un error sistematico, en vez de saltar a la
        # posicion GPS (lo que romperia la continuidad del mapa).
        escala = d_gps / d_odo
        escala = float(np.clip(escala, 0.5, 2.0))
        mezcla = 1.0 + (self.cfg.gps_blend * quality) * (escala - 1.0)

        dx = self.pose.x - self._last_gps_pose.x
        dy = self.pose.y - self._last_gps_pose.y
        self.pose.x = self._last_gps_pose.x + dx * mezcla
        self.pose.y = self._last_gps_pose.y + dy * mezcla

        self.gps_corrections += 1
        self._last_gps_xy = gps_xy
        self._last_gps_pose = Pose(self.pose.x, self.pose.y, self.pose.theta)

    # ------------------------------------------------------------- EKF Heading

    def _maybe_correct_with_ekf_heading(self, ekf_heading_deg: float | None,
                                         ekf_timestamp: float | None,
                                         now: float | None = None) -> None:
        """Corrige la deriva de pose.theta hacia el heading EKF (UDP), que es
        mucho mas confiable que la integracion local de giroscopo a largo plazo.

        No salta discontinuamente: mezcla gradual, igual que la correccion de
        escala del GPS, para no introducir un salto brusco en la pose que usa
        el mapa persistente y el seguimiento de trayectoria.

        Convencion de angulos:
          * pose.theta: angulo antihorario (CCW) en radianes desde el eje x inicial (theta=0 al arrancar).
          * ekf_heading_deg: rumbo absoluto en grados brujula (0=Norte, 90=Este, sentido horario / CW).

        Mapeo:
          Al recibir la primera lectura valida del EKF con heading H0, se fija H0 como rumbo
          del eje x inicial (anclaje inicial):
            H0 = wrap_deg(ekf_heading_deg + degrees(pose.theta))
          Para cualquier rumbo posterior H:
            theta_ekf = radians(wrap_deg(H0 - H))
          Ejemplo:
            H0 = 345° (rumbo inicial).
            Robot gira 15° a la izquierda (antihorario): H pasa a 330°.
            wrap_deg(345° - 330°) = +15° -> theta_ekf = +15° * pi/180 = +0.2618 rad.
            Coincide con pose.theta.
        """
        if ekf_heading_deg is None or ekf_timestamp is None:
            return
        t_now = time.time() if now is None else now
        if (t_now - ekf_timestamp) > self.cfg.ekf_heading_max_age_s:
            return  # dato vencido, no corregir con algo stale

        if self._origin_heading is None:
            self._origin_heading = (float(ekf_heading_deg) + math.degrees(self.pose.theta)) % 360.0
            return


        ekf_theta = math.radians(wrap_deg(self._origin_heading - float(ekf_heading_deg)))
        error = wrap_rad(ekf_theta - self.pose.theta)

        # Degradacion de confianza segun inclinacion estimada
        factor = tilt_confidence_factor(
            self.last_pitch if self.tilt_gate_open else None,
            self.last_roll if self.tilt_gate_open else None,
            tilt_start_deg=self.cfg.tilt_blend_start_deg,
            tilt_max_deg=self.cfg.tilt_blend_max_deg,
        )
        blend_efectivo = self.cfg.heading_blend * factor
        self.last_blend_effective = blend_efectivo

        if blend_efectivo > 0.0:
            self.pose.theta = wrap_rad(self.pose.theta + blend_efectivo * error)
            self.heading_corrections += 1

    @property
    def origin_heading(self) -> float | None:
        return self._origin_heading

    def reset(self) -> None:
        self.pose = Pose()
        self._last_gyro_t = None
        self._origin_latlon = None
        self._origin_heading = None
        self._last_gps_pose = None
        self._last_gps_xy = None
        self.gps_corrections = 0
        self.heading_corrections = 0
        self.distance_travelled = 0.0
        self.samples_integrated = 0
        self.last_roll = None
        self.last_pitch = None
        self.last_accel_norm = None
        self.tilt_gate_open = False
        self.last_tilt_time = None
        self.last_blend_effective = self.cfg.heading_blend

    def current_roll_pitch(self, now: float | None = None,
                           max_staleness_s: float | None = None) -> tuple[float, float] | None:
        """Devuelve (roll_rad, pitch_rad) si el gate esta abierto o dentro de la vigencia max_staleness_s.

        Manejo de gate cerrado:
          - Gate abierto: medicion fresca, se devuelve inmediatamente.
          - Gate cerrado (aceleracion lineal o vibraciones): una pendiente fisica no desaparece
            en decenas o cientos de milisegundos. Se preserva la ultima medicion valida mientras
            su antiguedad no supere max_staleness_s (por defecto tilt_max_staleness_s del config, 5.0 s).
          - Antiguedad > max_staleness_s o sin medicion previa: devuelve None
            (el consumidor cae a la pose nominal nivelada).
        """
        if self.last_roll is None or self.last_pitch is None:
            return None
        if self.tilt_gate_open:
            return self.last_roll, self.last_pitch
        staleness_limit = max_staleness_s if max_staleness_s is not None else getattr(self.cfg, "tilt_max_staleness_s", 5.0)
        t_now = now if now is not None else time.time()
        if self.last_tilt_time is not None and (t_now - self.last_tilt_time) <= staleness_limit:
            return self.last_roll, self.last_pitch
        return None




# ------------------------------------------------------- calibracion del gyro

def calibrate_gyro_axis(samples: list[list[float]]) -> tuple[int, float]:
    """Determina que columna del giroscopo mide guiñada, y con que signo.

    Alimentar con muestras tomadas mientras el robot gira en el lugar hacia la
    IZQUIERDA. El eje de guiñada es el que mas se aleja de cero; el signo se
    elige para que un giro a la izquierda de positivo.
    """
    a = np.array(samples, dtype=float)
    if a.size == 0:
        return 2, 1.0
    medias = a[:, :3].mean(axis=0)
    i = int(np.argmax(np.abs(medias)))
    signo = 1.0 if medias[i] > 0 else -1.0
    return i, signo


# --------------------------------------------------------------------- pruebas

def _self_test() -> None:
    """Las pruebas simulan LOTES de 5 muestras a 50 Hz, como manda /data.

    La version anterior de este test alimentaba una muestra por llamada, y por
    eso no detecto que update() estaba descartando 4 de cada 5. Reproducir el
    formato real del SDK es parte de la prueba.
    """
    cfg = OdometryConfig(gyro_yaw_bias_dps=0.0)
    dt_muestra = 0.02          # 50 Hz dentro del lote

    por_lote = 5               # /data devuelve 5 muestras
    dt_lote = dt_muestra * por_lote

    def lote(t0, rpm=(0, 0, 0, 0), gyro_dps=0.0):
        rpms, gyros = [], []
        for k in range(por_lote):
            t = t0 + k * dt_muestra
            rpms.append([*rpm, t])
            gyros.append([0.0, 0.0, gyro_dps, t])
        return {"rpms": rpms, "gyros": gyros}

    print("=== 2 m en linea recta (lotes de 5 muestras) ===")
    odo = Odometry(cfg)
    v = 30 * RPM_A_RAD_S * cfg.wheel_radius_m
    n_lotes = int(round(2.0 / (v * dt_lote)))
    t = dt_muestra
    for _ in range(n_lotes):
        odo.update(lote(t, rpm=(30, 30, 30, 30)))
        t += dt_lote
    print(f"  v={v:.4f} m/s, {n_lotes} lotes = {n_lotes*por_lote} muestras")
    print(f"  integradas: {odo.samples_integrated}")
    print(f"  x={odo.pose.x:.4f} y={odo.pose.y:.4f} "
          f"theta={math.degrees(odo.pose.theta):.2f} grados")
    assert abs(odo.pose.x - 2.0) < 0.02, f"esperaba 2 m, dio {odo.pose.x:.3f}"
    assert abs(odo.pose.y) < 1e-6

    print("\n=== 90 grados en el lugar ===")
    odo = Odometry(cfg)
    n_lotes = int(round(90 / (45 * dt_lote)))
    t = dt_muestra
    for _ in range(n_lotes):
        odo.update(lote(t, gyro_dps=45.0))
        t += dt_lote
    print(f"  theta = {math.degrees(odo.pose.theta):.2f} grados (esperado 90)")
    assert abs(math.degrees(odo.pose.theta) - 90) < 1.0
    assert abs(odo.pose.x) < 1e-6 and abs(odo.pose.y) < 1e-6

    print("\n=== cuarto de circulo de radio 1 m ===")
    odo = Odometry(cfg)
    radio = 1.0
    w_dps = math.degrees(v / radio)
    n_lotes = int(round(90 / (w_dps * dt_lote)))
    t = dt_muestra
    for _ in range(n_lotes):
        odo.update(lote(t, rpm=(30, 30, 30, 30), gyro_dps=w_dps))
        t += dt_lote
    print(f"  esperado x={radio:.3f} y={radio:.3f} theta=90")
    print(f"  obtenido x={odo.pose.x:.3f} y={odo.pose.y:.3f} "
          f"theta={math.degrees(odo.pose.theta):.1f}")
    err = math.hypot(odo.pose.x - radio, odo.pose.y - radio)
    print(f"  error: {err*1000:.1f} mm")
    assert err < 0.03

    print("\n=== usar todas las muestras vs solo la ultima ===")
    odo_bien = Odometry(cfg)
    odo_mal = Odometry(cfg)
    t = dt_muestra
    for _ in range(50):
        b = lote(t, rpm=(30, 30, 30, 30))
        odo_bien.update(b)
        odo_mal.update({"rpms": [b["rpms"][-1]], "gyros": [b["gyros"][-1]]})
        t += dt_lote
    print(f"  todas las muestras: {odo_bien.pose.x:.3f} m")
    print(f"  solo la ultima:     {odo_mal.pose.x:.3f} m")
    print(f"  relacion: {odo_bien.pose.x / max(odo_mal.pose.x, 1e-9):.2f}x")
    print("  (con velocidad constante casi no cambia; la diferencia aparece")
    print("   cuando la velocidad varia dentro del lote, p.ej. al acelerar)")

    print("\n=== hueco en los datos (reconexion) ===")
    odo = Odometry(cfg)
    t = dt_muestra
    for _ in range(20):
        odo.update(lote(t, rpm=(30, 30, 30, 30)))
        t += dt_lote
    x_antes = odo.pose.x
    t += 5.0                    # 5 segundos sin datos
    odo.update(lote(t, rpm=(30, 30, 30, 30)))
    salto = odo.pose.x - x_antes
    print(f"  avance inventado durante el hueco: {salto*1000:.1f} mm")
    assert salto < 0.05, "invento movimiento durante el corte"

    print("\n=== pose relativa (la usa el mapa) ===")
    a = Pose(1.0, 2.0, math.radians(30))
    b = Pose(2.0, 2.0, math.radians(30))
    rel = b.relative_to(a)
    print(f"  b desde a: x={rel.x:.3f} y={rel.y:.3f}")
    assert abs(rel.x - math.cos(math.radians(30))) < 1e-6
    assert abs(rel.y + math.sin(math.radians(30))) < 1e-6

    print("\n=== correccion por GPS: ruedas que sobreestiman 20% ===")
    cfg2 = OdometryConfig(gps_blend=1.0, min_gps_displacement_m=0.5)
    odo = Odometry(cfg2)
    lat0, lon0 = -34.9214, -57.9544
    R = 6378137.0
    t = dt_muestra
    for i in range(1, 81):
        b = lote(t, rpm=(30, 30, 30, 30))
        real = v * dt_lote * i * 0.8
        b["latitude"] = lat0 + math.degrees(real / R)
        b["longitude"] = lon0
        odo.update(b)
        t += dt_lote
    solo_ruedas = v * dt_lote * 80
    print(f"  solo ruedas:    {solo_ruedas:.3f} m")
    print(f"  real:           {solo_ruedas*0.8:.3f} m")
    print(f"  con correccion: {odo.pose.x:.3f} m ({odo.gps_corrections} correcciones)")
    assert odo.pose.x < solo_ruedas

    print("\n=== calibracion del eje del gyro ===")
    i, sg = calibrate_gyro_axis([[0.1, -0.2, 40.0, 0.0] for _ in range(20)])
    print(f"  eje {i}, signo {sg:+.0f}")
    assert i == 2 and sg > 0

    print("\n=== criterio 1: bias constante de 1.28 dps con rover en reposo (v=0) ===")
    # SIN el fix (bias=0.0): deriva linealmente porque 1.28 > deadband (0.5)
    cfg_sin_fix = OdometryConfig(gyro_yaw_bias_dps=0.0, gyro_deadband_dps=0.5, ekf_heading_correction=False)
    odo_sin_fix = Odometry(cfg_sin_fix)
    t = dt_muestra
    for _ in range(100):  # 10 segundos
        odo_sin_fix.update(lote(t, rpm=(0, 0, 0, 0), gyro_dps=1.28))
        t += dt_lote
    deriva_sin_fix_deg = math.degrees(odo_sin_fix.pose.theta)
    print(f"  SIN fix: deriva = {deriva_sin_fix_deg:+.2f} grados en 10 s (esperado ~12.8 gr)")
    assert deriva_sin_fix_deg > 10.0, "esperaba deriva significativa sin correccion de bias"

    # CON el fix (bias=1.2784): (1.28 - 1.2784) = 0.0016 < deadband (0.5) -> dps=0
    cfg_con_fix = OdometryConfig(gyro_yaw_bias_dps=1.2784, gyro_deadband_dps=0.5, ekf_heading_correction=False)
    odo_con_fix = Odometry(cfg_con_fix)
    t = dt_muestra
    for _ in range(100):  # 10 segundos
        odo_con_fix.update(lote(t, rpm=(0, 0, 0, 0), gyro_dps=1.28))
        t += dt_lote
    deriva_con_fix_deg = math.degrees(odo_con_fix.pose.theta)
    print(f"  CON fix (debiasing + deadband): deriva = {deriva_con_fix_deg:+.4f} grados")
    assert abs(deriva_con_fix_deg) < 1e-6, f"con debiasing deberia ser exactamente 0, dio {deriva_con_fix_deg}"

    print("\n=== criterio 2: correccion gradual hacia heading EKF (sin saltos discontinuos) ===")
    # Config con EKF correction habilitada, blend=0.3
    cfg_ekf = OdometryConfig(gyro_yaw_bias_dps=0.0, ekf_heading_correction=True, heading_blend=0.3)
    odo_ekf = Odometry(cfg_ekf)
    t = 100.0
    # Inicializacion: primera lectura a 345° con pose.theta = 0
    odo_ekf.update(lote(t, rpm=(0, 0, 0, 0), gyro_dps=0.0), ekf_heading=345.0, ekf_timestamp=t, now=t)
    assert abs(odo_ekf.pose.theta) < 1e-6
    assert odo_ekf.origin_heading == 345.0

    # Forzamos una divergencia artificial en pose.theta de +15 grados (+0.2618 rad)
    odo_ekf.pose.theta = math.radians(15.0)
    theta_inicial = odo_ekf.pose.theta

    # Primer ciclo de correccion: el heading EKF sigue reportando 345° (rumbo sin giro)
    t += 0.1
    odo_ekf.update(lote(t, rpm=(0, 0, 0, 0), gyro_dps=0.0), ekf_heading=345.0, ekf_timestamp=t, now=t)
    salto_1 = math.degrees(theta_inicial - odo_ekf.pose.theta)
    print(f"  Ciclo 1: theta pasa de 15.00 a {math.degrees(odo_ekf.pose.theta):.2f} gr (reduccion de {salto_1:.2f} gr, ~30%)")
    assert 4.0 < salto_1 < 5.0, f"esperaba reduccion suave de ~4.5 gr (30% de 15), dio {salto_1}"

    # Corremos 15 ciclos mas: debe converger suavemente
    for _ in range(15):
        t += 0.1
        odo_ekf.update(lote(t, rpm=(0, 0, 0, 0), gyro_dps=0.0), ekf_heading=345.0, ekf_timestamp=t, now=t)
    theta_final_deg = math.degrees(odo_ekf.pose.theta)
    print(f"  Tras 16 ciclos: theta = {theta_final_deg:.4f} gr (convergencia completa hacia 0)")
    assert abs(theta_final_deg) < 0.1, f"deberia haber convergido a < 0.1 gr, dio {theta_final_deg}"

    # Verificamos tambien el caso stale (> ekf_heading_max_age_s): NO debe corregir
    odo_ekf.pose.theta = math.radians(10.0)
    t_stale = t - 5.0  # dato de hace 5 segundos
    t += 0.1
    odo_ekf.update(lote(t, rpm=(0, 0, 0, 0), gyro_dps=0.0), ekf_heading=345.0, ekf_timestamp=t_stale, now=t)
    print(f"  Con dato EKF stale (age=5.1s): theta={math.degrees(odo_ekf.pose.theta):.2f} gr (sin cambios)")
    assert abs(math.degrees(odo_ekf.pose.theta) - 10.0) < 1e-6, "no debe corregir si el dato esta stale"

    print("\n=== criterio 3: simulacion del escenario real del log (130 iteraciones en reposo) ===")
    np.random.seed(42)
    n_iter = 130
    dt_iter = 0.1

    # Stack original (SIN correcciones):
    odo_original = Odometry(OdometryConfig(gyro_yaw_bias_dps=0.0, gyro_deadband_dps=0.5, ekf_heading_correction=False))
    # Stack con FIX (bias debiased + deadband + correccion EKF):
    odo_fixed = Odometry(OdometryConfig(gyro_yaw_bias_dps=1.2784, gyro_deadband_dps=0.5, ekf_heading_correction=True, heading_blend=0.3))

    t_sim = 1000.0
    ekf_h = 348.0  # Heading estable medido en el log real (347-348°)

    for step in range(n_iter):
        rpms, gyros = [], []
        for k in range(5):
            t_sample = t_sim + k * 0.02
            noise = float(np.random.normal(0.0, 0.055))
            gyros.append([0.08, -0.01, 1.2784 + noise, t_sample])
            rpms.append([0.0, 0.0, 0.0, 0.0, t_sample])
        b = {"rpms": rpms, "gyros": gyros}

        odo_original.update(b)
        odo_fixed.update(b, ekf_heading=ekf_h, ekf_timestamp=t_sim, now=t_sim)
        t_sim += dt_iter

    deriva_orig = math.degrees(odo_original.pose.theta)
    deriva_fix = math.degrees(odo_fixed.pose.theta)
    print(f"  En 130 iteraciones (~13 s):")
    print(f"    Original (sin fix): pose.theta deriva {deriva_orig:+.2f} grados (como en el log)")
    print(f"    Con fix (nuestro):  pose.theta deriva {deriva_fix:+.4f} grados")
    assert abs(deriva_orig) > 7.0, f"esperaba deriva > 7 gr en original, dio {deriva_orig}"
    print("\n=== criterio 1 (tilt): acelerometro en reposo nivelado ===")
    accels_nivelado = [[0.0, 0.0, 1.0, 0.0 + k * 0.02] for k in range(5)]
    res = estimate_roll_pitch(accels_nivelado)
    assert res is not None, "gate deberia pasar en reposo nivelado"
    roll, pitch, norm = res
    print(f"  Estimado: roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, norm={norm:.3f}g")
    assert abs(math.degrees(roll)) < 0.1
    assert abs(math.degrees(pitch)) < 0.1
    assert abs(norm - 1.0) < 1e-4

    print("\n=== criterio 2 (tilt): pitch simulado de 15° en reposo ===")
    pitch_sim_rad = math.radians(15.0)
    ax_sim = -math.sin(pitch_sim_rad)
    az_sim = math.cos(pitch_sim_rad)
    accels_pitch15 = [[ax_sim, 0.0, az_sim, 0.0 + k * 0.02] for k in range(5)]
    res15 = estimate_roll_pitch(accels_pitch15)
    assert res15 is not None, "gate deberia pasar en reposo inclinado 15° (|a|=1.0g)"
    roll15, pitch15, norm15 = res15
    print(f"  Estimado: roll={math.degrees(roll15):.2f}°, pitch={math.degrees(pitch15):.2f}°, norm={norm15:.3f}g")
    assert abs(math.degrees(roll15)) < 0.1
    assert abs(math.degrees(pitch15) - 15.0) < 0.1, f"esperaba pitch=15°, dio {math.degrees(pitch15):.2f}°"
    assert abs(norm15 - 1.0) < 1e-4

    print("\n=== criterio 3 (tilt): aceleracion fuerte (gate NO pasa, fallback a eje Z) ===")
    accels_acel = [[0.5, 0.0, 1.0, 0.0 + k * 0.02] for k in range(5)]
    res_acel = estimate_roll_pitch(accels_acel)
    print(f"  Aceleracion 0.5g: estimate_roll_pitch devolvio {res_acel} (esperado None, gate cerrado)")
    assert res_acel is None, "gate no deberia pasar durante aceleracion fuerte"

    cfg_fall = OdometryConfig(gyro_yaw_bias_dps=0.0, use_tilt_projection=True)
    odo_fall = Odometry(cfg_fall)
    t_base = 10.0
    lote_acel = {
        "accels": [[0.5, 0.0, 1.0, t_base + k * 0.02] for k in range(5)],
        "rpms": [[0, 0, 0, 0, t_base + k * 0.02] for k in range(5)],
        "gyros": [[50.0, 0.0, 30.0, t_base + k * 0.02] for k in range(5)],
    }
    odo_fall.update(lote_acel)
    assert not odo_fall.tilt_gate_open, "tilt_gate_open deberia ser False"
    # 4 intervalos de dt=0.02 integrados tras el primer timestamp de anclaje
    rate_integrated_dps = math.degrees(odo_fall.pose.theta) / (4 * 0.02)
    print(f"  Fallback activo: tasa integrada = {rate_integrated_dps:.2f} dps (esperado ~30 dps puro de eje Z, ignorando 50 dps de roll)")
    assert abs(rate_integrated_dps - 30.0) < 1.0, f"deberia integrar gz puro (30 dps), dio {rate_integrated_dps}"

    print("\n=== criterio 4 (tilt): degradacion de blend_efectivo a 15° de inclinacion ===")
    cfg_blend = OdometryConfig(
        gyro_yaw_bias_dps=0.0,
        heading_blend=0.30,
        tilt_blend_start_deg=5.0,
        tilt_blend_max_deg=20.0,
    )
    odo_blend = Odometry(cfg_blend)
    t = 200.0
    lote_p15 = {
        "accels": [[ax_sim, 0.0, az_sim, t + k * 0.02] for k in range(5)],
        "rpms": [[0, 0, 0, 0, t + k * 0.02] for k in range(5)],
        "gyros": [[0.0, 0.0, 0.0, t + k * 0.02] for k in range(5)],
    }
    odo_blend.update(lote_p15, ekf_heading=345.0, ekf_timestamp=t, now=t)
    assert odo_blend.tilt_gate_open
    odo_blend.pose.theta = math.radians(10.0)
    t += 0.1
    odo_blend.update(lote_p15, ekf_heading=345.0, ekf_timestamp=t, now=t)

    blend_nom = cfg_blend.heading_blend
    blend_efec = odo_blend.last_blend_effective
    print(f"  blend nominal:   {blend_nom:.3f}")
    print(f"  blend efectivo:  {blend_efec:.4f} (factor = {blend_efec / blend_nom:.2f})")
    assert blend_efec < blend_nom, f"blend efectivo ({blend_efec}) deberia ser menor que el nominal ({blend_nom})"
    assert abs(blend_efec - 0.075) < 0.005, f"esperaba blend_efectivo ~0.075, dio {blend_efec}"

    print("\n=== proyeccion horizontal de giro 3D en rampa de 15° ===")
    cfg_proj = OdometryConfig(gyro_yaw_bias_dps=0.0, use_tilt_projection=True)
    odo_proj = Odometry(cfg_proj)
    t_p = 50.0
    lote_proj = {
        "accels": [[ax_sim, 0.0, az_sim, t_p + k * 0.02] for k in range(5)],
        "rpms": [[0, 0, 0, 0, t_p + k * 0.02] for k in range(5)],
        "gyros": [[0.0, 0.0, 45.0, t_p + k * 0.02] for k in range(5)],
    }
    odo_proj.update(lote_proj)
    rate_proj = math.degrees(odo_proj.pose.theta) / (4 * 0.02)
    print(f"  gz crudo en chasis: 45.00 dps")
    print(f"  tasa proyectada:    {rate_proj:.2f} dps (esperado 45 * cos(15°) = {45.0 * math.cos(math.radians(15)):.2f} dps)")
    assert abs(rate_proj - 45.0 * math.cos(math.radians(15))) < 0.5


    print("\nTodos los asserts pasaron.")




if __name__ == "__main__":
    _self_test()
