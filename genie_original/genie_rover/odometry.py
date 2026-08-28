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
from dataclasses import dataclass, field

import numpy as np

from .navigation import latlon_to_local_ne, wrap_deg

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
    # Eje del giroscopo que mide guiñada (rotacion en el plano). Se determina
    # empiricamente con calibrate_gyro_axis().
    gyro_yaw_index: int = 2
    gyro_sign: float = 1.0
    # Las lecturas por debajo de esto se toman como ruido y se descartan, para
    # que el robot quieto no acumule deriva.
    gyro_deadband_dps: float = 0.5
    max_dt_s: float = 0.5


class Odometry:
    """Integra telemetria en una pose. Alimentar con update() en cada ciclo."""

    def __init__(self, cfg: OdometryConfig):
        self.cfg = cfg
        self.pose = Pose()
        self._last_gyro_t: float | None = None
        self._last_wheel_t: float | None = None
        self._origin_latlon: tuple[float, float] | None = None
        self._last_gps_pose: Pose | None = None
        self._last_gps_xy: tuple[float, float] | None = None
        self.gps_corrections = 0
        self.distance_travelled = 0.0
        self.samples_integrated = 0

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

    def _gyro_series(self, gyros: list) -> list[tuple[float, float]]:
        """Convierte el lote del giroscopo en [(timestamp, omega_rad_s), ...]."""
        salida = []
        i = self.cfg.gyro_yaw_index
        for fila in gyros:
            if len(fila) <= max(i, 3):
                continue
            dps = float(fila[i]) * self.cfg.gyro_sign
            t = float(fila[-1])
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

    def update(self, telemetry_raw: dict) -> Pose:
        """Integra un lote de /data. Devuelve la pose actualizada.

        Cada llamada trae varias muestras: se integran una por una, usando el
        intervalo real entre timestamps consecutivos.
        """
        serie_v = self._wheel_series(telemetry_raw.get("rpms", []))
        serie_w_gyro = self._gyro_series(telemetry_raw.get("gyros", []))
        serie_w_ruedas = self._wheel_omega_series(telemetry_raw.get("rpms", []))
        serie_w = serie_w_gyro if self.cfg.use_gyro_for_rotation else serie_w_ruedas

        # Instantes a integrar: la union de ambos relojes, para no perder nada.
        tiempos = sorted({t for t, _ in serie_v} | {t for t, _ in serie_w})
        if not tiempos:
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
        mezcla = 1.0 + self.cfg.gps_blend * (escala - 1.0)

        dx = self.pose.x - self._last_gps_pose.x
        dy = self.pose.y - self._last_gps_pose.y
        self.pose.x = self._last_gps_pose.x + dx * mezcla
        self.pose.y = self._last_gps_pose.y + dy * mezcla

        self.gps_corrections += 1
        self._last_gps_xy = gps_xy
        self._last_gps_pose = Pose(self.pose.x, self.pose.y, self.pose.theta)

    def reset(self) -> None:
        self.pose = Pose()
        self._last_gyro_t = None
        self._origin_latlon = None
        self._last_gps_pose = None
        self._last_gps_xy = None
        self.distance_travelled = 0.0
        self.samples_integrated = 0


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
    cfg = OdometryConfig()
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

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
