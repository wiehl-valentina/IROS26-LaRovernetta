#!/usr/bin/env python3
"""Odometria por ruedas+giroscopo (+correccion GPS opcional), autocontenida.

Port 1:1 de la logica de `genie/genie_rover/odometry.py` (mismo formato de
entrada: lotes `/data` del SDK con "rpms"/"gyros"/"latitude"/"longitude"),
pero SIN importar el paquete `genie_rover` -- para que este bridge de ROS2
(carpeta `examples/ros2/`) no dependa de que el otro repo este en el
PYTHONPATH. Si algun dia se prefiere una unica fuente de verdad, este archivo
se puede borrar y reemplazar por `from genie_rover.odometry import Pose,
Odometry, OdometryConfig` (agregando `genie/` al sys.path).

Marco de referencia: x adelante, y izquierda, theta antihorario desde el eje
x inicial (el mismo marco que usa PersistentMap/planner del lado genie_rover
-- asi el mapa que arma este bridge queda directamente compatible).

Autoprueba (no necesita ROS2 ni robot):
    python3 differential_odometry.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RPM_A_RAD_S = 2.0 * math.pi / 60.0
EARTH_RADIUS_M = 6378137.0


def wrap_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def latlon_to_local_ne(lat0: float, lon0: float, lat1: float, lon1: float) -> tuple[float, float]:
    """Aproximacion equirectangular (valida para desplazamientos chicos,
    del orden de metros/decenas de metros -- de sobra para corregir deriva
    de odometria dentro de una sesion de mapeo)."""
    dlat = math.radians(lat1 - lat0)
    dlon = math.radians(lon1 - lon0)
    norte = dlat * EARTH_RADIUS_M
    este = dlon * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return norte, este


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
        dx, dy = self.x - other.x, self.y - other.y
        c, s = math.cos(-other.theta), math.sin(-other.theta)
        return Pose(c * dx - s * dy, s * dx + c * dy,
                    wrap_rad(self.theta - other.theta))


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
    gps_blend: float = 0.25
    gyro_yaw_index: int = 2
    gyro_sign: float = 1.0
    gyro_deadband_dps: float = 0.5
    max_dt_s: float = 0.5


class Odometry:
    """Integra telemetria del SDK (rpms/gyros/[lat,lon]) en una Pose."""

    def __init__(self, cfg: OdometryConfig):
        self.cfg = cfg
        self.pose = Pose()
        self._last_gyro_t: float | None = None
        self._origin_latlon: tuple[float, float] | None = None
        self._last_gps_pose: Pose | None = None
        self._last_gps_xy: tuple[float, float] | None = None
        self.gps_corrections = 0
        self.distance_travelled = 0.0
        self.samples_integrated = 0

    # ------------------------------------------------------------------ ruedas

    def _wheel_series(self, rpms: list) -> list[tuple[float, float]]:
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
        """Integra un lote de /data (o /ws/data). Devuelve la pose actualizada."""
        serie_v = self._wheel_series(telemetry_raw.get("rpms", []))
        serie_w_gyro = self._gyro_series(telemetry_raw.get("gyros", []))
        serie_w_ruedas = self._wheel_omega_series(telemetry_raw.get("rpms", []))
        serie_w = serie_w_gyro if self.cfg.use_gyro_for_rotation else serie_w_ruedas

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
        if abs(lat) > 90 or abs(lon) > 180:
            return

        if self._origin_latlon is None:
            self._origin_latlon = (lat, lon)
            self._last_gps_xy = (0.0, 0.0)
            self._last_gps_pose = Pose(self.pose.x, self.pose.y, self.pose.theta)
            return

        norte, este = latlon_to_local_ne(*self._origin_latlon, lat, lon)
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


# --------------------------------------------------------------------- pruebas

def _self_test() -> None:
    cfg = OdometryConfig()
    dt_muestra = 0.02
    por_lote = 5
    dt_lote = dt_muestra * por_lote

    def lote(t0, rpm=(0, 0, 0, 0), gyro_dps=0.0):
        rpms, gyros = [], []
        for k in range(por_lote):
            t = t0 + k * dt_muestra
            rpms.append([*rpm, t])
            gyros.append([0.0, 0.0, gyro_dps, t])
        return {"rpms": rpms, "gyros": gyros}

    print("=== 2 m en linea recta ===")
    odo = Odometry(cfg)
    v = 30 * RPM_A_RAD_S * cfg.wheel_radius_m
    n_lotes = int(round(2.0 / (v * dt_lote)))
    t = dt_muestra
    for _ in range(n_lotes):
        odo.update(lote(t, rpm=(30, 30, 30, 30)))
        t += dt_lote
    print(f"  x={odo.pose.x:.4f} y={odo.pose.y:.4f}")
    assert abs(odo.pose.x - 2.0) < 0.02
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

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
