"""Regimen cercano: que hacer cuando el BEV instantaneo deja de ser confiable.

Por debajo de ~0.6 m la proyeccion a BEV esta geometricamente degradada: el
obstaculo ocupa la mayor parte del campo visual, la distorsion de lente es
maxima justo en los bordes (donde quedaria el hueco libre), y los caminos
candidatos del planner arrancan todos atravesandolo. El planner no esta roto
en ese punto: esta haciendo lo correcto con datos que ya no sirven para
planificar. Tocar sus umbrales no arregla nada ahi.

Este modulo es el segundo regimen. Dos reglas separan LEJANO de CERCANO:

    LEJANO (normal): front_clearance_m() por encima del umbral -> bridge.py
    sigue planificando con GeNIE como siempre. Este modulo solo le agrega,
    ANTES de planificar, el enmascarado de corredores bloqueados (ver abajo).

    CERCANO (este modulo): el frente se cierra, o el robot lleva varias
    iteraciones comandando movimiento sin desplazarse (el planner puede estar
    replanificando un camino distinto cada frame sin que el robot avance).
    Se congela GeNIE y se deja de consultar el BEV fresco para decidir hacia
    donde ir:

        1. retroceder -- la unica accion que reduce la ocupacion angular del
           obstaculo. Girar en el lugar no alcanza: el objeto sigue igual de
           cerca en cualquier orientacion, la zona ciega gira con el robot.
        2. elegir, SOLO del mapa persistente (no de la camara), el rumbo con
           mas area libre acumulada dentro de un radio corto
        3. rotar hacia ese rumbo
        4. volver a LEJANO con un bloqueo de re-disparo corto, para no
           oscilar entre regimenes en el limite del umbral

Escalamiento (para no reintentar para siempre contra lo mismo):
    fail_block_after fallos seguidos    -> el corredor elegido se marca
        bloqueado por corridor_block_s: LEJANO lo va a ver marcado
        intransitable en el BEV que le pasa el planner, y GeNIE debe rodearlo
        en vez de volver a mandar al robot ahi.
    fail_intervention_after fallos seguidos -> se pide intervencion humana.

Autoprueba (no necesita robot):
    python -m genie_rover.recovery
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .navigation import DriveCommand
from .odometry import Pose
from .persistent_map import PersistentMap


class Regime(Enum):
    LEJANO = "lejano"
    CERCANO = "cercano"


@dataclass
class NearRegimeConfig:
    # ---- disparo ----------------------------------------------------------
    clearance_thresh_m: float = 0.6
    clearance_check_m: float = 1.2          # horizonte de front_clearance_m
    stall_iters: int = 5
    stall_disp_m: float = 0.05
    # ---- maniobra -----------------------------------------------------
    reverse_distance_m: float = 0.35
    reverse_speed: float = 0.15
    reverse_timeout_s: float = 8.0
    heading_search_radius_m: float = 2.0
    heading_fan_deg: float = 20.0
    heading_candidates_deg: tuple[float, ...] = (
        0, 30, -30, 60, -60, 90, -90, 135, -135, 180,
    )
    align_tolerance_deg: float = 12.0
    rotate_timeout_s: float = 6.0
    turn_speed: float = 0.35
    # ---- salida / relock ------------------------------------------
    replan_lock_s: float = 2.0
    success_clearance_m: float = 0.6        # clearance minimo tras la maniobra para considerarla exitosa
    # ---- escalamiento --------------------------------------------------
    corridor_block_s: float = 30.0
    corridor_half_width_m: float = 0.35
    fail_block_after: int = 2
    fail_intervention_after: int = 3


@dataclass
class _BlockedCorridor:
    x: float
    y: float
    heading_rad: float           # absoluto, marco de mundo
    until_t: float


def _wrap_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class NearRegimeController:
    """Maquina de estados LEJANO/CERCANO.

    No tiene I/O propio (ni cliente HTTP ni sleeps sueltos fuera de execute):
    todo pasa por los callbacks que le da bridge.py, para no duplicar el
    cliente del SDK ni la logica de frenado que ya existe ahi.
    """

    def __init__(self, cfg: NearRegimeConfig, angular_sign: float = -1.0):
        self.cfg = cfg
        self.angular_sign = float(angular_sign)
        self.regime = Regime.LEJANO
        self._lock_until = 0.0
        self._pose_hist: deque[tuple[float, float, float]] = deque(maxlen=cfg.stall_iters)
        self._fail_count = 0
        self._blocked: list[_BlockedCorridor] = []

    # ------------------------------------------------------------ deteccion

    def note_iteration(self, pose: Pose, cmd: DriveCommand, t: float) -> None:
        """Llamar una vez por iteracion del bucle LEJANO, con el comando que
        se acaba de enviar. Si el comando es (0,0) no cuenta como intento de
        avance, asi que no ensucia la deteccion de estancamiento."""
        moving_cmd = abs(cmd.linear) > 1e-3 or abs(cmd.angular) > 1e-3
        if not moving_cmd:
            self._pose_hist.clear()
            return
        self._pose_hist.append((pose.x, pose.y, t))

    def is_stalled(self) -> bool:
        """Comandando movimiento pero sin desplazamiento neto en las ultimas
        stall_iters iteraciones. Cubre el caso donde el planner sigue
        devolviendo un camino "valido" -- distinto cada vez -- pero el robot
        no gana metros, tipico de estar contra un obstaculo fuera del cono
        que chequea front_is_blocked."""
        if len(self._pose_hist) < self.cfg.stall_iters:
            return False
        x0, y0, _ = self._pose_hist[0]
        x1, y1, _ = self._pose_hist[-1]
        return math.hypot(x1 - x0, y1 - y0) < self.cfg.stall_disp_m

    def should_enter_near(self, clearance_m: float, now: float) -> bool:
        if now < self._lock_until:
            return False
        return clearance_m < self.cfg.clearance_thresh_m or self.is_stalled()

    # -------------------------------------------------------- corredores

    def _purge_expired(self, now: float) -> None:
        self._blocked = [b for b in self._blocked if b.until_t > now]

    def has_blocked_corridors(self, now: float) -> bool:
        self._purge_expired(now)
        return bool(self._blocked)

    def mask_blocked_corridors(self, bev: np.ndarray, pose: Pose,
                               resolution_m: float, now: float) -> np.ndarray:
        """Copia el BEV con los corredores bloqueados vigentes pintados
        intransitables, para que el planner de LEJANO los rodee en vez de
        volver a mandar al robot justo ahi apenas se destraba.

        Se llama SIEMPRE antes de planificar en LEJANO (no solo tras una
        recuperacion). Trabaja sobre una copia del BEV que se le pasa al
        planner, no sobre el mapa persistente: el mapa debe seguir
        reflejando lo que la camara realmente ve.
        """
        self._purge_expired(now)
        if not self._blocked:
            return bev
        out = bev.copy()
        h, w = out.shape
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        half = max(1, int(self.cfg.corridor_half_width_m / resolution_m))
        for b in self._blocked:
            dx, dy = b.x - pose.x, b.y - pose.y
            x_r = c * dx + s * dy          # adelante, marco robot
            y_r = -s * dx + c * dy         # izquierda, marco robot
            head_r = b.heading_rad - pose.theta
            for d in np.linspace(0.0, self.cfg.heading_search_radius_m, 24):
                px = x_r + d * math.cos(head_r)
                py = y_r + d * math.sin(head_r)
                row = h - 1 - int(px / resolution_m)
                col = w // 2 - int(py / resolution_m)
                r0, r1 = max(0, row - half), min(h, row + half + 1)
                c0, c1 = max(0, col - half), min(w, col + half + 1)
                if r0 < r1 and c0 < c1:
                    out[r0:r1, c0:c1] = 0.0
        return out

    def _block_current_corridor(self, pose: Pose, heading_rad_robot: float, now: float) -> None:
        heading_abs = pose.theta + heading_rad_robot
        self._blocked.append(_BlockedCorridor(pose.x, pose.y, heading_abs,
                                              now + self.cfg.corridor_block_s))

    # ---------------------------------------------------------- maniobra

    def execute(self, *, pmap: PersistentMap,
               get_pose: Callable[[], Pose],
               compute_clearance: Callable[[], float],
               send: Callable[[DriveCommand], None],
               request_intervention: Callable[[], None],
               stopped_flag: Callable[[], bool]) -> str:
        """Corre la maniobra completa. Bloqueante, igual que _recover() /
        _unstick() en bridge.py: son maniobras cortas donde no vale la pena
        intercalar el resto del bucle.

        compute_clearance() debe capturar un frame fresco, correr percepcion,
        y devolver front_clearance_m(...) -- se usa solo para verificar el
        resultado, nunca para elegir el rumbo.

        Devuelve "ok", "corredor_bloqueado" o "intervencion".
        """
        self.regime = Regime.CERCANO
        print("[cercano] frente cerrado o atascado: freno GeNIE, "
              "trabajo solo con el mapa persistente")

        # 1) retroceder -----------------------------------------------------
        send(DriveCommand(-abs(self.cfg.reverse_speed), 0.0,
                          "regimen cercano: retrocediendo"))
        t0 = time.time()
        start_pose = get_pose()
        recorrido = 0.0
        while recorrido < self.cfg.reverse_distance_m:
            if stopped_flag() or (time.time() - t0) > self.cfg.reverse_timeout_s:
                break
            time.sleep(0.1)
            p = get_pose()
            recorrido = math.hypot(p.x - start_pose.x, p.y - start_pose.y)
        send(DriveCommand(0.0, 0.0, "regimen cercano: fin del retroceso"))
        print(f"[cercano] retrocedi {recorrido:.2f} m")

        # 2) elegir rumbo con mas area libre, SOLO del mapa persistente -----
        pose = get_pose()
        best_heading_deg, scores = pmap.best_heading(
            pose,
            candidate_headings_deg=list(self.cfg.heading_candidates_deg),
            radius_m=self.cfg.heading_search_radius_m,
            fan_deg=self.cfg.heading_fan_deg,
        )
        legible = {k: f"{v:.2f} m libres" for k, v in scores.items()}
        print(f"[cercano] rumbos evaluados (mapa): {legible}")
        print(f"[cercano] elijo {best_heading_deg:+.0f} grados relativos "
              f"({scores[best_heading_deg]:.2f} m libres por delante)")

        # 3) rotar hacia ese rumbo -------------------------------------------
        target_theta = pose.theta + math.radians(best_heading_deg)
        t0 = time.time()
        while not stopped_flag() and (time.time() - t0) < self.cfg.rotate_timeout_s:
            p = get_pose()
            err_deg = math.degrees(_wrap_rad(target_theta - p.theta))
            if abs(err_deg) <= self.cfg.align_tolerance_deg:
                break
            ang = self.angular_sign * math.copysign(self.cfg.turn_speed, err_deg)
            send(DriveCommand(0.0, ang, f"regimen cercano: rotando ({err_deg:+.0f} grados)"))
            time.sleep(0.1)
        send(DriveCommand(0.0, 0.0, "regimen cercano: fin de la rotacion"))

        # 4) verificar con un frame fresco si sirvio -------------------------
        clearance = compute_clearance()
        exito = clearance >= self.cfg.success_clearance_m
        now = time.time()

        if exito:
            self._fail_count = 0
            self._lock_until = now + self.cfg.replan_lock_s
            self.regime = Regime.LEJANO
            self._pose_hist.clear()
            print(f"[cercano] recuperado (clearance {clearance:.2f} m), vuelvo a LEJANO")
            return "ok"

        self._fail_count += 1
        print(f"[cercano] recuperacion sin exito (clearance {clearance:.2f} m); "
              f"fallo {self._fail_count}/{self.cfg.fail_intervention_after}")

        if self._fail_count >= self.cfg.fail_intervention_after:
            print("[cercano] demasiados intentos fallidos: pido intervencion")
            try:
                request_intervention()
            except Exception as exc:
                print(f"[cercano] no pude pedir intervencion: {exc}")
            self._fail_count = 0
            self._lock_until = now + self.cfg.replan_lock_s
            self.regime = Regime.LEJANO
            return "intervencion"

        resultado = "ok"
        if self._fail_count >= self.cfg.fail_block_after:
            self._block_current_corridor(pose, math.radians(best_heading_deg), now)
            print(f"[cercano] bloqueo el corredor por {self.cfg.corridor_block_s:.0f} s "
                  "y dejo que LEJANO replanifique alrededor")
            resultado = "corredor_bloqueado"

        self._lock_until = now + self.cfg.replan_lock_s
        self.regime = Regime.LEJANO
        return resultado


# --------------------------------------------------------------------- test

def _self_test() -> None:
    from .persistent_map import MapConfig

    print("=== eleccion de rumbo desde un mapa sintetico ===")
    m = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    bev = np.ones((134, 134), dtype=np.float32)
    obs = np.ones((134, 134), dtype=np.uint8)
    # una maceta angosta a ~0.2-0.6 m adelante, centrada -- no una pared que
    # tapa todo el frente: eso es justamente el escenario de la seccion 04
    bev[90:120, 55:80] = 0.0
    pose = Pose(0.0, 0.0, 0.0)
    m.integrate(bev, obs, pose, 2.0, 1.2, t=0.0)

    best, scores = m.best_heading(pose, [0, 90, -90, 180], radius_m=1.5, fan_deg=18)
    print(f"  rumbo elegido: {best:+.0f} grados, scores={ {k: round(v,2) for k,v in scores.items()} }")
    assert scores[0] < scores[90] or scores[0] < scores[-90], (
        "deberia preferir un costado antes que seguir de frente contra la maceta")

    print("\n=== deteccion de estancamiento ===")
    ctrl = NearRegimeController(NearRegimeConfig(stall_iters=3, stall_disp_m=0.05))
    quieto = Pose(1.0, 1.0, 0.0)
    for t in range(3):
        ctrl.note_iteration(quieto, DriveCommand(0.0, 0.3, "girando"), float(t))
    print(f"  estancado (girando sin avanzar): {ctrl.is_stalled()}")
    assert ctrl.is_stalled()

    ctrl2 = NearRegimeController(NearRegimeConfig(stall_iters=3, stall_disp_m=0.05))
    for t in range(3):
        p = Pose(1.0 + 0.1 * t, 1.0, 0.0)
        ctrl2.note_iteration(p, DriveCommand(0.3, 0.0, "avanzando"), float(t))
    print(f"  avanzando de verdad: {ctrl2.is_stalled()}")
    assert not ctrl2.is_stalled()

    print("\n=== enmascarado de corredor bloqueado ===")
    ctrl3 = NearRegimeController(NearRegimeConfig(corridor_block_s=30.0))
    now = time.time()
    ctrl3._block_current_corridor(Pose(0, 0, 0), 0.0, now)  # bloquea "adelante"
    bev_libre = np.ones((134, 134), dtype=np.float32)
    masked = ctrl3.mask_blocked_corridors(bev_libre, Pose(0, 0, 0), 0.03, now)
    celdas_bloqueadas = int((masked == 0.0).sum())
    print(f"  celdas puestas en 0 por el bloqueo: {celdas_bloqueadas}")
    assert celdas_bloqueadas > 0
    assert ctrl3.has_blocked_corridors(now)
    assert not ctrl3.has_blocked_corridors(now + 31.0), "el bloqueo deberia expirar"

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
