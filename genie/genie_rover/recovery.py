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
    iteraciones comandando movimiento sin desplazarse. Se congela GeNIE y
    se resuelve la maniobra en dos etapas que usan fuentes DISTINTAS a
    proposito, porque la confiabilidad de la camara cambia con la distancia:

        1. retroceder -- la unica accion que reduce la ocupacion angular del
           obstaculo. Girar en el lugar no alcanza: el objeto sigue igual de
           cerca en cualquier orientacion, la zona ciega gira con el robot.
           ANTES de retroceder la camara sigue degradada: no se consulta.

        2. una vez retrocedido (~0.35 m mas lejos), un frame fresco YA es
           util. Se parte el frente en tres corredores (izquierda / centro /
           derecha) y se elige el de mayor espacio libre. Si la camara no
           tiene evidencia suficiente en ningun corredor (observed vacio, o
           todos por debajo del umbral de exito), se cae a
           PersistentMap.best_heading() como respaldo -- ahi si tiene sentido
           confiar en la memoria acumulada en vez de un frame sin datos.

        3. girar en pasos CHICOS (turn_step_deg) hacia el corredor elegido,
           re-verificando con la camara despues de cada paso. Si un paso no
           alcanza, se sigue girando para el mismo lado; nunca se compromete
           de una sola vez a un heading grande calculado de antemano, porque
           ese heading pudo haberse calculado con datos ya viejos.

    Los headings candidatos NUNCA incluyen algo cercano a 90/135/180: este
    robot no tiene sensores traseros ni reversa util, girar tanto significa
    quedar mirando a ciegas hacia donde no se sabe que hay.

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

from .navigation import CorridorClearances, DriveCommand, corridor_clearances
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
    # ---- retroceso -----------------------------------------------------
    reverse_distance_m: float = 0.35
    reverse_speed: float = 0.15
    reverse_timeout_s: float = 8.0
    # ---- corredor por camara (post-retroceso) --------------------------
    corridor_band_width_m: float = 0.30
    corridor_check_m: float = 1.2
    turn_step_deg: float = 15.0
    max_local_turns: int = 4
    # ---- respaldo por mapa persistente (solo si la camara no alcanza) --
    heading_search_radius_m: float = 2.0
    heading_fan_deg: float = 20.0
    fallback_heading_candidates_deg: tuple[float, ...] = (-45.0, -30.0, 30.0, 45.0)
    # ---- rotacion (comun a ambas fuentes) -------------------------------
    align_tolerance_deg: float = 12.0
    rotate_timeout_s: float = 6.0
    turn_speed: float = 0.35
    # ---- salida / relock ------------------------------------------
    replan_lock_s: float = 2.0
    success_clearance_m: float = 0.6        # clearance minimo para dar la maniobra por exitosa
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

    # ------------------------------------------------- eleccion de corredor

    def _choose_from_camera(self, corridors: CorridorClearances) -> float | None:
        """Devuelve el heading relativo (grados) a intentar segun la camara,
        o None si ningun corredor tiene evidencia suficiente para decidir.

        BUGFIX: la version anterior elegia CENTRO en cuanto superaba el
        umbral, sin comparar contra izquierda/derecha -- si centro daba
        0.65 m e izquierda 1.5 m (umbral 0.6 m), se quedaba con centro pese
        a que izquierda tenia mucho mas espacio, contradiciendo el docstring
        del modulo ("se elige el de mayor espacio libre"). Ahora se compara
        el maximo real entre los tres candidatos.
        """
        opciones = {
            0.0: corridors.center,
            self.cfg.turn_step_deg: corridors.left,
            -self.cfg.turn_step_deg: corridors.right,
        }
        mejor_heading, mejor_clearance = max(opciones.items(), key=lambda kv: kv[1])
        if mejor_clearance < self.cfg.success_clearance_m:
            return None
        return mejor_heading

    # ---------------------------------------------------------- maniobra

    def execute(self, *, pmap: PersistentMap,
               get_pose: Callable[[], Pose],
               capture_bev: Callable[[], tuple[np.ndarray, float]],
               compute_clearance: Callable[[], float],
               send: Callable[[DriveCommand], None],
               request_intervention: Callable[[], None],
               stopped_flag: Callable[[], bool]) -> str:
        """Corre la maniobra completa. Bloqueante, igual que _recover() /
        _unstick() en bridge.py: son maniobras cortas donde no vale la pena
        intercalar el resto del bucle.

        capture_bev() debe capturar un frame fresco y correr percepcion,
        devolviendo (bev, resolution_m) -- se usa para elegir de que lado
        girar DESPUES del retroceso.

        compute_clearance() debe capturar un frame fresco y devolver
        front_clearance_m(...) -- se usa para verificar el resultado de cada
        paso de giro.

        Devuelve "ok", "corredor_bloqueado" o "intervencion".
        """
        self.regime = Regime.CERCANO
        print("[cercano] frente cerrado o atascado: freno GeNIE")

        # 1) retroceder -- la camara sigue degradada aca, no se consulta ----
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

        # 2) elegir corredor con un frame fresco (post-retroceso) -----------
        pose = get_pose()
        bev, res_m = capture_bev()
        corridors = corridor_clearances(
            bev, res_m,
            band_width_m=self.cfg.corridor_band_width_m,
            max_check_m=self.cfg.corridor_check_m,
        )
        print(f"[cercano] corredores (camara): izq={corridors.left:.2f} "
              f"centro={corridors.center:.2f} der={corridors.right:.2f}")

        heading_deg = self._choose_from_camera(corridors)
        if heading_deg is None:
            # la camara no tiene evidencia suficiente: respaldo en el mapa,
            # con headings acotados (nunca cerca de 90/135/180)
            heading_deg, scores = pmap.best_heading(
                pose,
                candidate_headings_deg=list(self.cfg.fallback_heading_candidates_deg),
                radius_m=self.cfg.heading_search_radius_m,
                fan_deg=self.cfg.heading_fan_deg,
            )
            legible = {k: f"{v:.2f} m libres" for k, v in scores.items()}
            print(f"[cercano] camara sin evidencia suficiente, uso mapa: {legible}")
        print(f"[cercano] primer paso: {heading_deg:+.0f} grados relativos")

        # 3) girar en pasos chicos, re-verificando la camara en cada uno ----
        exito = False
        for intento in range(1, self.cfg.max_local_turns + 1):
            if abs(heading_deg) > 1e-6:
                target_theta = get_pose().theta + math.radians(heading_deg)
                t0 = time.time()
                while not stopped_flag() and (time.time() - t0) < self.cfg.rotate_timeout_s:
                    p = get_pose()
                    err_deg = math.degrees(_wrap_rad(target_theta - p.theta))
                    if abs(err_deg) <= self.cfg.align_tolerance_deg:
                        break
                    ang = self.angular_sign * math.copysign(self.cfg.turn_speed, err_deg)
                    send(DriveCommand(0.0, ang,
                                      f"regimen cercano: paso de giro {intento} ({err_deg:+.0f} grados)"))
                    time.sleep(0.1)
                send(DriveCommand(0.0, 0.0, "regimen cercano: fin del paso de giro"))

            clearance = compute_clearance()
            print(f"[cercano] intento {intento}/{self.cfg.max_local_turns}: "
                  f"clearance {clearance:.2f} m")
            if clearance >= self.cfg.success_clearance_m:
                exito = True
                break

            # no alcanzo: seguir girando para el mismo lado, otro paso chico
            lado = heading_deg if abs(heading_deg) > 1e-6 else self.cfg.turn_step_deg
            heading_deg = math.copysign(self.cfg.turn_step_deg, lado)

        # 4) resolver el resultado -------------------------------------------
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
            self._block_current_corridor(pose, math.radians(heading_deg), now)
            print(f"[cercano] bloqueo el corredor por {self.cfg.corridor_block_s:.0f} s "
                  "y dejo que LEJANO replanifique alrededor")
            resultado = "corredor_bloqueado"

        self._lock_until = now + self.cfg.replan_lock_s
        self.regime = Regime.LEJANO
        return resultado


# --------------------------------------------------------------------- test

def _self_test() -> None:
    from .persistent_map import MapConfig

    print("=== eleccion de corredor desde un BEV sintetico (camara) ===")
    bev_izq_libre = np.ones((134, 134), dtype=np.float32)
    bev_izq_libre[90:120, 55:90] = 0.0  # bloqueado centro/derecha, izquierda libre
    ctrl = NearRegimeController(NearRegimeConfig(success_clearance_m=0.5))
    corridors = corridor_clearances(bev_izq_libre, 0.015, band_width_m=0.3, max_check_m=1.2)
    print(f"  corredores: izq={corridors.left:.2f} centro={corridors.center:.2f} "
          f"der={corridors.right:.2f}")
    heading = ctrl._choose_from_camera(corridors)
    print(f"  heading elegido: {heading}")
    assert heading is not None and heading > 0, "deberia preferir izquierda (heading positivo)"

    print("\n=== camara sin evidencia suficiente -> None (cae a mapa) ===")
    bev_todo_cerca = np.ones((134, 134), dtype=np.float32)
    bev_todo_cerca[100:134, :] = 0.0
    corridors2 = corridor_clearances(bev_todo_cerca, 0.015, band_width_m=0.3, max_check_m=1.2)
    heading2 = ctrl._choose_from_camera(corridors2)
    print(f"  corredores: izq={corridors2.left:.2f} centro={corridors2.center:.2f} "
          f"der={corridors2.right:.2f} -> heading={heading2}")
    assert heading2 is None

    print("\n=== deteccion de estancamiento ===")
    ctrl3 = NearRegimeController(NearRegimeConfig(stall_iters=3, stall_disp_m=0.05))
    quieto = Pose(1.0, 1.0, 0.0)
    for t in range(3):
        ctrl3.note_iteration(quieto, DriveCommand(0.0, 0.3, "girando"), float(t))
    print(f"  estancado (girando sin avanzar): {ctrl3.is_stalled()}")
    assert ctrl3.is_stalled()

    ctrl4 = NearRegimeController(NearRegimeConfig(stall_iters=3, stall_disp_m=0.05))
    for t in range(3):
        p = Pose(1.0 + 0.1 * t, 1.0, 0.0)
        ctrl4.note_iteration(p, DriveCommand(0.3, 0.0, "avanzando"), float(t))
    print(f"  avanzando de verdad: {ctrl4.is_stalled()}")
    assert not ctrl4.is_stalled()

    print("\n=== enmascarado de corredor bloqueado ===")
    ctrl5 = NearRegimeController(NearRegimeConfig(corridor_block_s=30.0))
    now = time.time()
    ctrl5._block_current_corridor(Pose(0, 0, 0), 0.0, now)  # bloquea "adelante"
    bev_libre = np.ones((134, 134), dtype=np.float32)
    masked = ctrl5.mask_blocked_corridors(bev_libre, Pose(0, 0, 0), 0.03, now)
    celdas_bloqueadas = int((masked == 0.0).sum())
    print(f"  celdas puestas en 0 por el bloqueo: {celdas_bloqueadas}")
    assert celdas_bloqueadas > 0
    assert ctrl5.has_blocked_corridors(now)
    assert not ctrl5.has_blocked_corridors(now + 31.0), "el bloqueo deberia expirar"

    print("\n=== ejecucion completa con fakes (sin robot) ===")
    m = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    pose_fake = Pose(0.0, 0.0, 0.0)
    sent: list[DriveCommand] = []

    def _send(cmd: DriveCommand) -> None:
        sent.append(cmd)
        # simular que el robot efectivamente retrocede/gira un poco
        if cmd.linear < 0:
            pose_fake.x -= 0.02
        pose_fake.theta += cmd.angular * 0.1

    bev_seq = iter([bev_izq_libre, bev_izq_libre, bev_izq_libre])

    def _capture_bev():
        try:
            return next(bev_seq), 0.015
        except StopIteration:
            return bev_izq_libre, 0.015

    clearances_seq = iter([0.2, 0.7])  # primer chequeo insuficiente, segundo ok

    def _compute_clearance():
        try:
            return next(clearances_seq)
        except StopIteration:
            return 0.7

    ctrl6 = NearRegimeController(NearRegimeConfig(
        reverse_distance_m=0.01, reverse_timeout_s=0.5,
        rotate_timeout_s=0.5, align_tolerance_deg=180.0,  # no bloquear en el giro fake
        success_clearance_m=0.5, max_local_turns=3,
    ))
    resultado = ctrl6.execute(
        pmap=m, get_pose=lambda: pose_fake, capture_bev=_capture_bev,
        compute_clearance=_compute_clearance, send=_send,
        request_intervention=lambda: None, stopped_flag=lambda: False,
    )
    print(f"  resultado: {resultado}, regimen final: {ctrl6.regime}")
    assert resultado == "ok"
    assert ctrl6.regime == Regime.LEJANO

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()