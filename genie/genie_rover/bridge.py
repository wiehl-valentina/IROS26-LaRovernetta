"""Bucle de control: Earth Rover SDK <-> SAM-TP <-> planner BEV de GeNIE.

Por defecto arranca en DRY-RUN: calcula todo e imprime los comandos sin
enviarlos. Para que el robot se mueva de verdad hace falta pasar --go.

    # simulacro, el robot no se mueve
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml

    # de verdad
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml --go \
        --start-mission --max-seconds 120 --debug-dir debug/run1
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from genie_path_planner.planner import PlannerConfig, plan_on_bev

from .navigation import (
    DriveCommand,
    HeadingEstimator,
    PathFollower,
    front_is_blocked,
    goal_from_gps,
)
from .odometry import Odometry, OdometryConfig, Pose
from .perception import PerceptionPipeline
from .persistent_map import MapConfig, PersistentMap
from .sdk_client import Checkpoint, RoverClient, RoverError
from .console_report import MissionConsoleReporter

@dataclass
class LoopStats:
    iterations: int = 0
    plans_ok: int = 0
    plans_empty: int = 0
    blocked: int = 0
    errors: int = 0
    unstucks: int = 0


class Bridge:
    # IndoorBridge (mision sin GPS, busca un cono) pisa esto a False: no tiene
    # sentido leer/anunciar checkpoints GPS en una mision que nunca los usa.
    uses_gps_checkpoints = True

    def __init__(self, cfg: dict, dry_run: bool = True, debug_dir: str | None = None):
        self.cfg = cfg
        self.dry_run = bool(dry_run)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.client = RoverClient(cfg["rover"]["base_url"], timeout=cfg["rover"].get("timeout_s", 5.0))
        self.perception = PerceptionPipeline(cfg)

        nav = cfg["navigation"]
        self.heading_est = HeadingEstimator(
            min_displacement_m=nav.get("heading_min_displacement_m", 1.5),
            orientation_offset_deg=nav.get("orientation_offset_deg", 0.0),
            orientation_sign=nav.get("orientation_sign", 1.0),
            trust_orientation=nav.get("trust_orientation", False),
        )
        self.follower = PathFollower(
            lookahead_m=nav.get("lookahead_m", 1.0),
            align_threshold_deg=nav.get("align_threshold_deg", 25.0),
            max_linear=nav["max_linear"],
            max_angular=nav["max_angular"],
            turn_speed=nav.get("turn_speed", 0.35),
            kp_angular=nav.get("kp_angular", 0.9),
            angular_sign=nav["angular_sign"],
        )
        self.goal_range_m = float(nav.get("goal_range_m", 3.5))
        self.claim_radius_m = float(nav.get("claim_radius_m", 8.0))

        self.planner_cfg = PlannerConfig(**cfg.get("planner", {}))
        self.resolution = float(cfg["projection"]["resolution_m_per_px"])
        self.forward_range = float(cfg["projection"]["forward_range_m"])
        self.side_range = float(cfg["projection"]["side_range_m"])

        # ---- memoria espacial ------------------------------------------------
        # Sin esto el planner ve una foto de 2x2 m que se descarta en cada
        # frame: no recuerda el obstaculo que acaba de salir de cuadro, ni que
        # ya giro buscando salida.
        mem = cfg.get("memory", {})
        self.use_map = bool(mem.get("enabled", True))
        self.odometry = None
        self.pmap = None
        if self.use_map:
            odo_cfg = cfg.get("odometry", {})
            self.odometry = Odometry(OdometryConfig(
                wheel_radius_m=float(odo_cfg.get("wheel_radius_m", 0.045)),
                track_width_m=float(odo_cfg.get("track_width_m", 0.15)),
                left_rpm_indices=tuple(odo_cfg.get("left_rpm_indices", (0, 2))),
                right_rpm_indices=tuple(odo_cfg.get("right_rpm_indices", (1, 3))),
                rotation_sign=float(odo_cfg.get("rotation_sign", -1.0)),
                use_gyro_for_rotation=bool(odo_cfg.get("use_gyro_for_rotation", True)),
                gps_correction=bool(odo_cfg.get("gps_correction", True)),
                min_gps_displacement_m=float(odo_cfg.get("min_gps_displacement_m", 1.0)),
                gyro_yaw_index=int(odo_cfg.get("gyro_yaw_index", 2)),
                gyro_sign=float(odo_cfg.get("gyro_sign", 1.0)),
            ))
            self.pmap = PersistentMap(MapConfig(
                size_m=float(mem.get("map_size_m", 8.0)),
                resolution_m_per_px=self.resolution,
                update_weight=float(mem.get("update_weight", 0.45)),
                decay_per_s=float(mem.get("decay_per_s", 0.08)),
                recenter_margin_m=float(mem.get("recenter_margin_m", 1.5)),
                min_confidence=float(mem.get("min_confidence", 0.15)),
            ))
            # Cuanto mas lejos y ancho puede mirar el planner gracias al mapa.
            self.plan_forward_m = float(mem.get("plan_forward_m", 3.0))
            self.plan_side_m = float(mem.get("plan_side_m", 2.0))

        safety = cfg.get("safety", {})
        self.stale_frame_s = float(safety.get("stale_frame_s", 2.0))
        self.max_consecutive_errors = int(safety.get("max_consecutive_errors", 5))
        self.recovery_after_empty = int(safety.get("recovery_after_empty_plans", 3))
        self.recovery_turn_s = float(safety.get("recovery_turn_s", 1.5))
        self.loop_period_s = float(safety.get("loop_period_s", 0.0))

        # --- recuperacion informada (antes: config declarada y nunca leida) ---
        # frodobot_rover.yaml ya traia use_vlm_recovery / vlm_recovery_* /
        # recovery_headings_deg / recovery_min_cobertura_pct, pero _recover()
        # no los miraba: giraba siempre el mismo tiempo hacia el mismo lado.
        # Ahora _recover() elige el rumbo con la memoria espacial y, si esta
        # habilitado y hay credenciales, lo contrasta con el VLM.
        self.use_vlm_recovery = bool(safety.get("use_vlm_recovery", False))
        self.vlm_recovery_timeout_s = float(safety.get("vlm_recovery_timeout_s", 4.0))
        self.vlm_recovery_min_confidence = float(
            safety.get("vlm_recovery_min_confidence", 0.35))
        # Rumbos candidatos, en grados relativos al rumbo actual. 0 = adelante
        # (por si el bloqueo ya se disolvio), 180 = atras. Se evaluan contra el
        # PersistentMap; si no hay memoria, queda el barrido de siempre.
        self.recovery_headings_deg = [
            float(v) for v in safety.get("recovery_headings_deg", [0.0, 90.0, -90.0, 180.0])
        ] or [90.0]
        # Cobertura minima (%) que tiene que tener un rumbo en la memoria para
        # que su puntaje se tome en serio. Por debajo, "no se sabe" != "libre".
        self.recovery_min_cobertura_pct = float(
            safety.get("recovery_min_cobertura_pct", 25.0))
        self._vlm_recovery_failed = False   # avisar una sola vez si no se puede usar
        self._vlm_recovery_errors = 0       # fallas en tiempo de ejecucion (timeout/red)

        self.stats = LoopStats()
        self._stop_requested = False
        self._checkpoints: list[Checkpoint] = []
        self._latest_scanned = 0
        self._last_frame_ts = 0.0
        self._last_frame_change = time.time()
        self._consecutive_errors = 0
        self._consecutive_empty = 0
        # Deteccion de giro sin avance: el planner puede quedar en un ciclo
        # donde cada giro revela una escena que vuelve a pedir girar. Sin
        # memoria entre frames, eso no se rompe solo.
        self._consecutive_turns = 0
        self._turn_sign_history: list[float] = []
        self.max_consecutive_turns = int(safety.get("max_consecutive_turns", 6))
        self.unstick_forward_s = float(safety.get("unstick_forward_s", 1.2))

        # Histeresis de lado de esquive. Cada frame se planifica de cero, asi
        # que nada impide cambiar de "paso por derecha" a "paso por izquierda"
        # a mitad de maniobra. Ese titubeo consume el margen que hacia falta
        # para cualquiera de los dos lados. Aca recordamos el lado elegido y
        # exigimos una diferencia grande para cambiarlo.
        self._commit_side = 0          # -1 izquierda, +1 derecha, 0 sin compromiso
        self._commit_until = 0.0       # timestamp hasta el que vale el compromiso
        self.commit_hold_s = float(nav.get("commit_hold_s", 2.0))
        self.commit_min_deg = float(nav.get("commit_min_deg", 8.0))
        self.commit_override_deg = float(nav.get("commit_override_deg", 30.0))

        # Consola en tabla: una fila prolija por iteracion en vez de las 2+
        # lineas sueltas que imprimia antes cada paso del loop, mas los
        # cambios de estado / checkpoints / eventos raros en su propia linea
        # bien marcada (ver genie_rover/console_report.py). mission.console_color
        # en el config permite forzar sin color (por ejemplo grabando la
        # salida con script/tee) incluso en una terminal interactiva.
        mission_cfg = cfg.get("mission", {}) or {}
        self._reporter = MissionConsoleReporter(
            target_label="checkpoint",
            enable_color=mission_cfg.get("console_color"))

    # ------------------------------------------------------------------ ciclo

    def request_stop(self, *_a) -> None:
        print("\n[bridge] parada solicitada, frenando ...")
        self._stop_requested = True

    def send(self, cmd: DriveCommand, quiet: bool = False) -> None:
        """Envia el comando (o lo simula, en dry-run).

        quiet=True suprime la linea "[ENVIADO]/[DRY-RUN] ..." de siempre --
        MissionConsoleReporter ya muestra la accion como parte de la fila de
        tabla (ver _step() mas abajo), asi que imprimirla de nuevo aca
        duplicaba la misma info en dos lineas por iteracion. Los eventos
        raros (_recover/_unstick, freno por error) siguen llamando con el
        default quiet=False, para que queden bien visibles fuera de la tabla.
        """
        if not quiet:
            tag = "DRY-RUN" if self.dry_run else "ENVIADO"
            print(f"  [{tag}] linear={cmd.linear:+.2f} angular={cmd.angular:+.2f}  {cmd.reason}")
        if not self.dry_run:
            self.client.control(cmd.linear, cmd.angular)

    def refresh_checkpoints(self) -> None:
        try:
            self._checkpoints, self._latest_scanned = self.client.checkpoints()
        except Exception as exc:
            print(f"[bridge] no pude leer los checkpoints: {exc}")

    def current_target(self) -> Checkpoint | None:
        for cp in self._checkpoints:
            if cp.sequence > self._latest_scanned:
                return cp
        return None

    def run(self, max_seconds: float | None = None) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        # IndoorBridge (uses_gps_checkpoints = False) no usa checkpoints GPS
        # en absoluto -- la mision de cono los ignora por completo, asi que
        # leerlos y anunciarlos aca era puro ruido en el log.
        if self.uses_gps_checkpoints:
            self.refresh_checkpoints()
            target = self.current_target()
            if target is None:
                print("[bridge] No hay checkpoint pendiente. Voy a navegar solo evitando "
                      "obstaculos, con la meta fija derecho adelante.")
            else:
                print(f"[bridge] Objetivo: checkpoint #{target.sequence} "
                      f"({target.latitude}, {target.longitude})")

        t_start = time.time()
        try:
            while not self._stop_requested:
                if max_seconds is not None and (time.time() - t_start) > max_seconds:
                    print(f"[bridge] limite de {max_seconds:.0f} s alcanzado")
                    break
                t_iter = time.time()
                try:
                    self._step()
                    self._consecutive_errors = 0
                except Exception as exc:
                    self._consecutive_errors += 1
                    self.stats.errors += 1
                    print(f"[bridge] error en la iteracion ({self._consecutive_errors}"
                          f"/{self.max_consecutive_errors}): {exc}")
                    try:
                        self.send(DriveCommand(0.0, 0.0, "freno por error"))
                    except Exception as exc_freno:
                        # Si el SDK esta caido, self.send() puede levantar tambien.
                        # No dejamos que eso escape y mate el proceso en el primer
                        # error: seguimos con el reintento normal de mas abajo.
                        print(f"[bridge] tampoco pude frenar: {exc_freno}")
                    if self._consecutive_errors >= self.max_consecutive_errors:
                        print("[bridge] demasiados errores seguidos, abandono")
                        break
                    time.sleep(1.0)

                self.stats.iterations += 1
                if self.loop_period_s > 0:
                    sleep = self.loop_period_s - (time.time() - t_iter)
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            # Este freno es lo mas importante del archivo: el SDK mantiene el
            # ultimo comando indefinidamente, asi que si salimos sin frenar el
            # rover se sigue moviendo solo.
            print("[bridge] frenando el rover")
            if not self.dry_run:
                self.client.stop()
                self.client.stop()  # dos veces, por si se pierde un mensaje RTM
            self._print_summary()

    # -------------------------------------------------------------- un paso

    def _step(self) -> None:
        rgb, frame_ts = self.client.front_frame()
        now = time.time()
        if frame_ts != self._last_frame_ts:
            self._last_frame_ts = frame_ts
            self._last_frame_change = now
        elif (now - self._last_frame_change) > self.stale_frame_s:
            raise RoverError(
                f"El frame no cambia desde hace {now - self._last_frame_change:.1f} s "
                "(video congelado)"
            )

        telem = self.client.telemetry()
        heading = self.heading_est.update(telem.latitude, telem.longitude,
                                          telem.orientation, telem.timestamp)

        target = self.current_target()
        if target is not None and heading is not None:
            goal = goal_from_gps(telem.latitude, telem.longitude, heading,
                                 target.latitude, target.longitude, self.goal_range_m)
            if goal.distance_m < self.claim_radius_m:
                ok, msg = self.client.claim_checkpoint()
                if ok:
                    self._reporter.checkpoint(target.sequence, msg)
                    self.refresh_checkpoints()
                else:
                    self._reporter.event(
                        "CERCA DEL CHECKPOINT",
                        f"{goal.distance_m:.1f} m, rechazado: {msg}", "yellow")
            goal_desc = (f"cp#{target.sequence} a {goal.distance_m:.0f} m, "
                         f"rel {goal.relative_bearing_deg:+.0f} grados")
        else:
            goal = type("G", (), {"x_right_m": 0.0, "y_forward_m": self.goal_range_m})()
            goal_desc = "derecho adelante (sin meta GPS)"

        res = self.perception.process(rgb)

        # Integrar en el mapa persistente y planificar sobre el acumulado.
        plan_bev, plan_obs = res.traversability, res.observed
        fwd, side = self.forward_range, self.side_range
        pose = None
        map_cells = None
        if self.use_map and self.odometry is not None and self.pmap is not None:
            pose = self.odometry.update(telem.raw)
            self.pmap.integrate(res.traversability, res.observed, pose,
                                self.forward_range, self.side_range, t=now)
            h, w = res.traversability.shape
            plan_bev, plan_obs = self.pmap.extract_bev(
                pose, self.plan_forward_m, self.plan_side_m, h, w)
            fwd, side = self.plan_forward_m, self.plan_side_m
            map_cells = self.pmap.stats()["celdas_vistas"]

        # Estado de la mision derivado del contexto de esta iteracion (Bridge
        # no tiene una FSM formal como ConeMissionFSM, pero igual conviene
        # avisar en su propia linea cuando cambia de fase) -- se afina mas
        # abajo segun el resultado del chequeo de obstaculo / del planner.
        bridge_state = "SIN_META" if target is None else (
            "CERCA_CHECKPOINT" if goal.distance_m < self.claim_radius_m * 1.5 else "NAVEGANDO")
        self._reporter.state_change(bridge_state, goal_desc)

        def _row(action: str, state: str = bridge_state) -> None:
            self._reporter.row(iteration=self.stats.iterations, state=state,
                               pose=pose, target_desc=goal_desc, map_cells=map_cells,
                               trav=res.traversability, action=action)

        # El chequeo de colision usa SIEMPRE la observacion fresca: si algo se
        # cruzo recien, no queremos que el promedio del mapa lo diluya.
        if front_is_blocked(res.traversability, self.resolution):
            self.stats.blocked += 1
            self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"), quiet=True)
            _row("frenado: obstaculo al frente", state="OBSTACULO")
            return

        plan = plan_on_bev(
            bev_traversability=plan_bev,
            observed_mask=plan_obs,
            goal_x_m=float(goal.x_right_m),
            goal_y_m=float(goal.y_forward_m),
            bev_resolution_m=(2.0 * side) / plan_bev.shape[1],
            config=self.planner_cfg,
        )

        path = plan.final_path_xy_m
        if path is None or len(path) < 2:
            self.stats.plans_empty += 1
            self._consecutive_empty += 1
            if self._consecutive_empty >= self.recovery_after_empty:
                _row(f"RECUPERACION ({self._consecutive_empty} planes vacios seguidos)",
                     state="RECUPERANDO")
                self._recover(rgb)
            else:
                self.send(DriveCommand(0.0, 0.0, "el planner no encontro camino"), quiet=True)
                _row(f"sin camino ({self._consecutive_empty}/{self.recovery_after_empty})",
                     state="SIN_CAMINO")
            return

        self._consecutive_empty = 0
        self.stats.plans_ok += 1

        cmd = self.follower.command(path)
        cmd = self._apply_commit(cmd, path)

        # Un comando con linear=0 y angular!=0 es "girar en el lugar". Si eso
        # se repite, el robot esta atrapado: cada giro le muestra una escena
        # que vuelve a pedir girar, y sin memoria no sale solo.
        if cmd.linear == 0.0 and cmd.angular != 0.0:
            self._consecutive_turns += 1
            self._turn_sign_history.append(1.0 if cmd.angular > 0 else -1.0)
            if self._consecutive_turns >= self.max_consecutive_turns:
                _row(f"DESATASCO ({self._consecutive_turns} giros seguidos)",
                     state="DESATASCO")
                self._unstick()
                return
        else:
            self._consecutive_turns = 0
            self._turn_sign_history.clear()

        self.send(cmd, quiet=True)
        _row(f"{cmd.linear:+.2f}m/s {cmd.angular:+.2f}rad/s  {cmd.reason}")
        self._maybe_dump_debug(rgb, res, plan)

    # --------------------------------------------------------- odometria ciega

    def _track_pose_during_maneuver(self) -> None:
        """Pide telemetria y actualiza odometria durante una maniobra ciega
        (_recover / _unstick).

        BUG que esto arregla: _recover()/_unstick() mueven al robot con
        send() + time.sleep(), sin llamar a client.telemetry()/odometry.update()
        en el medio. Odometry.update() tiene un guard de max_dt_s (0.5s por
        defecto): si el hueco entre dos muestras integradas supera eso,
        DESCARTA el intervalo entero en vez de integrarlo (para no inventar
        movimiento a partir de un salto grande de reloj). Como estas maniobras
        duran 1.2-2.5s tipicamente, el hueco caia siempre en esa rama de
        descarte: el robot se movia de verdad pero self.pose (y el
        PersistentMap, que ancla todo en ese marco) quedaban exactamente
        igual que antes de la maniobra -- el mapa y la pose se desalineaban
        del mundo real justo en el momento en que mas importa que no lo hagan
        (recuperandose de estar atascado).

        Protegido con use_map/odometry is not None y try/except: un fallo
        puntual de telemetry() no debe tumbar la maniobra de recuperacion.
        """
        if not self.use_map or self.odometry is None:
            return
        try:
            telem = self.client.telemetry()
            self.odometry.update(telem.raw)
        except Exception as exc:
            print(f"[bridge] no pude actualizar odometria durante la maniobra: {exc}")

    def _unstick(self) -> None:
        """Rompe el ciclo de giros sin avance.

        Avanza en linea recta un momento, ignorando el planner. Es seguro
        porque front_is_blocked ya se evaluo en esta misma iteracion y dio
        libre: si hubiera algo delante, no habriamos llegado hasta aca.
        """
        self.stats.unstucks += 1
        giros = self._consecutive_turns
        sentido = sum(self._turn_sign_history)
        self._reporter.event(
            "ATASCADO",
            f"{giros} giros seguidos sin avanzar (sentido dominante "
            f"{'izq' if sentido > 0 else 'der'}). Fuerzo un avance de "
            f"{self.unstick_forward_s:.1f} s.", "magenta")

        self.send(DriveCommand(self.follower.max_linear, 0.0, "avance forzado"))
        t0 = time.time()
        while time.time() - t0 < self.unstick_forward_s and not self._stop_requested:
            time.sleep(0.15)
            self._track_pose_during_maneuver()
            try:
                rgb, _ = self.client.front_frame()
                res = self.perception.process(rgb)
                if front_is_blocked(res.traversability, self.resolution):
                    self._reporter.event("DESATASCO", "obstaculo durante el avance forzado, corto")
                    break
            except Exception:
                break

        self.send(DriveCommand(0.0, 0.0, "fin del avance forzado"))
        # Una muestra mas despues del freno, para capturar el movimiento
        # hasta que el robot realmente se detiene (no solo hasta el ultimo
        # comando enviado).
        self._track_pose_during_maneuver()
        self._consecutive_turns = 0
        self._turn_sign_history.clear()
        self.heading_est.reset_track()

    def _apply_commit(self, cmd: DriveCommand, path: np.ndarray) -> DriveCommand:
        """Evita cambiar de lado de esquive a mitad de maniobra.

        Un desvio "cuenta" solo si supera commit_min_deg: por debajo de eso el
        camino va practicamente derecho y no hay lado que recordar. Una vez
        comprometido, solo se cambia si el nuevo lado supera
        commit_override_deg, o si pasaron commit_hold_s sin desvios.
        """
        target = self.follower.lookahead_point(path)
        if target is None:
            return cmd

        error_deg = math.degrees(math.atan2(float(target[0]), float(target[1])))
        now = time.time()

        if abs(error_deg) < self.commit_min_deg:
            # Camino casi recto: no hay maniobra en curso.
            if now > self._commit_until:
                self._commit_side = 0
            return cmd

        side = 1 if error_deg > 0 else -1

        if self._commit_side == 0 or now > self._commit_until:
            self._commit_side = side
            self._commit_until = now + self.commit_hold_s
            return cmd

        if side == self._commit_side:
            self._commit_until = now + self.commit_hold_s
            return cmd

        # Quiere cambiar de lado con un compromiso vigente.
        if abs(error_deg) >= self.commit_override_deg:
            lado = "derecha" if side > 0 else "izquierda"
            print(f"[bridge] cambio de lado justificado ({error_deg:+.0f} grados "
                  f"hacia {lado})")
            self._commit_side = side
            self._commit_until = now + self.commit_hold_s
            return cmd

        # Cambio no justificado: seguimos derecho en vez de titubear. Es mas
        # seguro que insistir con el lado viejo, porque el planner ya no lo
        # considera viable.
        return DriveCommand(cmd.linear, 0.0,
                            f"mantengo el rumbo (evito titubeo, {error_deg:+.0f} grados)")

    # ------------------------------------------------- recuperacion informada

    def _score_heading_from_memory(self, heading_rad: float) -> tuple[float, float]:
        """(fraccion_libre, cobertura) de la ventana de memoria en ese rumbo.

        Mira el PersistentMap como si el robot ya estuviera girado hacia
        `heading_rad` y pregunta cuanto de lo que hay adelante esta visto y es
        transitable. Devuelve (0.0, 0.0) si no hay memoria disponible.
        """
        if self.pmap is None or self.odometry is None:
            return 0.0, 0.0
        here = self.odometry.pose
        probe = Pose(here.x, here.y, heading_rad)
        try:
            bev, observed = self.pmap.extract_bev(
                probe,
                forward_range_m=self.plan_forward_m,
                side_range_m=self.plan_side_m,
                out_h=64, out_w=64,
            )
        except Exception:
            return 0.0, 0.0
        vistas = observed.astype(bool)
        cobertura = float(vistas.mean())
        if not vistas.any():
            return 0.0, 0.0
        libre = float((bev[vistas] > 0.5).mean())
        return libre, cobertura

    def _pick_recovery_heading(self, rgb=None) -> tuple[float | None, str]:
        """Elige cuantos grados girar (relativos al rumbo actual) y por que.

        Orden de preferencia:
          1. memoria espacial (PersistentMap): el rumbo con mas terreno visto
             y transitable, entre self.recovery_headings_deg;
          2. el VLM (si use_vlm_recovery y hay un frame), que puede pisar a la
             memoria cuando la memoria no vio casi nada;
          3. barrido ciego de siempre (el lado que ya usaba PathFollower).

        Devuelve (None, motivo) para pedir el barrido ciego.

        Sesgo deliberado contra "adelante" (~0 grados): a _recover() se llega
        JUSTAMENTE porque el planner no encontro camino varias veces seguidas
        con el rumbo actual, y ese planner ya planifica sobre el mapa. Asi que
        "la memoria dice que adelante esta libre" no es informacion nueva:
        elegirlo seria no hacer nada y volver a entrar en recuperacion en la
        proxima iteracion. Solo gana si es CLARAMENTE mejor que la mejor
        alternativa, y aun asi se resuelve con un barrido en vez de con un
        no-op.
        """
        MARGEN_ADELANTE = 0.15   # cuanto mejor tiene que ser "adelante" para ganar

        mejor_giro, mejor_giro_libre, mejor_giro_cob = None, -1.0, 0.0
        mejor_recto_libre = -1.0
        theta = self.odometry.pose.theta if self.odometry is not None else 0.0

        for deg in self.recovery_headings_deg:
            libre, cob = self._score_heading_from_memory(theta + math.radians(deg))
            if cob * 100.0 < self.recovery_min_cobertura_pct:
                continue    # "no se sabe" no es "libre"
            if abs(deg) < 15.0:
                mejor_recto_libre = max(mejor_recto_libre, libre)
                continue
            if libre > mejor_giro_libre:
                mejor_giro, mejor_giro_libre, mejor_giro_cob = deg, libre, cob

        memoria_util = mejor_giro is not None and mejor_giro_libre > 0.5

        if memoria_util and mejor_recto_libre > mejor_giro_libre + MARGEN_ADELANTE:
            # Raro: el mapa insiste en que adelante esta mucho mejor que
            # cualquier giro. No nos quedamos quietos (eso vuelve a fallar):
            # barrido ciego corto y que el planner reintente con datos frescos.
            # (No consultamos al VLM aca: la memoria ya decidio, y esto evita
            # pagar la latencia del VLM cuando su respuesta no se va a usar.)
            return None, (f"la memoria prefiere seguir derecho "
                          f"({mejor_recto_libre * 100:.0f}% libre), hago un barrido corto")

        if memoria_util:
            # Idem: la memoria ya alcanza, no hace falta esperar al VLM.
            return (mejor_giro,
                    f"memoria: {mejor_giro:+.0f} grados, {mejor_giro_libre * 100:.0f}% libre "
                    f"({mejor_giro_cob * 100:.0f}% visto)")

        # La memoria no dio una respuesta util: reciem aca vale la pena pagar
        # la latencia del VLM (hasta vlm_recovery_timeout_s, default 4s).
        vlm_deg = None
        if self.use_vlm_recovery and rgb is not None and not self._vlm_recovery_failed:
            vlm_deg = self._ask_vlm_heading(rgb)

        if vlm_deg is not None and abs(vlm_deg) >= 15.0:
            return vlm_deg, f"VLM sugiere {vlm_deg:+.0f} grados (memoria sin cobertura util)"
        if vlm_deg is not None:
            return None, "el VLM dice que adelante esta bien; barrido corto igual"

        return None, "barrido ciego (sin memoria ni VLM utiles)"

    def _ask_vlm_heading(self, rgb) -> float | None:
        """Grados relativos sugeridos por el VLM, o None. Nunca levanta."""
        try:
            from .vlm_recovery import ask_recovery_heading
        except Exception as exc:
            self._vlm_recovery_failed = True
            self._reporter.event(
                "RECUPERACION",
                f"use_vlm_recovery esta en true pero no pude importar vlm_recovery "
                f"({exc}); sigo sin VLM el resto de la corrida", "yellow")
            return None
        try:
            decision = ask_recovery_heading(
                rgb,
                min_confidence=self.vlm_recovery_min_confidence,
                timeout_s=self.vlm_recovery_timeout_s,
            )
        except Exception as exc:   # ask_recovery_heading ya no deberia levantar
            self._vlm_recovery_errors += 1
            self._reporter.event("RECUPERACION", f"VLM fallo ({exc}), sigo sin el", "yellow")
            if self._vlm_recovery_errors >= 3:
                # 3 fallas seguidas en tiempo de ejecucion (timeout/red): dejamos
                # de pagar el timeout completo (hasta vlm_recovery_timeout_s) en
                # cada recuperacion el resto de la corrida, igual que cuando
                # falla el import mas arriba.
                self._vlm_recovery_failed = True
                self._reporter.event(
                    "RECUPERACION",
                    f"VLM fallo {self._vlm_recovery_errors} veces seguidas; "
                    f"sigo sin VLM el resto de la corrida", "yellow")
            return None
        self._vlm_recovery_errors = 0
        if decision is None:
            return None
        grados = {
            "adelante": 0.0, "forward": 0.0,
            "izquierda": 90.0, "left": 90.0,
            "derecha": -90.0, "right": -90.0,
            "atras": 180.0, "back": 180.0, "backward": 180.0,
        }.get(str(decision.heading).lower())
        if grados is None:
            return None
        self._reporter.event(
            "RECUPERACION",
            f"VLM: {decision.heading} (conf {decision.confidence:.2f}) — {decision.reason}",
            "magenta")
        return grados

    def _turn_towards(self, delta_deg: float) -> None:
        """Gira en el lugar `delta_deg` grados usando la odometria como
        realimentacion, con el tiempo de barrido como tope duro.

        Sin odometria cae al giro por tiempo de siempre.
        """
        ang = float(np.clip(self.follower.angular_sign * self.follower.turn_speed,
                            -self.follower.max_angular, self.follower.max_angular))
        if delta_deg < 0:
            ang = -ang

        objetivo = abs(math.radians(delta_deg))
        theta0 = self.odometry.pose.theta if self.odometry else None
        # Tope de tiempo generoso pero acotado: un giro de 180 grados no puede
        # durar lo mismo que uno de 90, pero tampoco puede durar para siempre
        # si la odometria no avanza (rueda patinando, telemetria caida).
        tope_s = self.recovery_turn_s * max(1.0, abs(delta_deg) / 90.0) * 2.0

        self.send(DriveCommand(0.0, ang, "barrido de recuperacion"))
        t0 = time.time()
        while time.time() - t0 < tope_s and not self._stop_requested:
            time.sleep(0.15)
            self._track_pose_during_maneuver()
            if theta0 is not None and self.odometry is not None:
                girado = abs(math.atan2(math.sin(self.odometry.pose.theta - theta0),
                                        math.cos(self.odometry.pose.theta - theta0)))
                if girado >= objetivo:
                    break
        self.send(DriveCommand(0.0, 0.0, "fin del barrido"))
        self._track_pose_during_maneuver()

    def _recover(self, rgb=None) -> None:
        """Recuperacion: girar en el lugar hacia el rumbo con mejor pinta.

        Antes giraba a ciegas siempre el mismo tiempo y hacia el mismo lado,
        ignorando `use_vlm_recovery` / `recovery_headings_deg` / 
        `recovery_min_cobertura_pct` — que ya estaban en el config. Ahora:
        consulta la memoria espacial, opcionalmente el VLM, y gira lo que haga
        falta con realimentacion de odometria. Si no hay ni memoria ni VLM el
        comportamiento es exactamente el de antes.
        """
        # Freno antes de deliberar: _pick_recovery_heading() puede bloquear
        # hasta vlm_recovery_timeout_s (default 4s) esperando al VLM, y antes
        # el rover seguia con el ultimo comando de movimiento durante ese
        # tiempo entero. Frenamos primero y seguimos alimentando la odometria
        # mientras se decide el rumbo.
        self.send(DriveCommand(0.0, 0.0, "freno para decidir la recuperacion"), quiet=True)
        self._track_pose_during_maneuver()

        delta_deg, motivo = self._pick_recovery_heading(rgb)
        self._reporter.event("RECUPERACION", f"girando para buscar salida — {motivo}", "magenta")

        if delta_deg is None:
            self._recover_blind()
        else:
            self._turn_towards(delta_deg)

        self.heading_est.reset_track()  # el track GPS previo ya no dice el rumbo
        self._consecutive_empty = 0

    def _recover_blind(self) -> None:
        """El barrido de siempre: girar un tiempo fijo hacia el lado del follower."""
        # (el evento ya lo emitio _recover)
        # Clampeado a max_angular: nada obligaba antes a que turn_speed fuera
        # <= max_angular, asi que un turn_speed mal configurado se saltaba el
        # tope de seguridad que si respeta PathFollower.command().
        ang = float(np.clip(self.follower.angular_sign * self.follower.turn_speed,
                            -self.follower.max_angular, self.follower.max_angular))
        self.send(DriveCommand(0.0, ang, "barrido de recuperacion"))

        # Loop de polling en vez de un unico time.sleep(): alimenta odometria
        # durante todo el barrido (ver _track_pose_during_maneuver) y permite
        # que Ctrl-C (_stop_requested) corte la maniobra a mitad de camino en
        # vez de esperar los recovery_turn_s completos.
        t0 = time.time()
        while time.time() - t0 < self.recovery_turn_s and not self._stop_requested:
            time.sleep(0.15)
            self._track_pose_during_maneuver()

        self.send(DriveCommand(0.0, 0.0, "fin del barrido"))
        self._track_pose_during_maneuver()

    def _maybe_dump_debug(self, rgb, res, plan) -> None:
        if not self.debug_dir:
            return
        try:
            from PIL import Image
            n = self.stats.iterations
            Image.fromarray(rgb).save(self.debug_dir / f"{n:05d}_rgb.jpg", quality=80)
            Image.fromarray(plan.visualization).save(self.debug_dir / f"{n:05d}_plan.png")
            np.save(self.debug_dir / f"{n:05d}_bev.npy", res.traversability)
            if self.pmap is not None and self.odometry is not None:
                Image.fromarray(self.pmap.to_image(self.odometry.pose)).save(
                    self.debug_dir / f"{n:05d}_mapa.png")
        except Exception as exc:
            print(f"[bridge] no pude escribir el debug: {exc}")

    def _print_summary(self) -> None:
        s = self.stats
        print("\n--- resumen ---")
        print(f"  iteraciones:            {s.iterations}")
        print(f"  planes exitosos:        {s.plans_ok}")
        print(f"  planes vacios:          {s.plans_empty}")
        print(f"  frenadas por obstaculo: {s.blocked}")
        print(f"  desatascos forzados:    {s.unstucks}")
        if self.pmap is not None and self.odometry is not None:
            st = self.pmap.stats()
            p = self.odometry.pose
            print(f"  --- memoria espacial ---")
            print(f"  celdas del mapa:        {st['celdas_vistas']}")
            print(f"  recentrados:            {st['recentrados']}")
            print(f"  pose final:             ({p.x:+.2f}, {p.y:+.2f}) "
                  f"{math.degrees(p.theta):+.0f} grados")
            print(f"  distancia recorrida:    {self.odometry.distance_travelled:.2f} m")
            print(f"  correcciones GPS:       {self.odometry.gps_corrections}")
        print(f"  errores:                {s.errors}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--go", action="store_true",
                    help="enviar comandos de verdad (sin esto es simulacro)")
    ap.add_argument("--start-mission", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="cortar despues de N segundos (usalo siempre las primeras veces)")
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _check_placeholders(cfg)

    bridge = Bridge(cfg, dry_run=not args.go, debug_dir=args.debug_dir)

    if args.start_mission:
        print("[bridge] iniciando mision ...")
        print(f"   {bridge.client.start_mission()}")

    if args.go:
        print("\n" + "=" * 62)
        print("  MODO REAL: el rover se va a mover. Ctrl-C frena.")
        print("  Tene el robot a la vista y espacio libre alrededor.")
        print("=" * 62)
        for i in (3, 2, 1):
            print(f"  {i} ...")
            time.sleep(1)

    bridge.run(max_seconds=args.max_seconds)
    return 0


def _check_placeholders(cfg: dict) -> None:
    """Impide arrancar con la calibracion de ejemplo todavia puesta."""
    cam = cfg.get("camera", {})
    if cam.get("calibrated") is not True:
        raise SystemExit(
            "El config todavia tiene camera.calibrated: false.\n"
            "Corre tools/calibrate_camera.py, medi la altura y el pitch de la camara, "
            "y recien despues poné calibrated: true.\n"
            "Sin la calibracion correcta la proyeccion a BEV da basura y el robot "
            "va a chocar."
        )


if __name__ == "__main__":
    sys.exit(main())
