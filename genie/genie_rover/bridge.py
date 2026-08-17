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
    front_clearance_m,
    front_is_blocked,
    goal_from_gps,
    path_robot_to_world,
    path_world_to_robot,
)
from .odometry import Odometry, OdometryConfig, Pose
from .perception import PerceptionPipeline
from .persistent_map import MapConfig, PersistentMap
from .recovery import NearRegimeConfig, NearRegimeController, Regime
from .sdk_client import Checkpoint, RoverClient, RoverError


@dataclass
class LoopStats:
    iterations: int = 0
    plans_ok: int = 0
    plans_empty: int = 0
    blocked: int = 0
    errors: int = 0
    unstucks: int = 0
    near_regime_activations: int = 0
    near_regime_interventions: int = 0


class Bridge:
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
            min_linear_while_planned=nav.get("min_linear_while_planned", 0.08),  # <-- agregar
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
                # BUGFIX: el default aca era 1.0, pero OdometryConfig.rotation_sign
                # default es -1.0. Si el YAML no trae rotation_sign, se estaba
                # invirtiendo el signo silenciosamente respecto del default
                # "oficial" del dataclass. Hoy es inofensivo porque
                # use_gyro_for_rotation=True por defecto (se ignora la
                # rotacion por ruedas), pero es una trampa si algun dia se
                # desactiva el gyro.
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

        #FIX
        # ---- replanificacion por disparo espacial ------------------------
        # En vez de replanificar cada frame (5 Hz, ~5 cm de avance = ruido
        # dominando la decision), guardamos el ultimo plan y solo llamamos a
        # plan_on_bev cuando el robot se movio lo suficiente como para que
        # el BEV muestre informacion realmente nueva.
        self._last_plan_pose: Pose | None = None
        self._current_plan_xy: np.ndarray | None = None   # camino vigente, en MUNDO
        self._last_plan_t = 0.0
        self.replan_distance_m = float(nav.get("replan_distance_m", 0.30))
        self.replan_max_s = float(nav.get("replan_max_s", 2.0))
        self.min_linear_while_planned = float(nav.get("min_linear_while_planned", 0.08))
        # --------------------------------------------------------------------------------
        

        safety = cfg.get("safety", {})
        self.stale_frame_s = float(safety.get("stale_frame_s", 2.0))
        self.max_consecutive_errors = int(safety.get("max_consecutive_errors", 5))
        self.recovery_after_empty = int(safety.get("recovery_after_empty_plans", 3))
        self.recovery_turn_s = float(safety.get("recovery_turn_s", 1.5))
        self.loop_period_s = float(safety.get("loop_period_s", 0.0))

        # ---- regimen cercano ---------------------------------------------
        # Por debajo de near.clearance_thresh_m la proyeccion a BEV esta
        # geometricamente degradada (el obstaculo tapa la mayor parte del
        # campo visual, la distorsion es maxima justo donde estaria el hueco
        # libre): GeNIE deja de ser una fuente de informacion valida y hay
        # que resolverlo con el mapa persistente en vez de tocar umbrales del
        # planner. Ver genie_rover/recovery.py para el detalle completo.
        #
        # Requiere memoria espacial activa: sin mapa persistente no hay de
        # donde sacar el rumbo de escape cuando la camara ya no sirve.
        near = cfg.get("near_regime", {})
        self.near_ctrl: NearRegimeController | None = None
        if self.use_map:
            self.near_ctrl = NearRegimeController(
                NearRegimeConfig(
                    clearance_thresh_m=float(near.get("clearance_thresh_m", 0.6)),
                    clearance_check_m=float(near.get("clearance_check_m", 1.2)),
                    stall_iters=int(near.get("stall_iters", 5)),
                    stall_disp_m=float(near.get("stall_disp_m", 0.05)),
                    reverse_distance_m=float(near.get("reverse_distance_m", 0.35)),
                    reverse_speed=float(near.get("reverse_speed", 0.15)),
                    reverse_timeout_s=float(near.get("reverse_timeout_s", 8.0)),
                    # ---- corredor por camara (post-retroceso) --------------
                    corridor_band_width_m=float(near.get("corridor_band_width_m", 0.30)),
                    corridor_check_m=float(near.get("corridor_check_m", 1.2)),
                    turn_step_deg=float(near.get("turn_step_deg", 15.0)),
                    max_local_turns=int(near.get("max_local_turns", 4)),
                    # ---- respaldo por mapa (solo si la camara no alcanza) --
                    heading_search_radius_m=float(near.get("heading_search_radius_m", 2.0)),
                    heading_fan_deg=float(near.get("heading_fan_deg", 20.0)),
                    fallback_heading_candidates_deg=tuple(
                        near.get("fallback_heading_candidates_deg", (-45.0, -30.0, 30.0, 45.0))
                    ),
                    align_tolerance_deg=float(near.get("align_tolerance_deg", 12.0)),
                    rotate_timeout_s=float(near.get("rotate_timeout_s", 6.0)),
                    turn_speed=float(near.get("turn_speed", nav.get("turn_speed", 0.35))),
                    replan_lock_s=float(near.get("replan_lock_s", 2.0)),
                    success_clearance_m=float(near.get("success_clearance_m", 0.6)),
                    corridor_block_s=float(near.get("corridor_block_s", 30.0)),
                    corridor_half_width_m=float(near.get("corridor_half_width_m", 0.35)),
                    fail_block_after=int(near.get("fail_block_after", 2)),
                    fail_intervention_after=int(near.get("fail_intervention_after", 3)),
                ),
                angular_sign=nav["angular_sign"],
            )
        else:
            print("[bridge] AVISO: memory.enabled=false -> el regimen cercano queda "
                  "desactivado (no tiene de donde sacar el rumbo de escape sin mapa "
                  "persistente). El robot va a depender solo de _recover()/_unstick().")

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

    # ------------------------------------------------------------------ ciclo

    def request_stop(self, *_a) -> None:
        print("\n[bridge] parada solicitada, frenando ...")
        self._stop_requested = True

    def send(self, cmd: DriveCommand) -> None:
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
                    self.send(DriveCommand(0.0, 0.0, "freno por error"))
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

        if self.debug_dir:
            print("DEBUG telem.raw keys:", list(telem.raw.keys()),
                  "rpms:", len(telem.raw.get("rpms", [])),
                  "gyros:", len(telem.raw.get("gyros", [])))
        heading = self.heading_est.update(telem.latitude, telem.longitude,
                                          telem.orientation, telem.timestamp)

        target = self.current_target()
        if target is not None and heading is not None:
            goal = goal_from_gps(telem.latitude, telem.longitude, heading,
                                 target.latitude, target.longitude, self.goal_range_m)
            if goal.distance_m < self.claim_radius_m:
                ok, msg = self.client.claim_checkpoint()
                if ok:
                    print(f"[bridge] ✓ checkpoint #{target.sequence} conseguido: {msg}")
                    self.refresh_checkpoints()
                    # BUGFIX: sin esto, el resto de la iteracion (goal_desc,
                    # planificacion, comando) seguia usando target/goal del
                    # checkpoint que ACABAMOS de reclamar en vez del proximo.
                    # Recalculamos con el nuevo target antes de seguir.
                    target = self.current_target()
                    if target is not None:
                        goal = goal_from_gps(telem.latitude, telem.longitude, heading,
                                             target.latitude, target.longitude,
                                             self.goal_range_m)
                    else:
                        goal = type("G", (), {"x_right_m": 0.0,
                                              "y_forward_m": self.goal_range_m})()
                else:
                    print(f"[bridge] cerca del checkpoint ({goal.distance_m:.1f} m) "
                          f"pero rechazado: {msg}")
            if target is not None:
                goal_desc = (f"cp#{target.sequence} a {goal.distance_m:.0f} m, "
                             f"rel {goal.relative_bearing_deg:+.0f} grados")
            else:
                goal_desc = "derecho adelante (sin mas checkpoints)"
        else:
            goal = type("G", (), {"x_right_m": 0.0, "y_forward_m": self.goal_range_m})()
            goal_desc = "derecho adelante (sin meta GPS)"

        res = self.perception.process(rgb)

        # Integrar en el mapa persistente y planificar sobre el acumulado.
        plan_bev, plan_obs = res.traversability, res.observed
        fwd, side = self.forward_range, self.side_range
        nota_mapa = ""
        if self.use_map and self.odometry is not None and self.pmap is not None:
            pose = self.odometry.update(telem.raw)
            self.pmap.integrate(res.traversability, res.observed, pose,
                                self.forward_range, self.side_range, t=now)
            h, w = res.traversability.shape
            plan_bev, plan_obs = self.pmap.extract_bev(
                pose, self.plan_forward_m, self.plan_side_m, h, w)
            fwd, side = self.plan_forward_m, self.plan_side_m
            st = self.pmap.stats()
            nota_mapa = (f"  mapa: {st['celdas_vistas']} celdas  "
                         f"pose=({pose.x:+.2f},{pose.y:+.2f},"
                         f"{math.degrees(pose.theta):+.0f}gr)")

        print(f"[{self.stats.iterations:04d}] rumbo={heading if heading is None else round(heading)} "
              f"({self.heading_est.source})  meta: {goal_desc}  "
              f"celdas BEV={res.stats['bev_observed_cells']:.0f}{nota_mapa}")

        pose_now = self.odometry.pose if (self.use_map and self.odometry is not None) else None

        # ---- regimen cercano: freeze de GeNIE si el frente esta degradado -
        # Por debajo de clearance_thresh_m el BEV instantaneo deja de ser una
        # fuente de informacion valida para planificar (ver recovery.py). Se
        # evalua ANTES del freno duro de front_is_blocked: ese es un freno de
        # un solo frame, y si el robot queda ahi para siempre sin salida es
        # exactamente el congelamiento que este regimen existe para resolver.
        if self.near_ctrl is not None and pose_now is not None:
            clearance = front_clearance_m(res.traversability, self.resolution,
                                          max_check_m=self.near_ctrl.cfg.clearance_check_m)
            if self.near_ctrl.should_enter_near(clearance, now):
                print(f"[bridge] clearance={clearance:.2f} m -> entro a regimen CERCANO")
                self._run_near_regime()
                return

        # El chequeo de colision usa SIEMPRE la observacion fresca: si algo se
        # cruzo recien, no queremos que el promedio del mapa lo diluya. Esto
        # se mantiene como red de seguridad de un solo frame incluso con el
        # regimen cercano activo (p.ej. si use_map=false).
        if front_is_blocked(res.traversability, self.resolution):
            self.stats.blocked += 1
            self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"))
            return

        # ---- disparo espacial: solo replanificar si hace falta ----------
        need_replan = (
            pose_now is None                      # sin odometria, no hay opcion: replanificar
            or self._current_plan_xy is None
            or self._last_plan_pose is None
            or math.hypot(pose_now.x - self._last_plan_pose.x,
                          pose_now.y - self._last_plan_pose.y) >= self.replan_distance_m
            or (time.time() - self._last_plan_t) >= self.replan_max_s
        )

        if not need_replan:
            path = path_world_to_robot(self._current_plan_xy, pose_now)
            cmd = self.follower.command(path)
            cmd = self._apply_commit(cmd, path)
            # BUGFIX: usaba path[0][0] (primer punto del camino, no el punto
            # de lookahead) sobre un denominador fijo en 1.0, que no es la
            # misma formula que PathFollower.command()/_apply_commit() usan
            # para el error de rumbo (atan2(x_right, y_forward) del punto de
            # lookahead). Con eso el chequeo "ya esta alineado" solo daba el
            # resultado correcto por casualidad, cuando y_forward del primer
            # punto era ~1 m.
            target = self.follower.lookahead_point(path)
            if (cmd.linear == 0.0 and cmd.angular != 0.0 and target is not None
                    and abs(math.degrees(math.atan2(float(target[0]), float(target[1]))))
                    <= self.follower.align_threshold):
                cmd = DriveCommand(max(cmd.linear, self.min_linear_while_planned), cmd.angular, cmd.reason)
            self.send(cmd)
            if self.near_ctrl is not None and pose_now is not None:
                self.near_ctrl.note_iteration(pose_now, cmd, time.time())
            return

        bev_resolution_m = (2.0 * side) / plan_bev.shape[1]
        if self.near_ctrl is not None and pose_now is not None:
            # Corredores que quedaron marcados por recuperaciones fallidas
            # recientes: se le esconden al planner de LEJANO para que no
            # vuelva a mandar al robot justo ahi. Es una copia, no toca el
            # mapa persistente en si.
            plan_bev = self.near_ctrl.mask_blocked_corridors(
                plan_bev, pose_now, bev_resolution_m, time.time())

        plan = plan_on_bev(
            bev_traversability=plan_bev,
            observed_mask=plan_obs,
            goal_x_m=float(goal.x_right_m),
            goal_y_m=float(goal.y_forward_m),
            bev_resolution_m=bev_resolution_m,
            config=self.planner_cfg,
        )

        path = plan.final_path_xy_m
        if pose_now is not None and path is not None and len(path) >= 2:
            self._current_plan_xy = path_robot_to_world(path, pose_now)
            # BUGFIX: Odometry.update() muta self.odometry.pose in-place
            # (nunca reasigna un Pose nuevo), asi que pose_now es siempre el
            # MISMO objeto que self.odometry.pose. Guardar la referencia
            # directa hacia que _last_plan_pose "avanzara sola" junto con la
            # pose actual, dejando la distancia recorrida en 0 para siempre
            # y matando el disparo de replanificacion por distancia
            # (replan_distance_m). Hay que guardar una copia congelada.
            self._last_plan_pose = Pose(pose_now.x, pose_now.y, pose_now.theta)
            self._last_plan_t = time.time()
        
        path = plan.final_path_xy_m
        if path is None or len(path) < 2:
            self.stats.plans_empty += 1
            self._consecutive_empty += 1
            if self._consecutive_empty >= self.recovery_after_empty:
                self._recover()
            else:
                self.send(DriveCommand(0.0, 0.0, "el planner no encontro camino"))
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
                self._unstick()
                return
        else:
            self._consecutive_turns = 0
            self._turn_sign_history.clear()

        self.send(cmd)
        if self.near_ctrl is not None and pose_now is not None:
            self.near_ctrl.note_iteration(pose_now, cmd, time.time())
        self._maybe_dump_debug(rgb, res, plan)

    def _run_near_regime(self) -> None:
        """Delega la maniobra completa (retroceder / elegir corredor por
        camara / girar en pasos / escalar) a NearRegimeController.execute(),
        pasandole los callbacks que necesita para hablar con el rover, la
        camara y el mapa.

        No atrapa excepciones de red a proposito: si el SDK falla en medio de
        la maniobra, que suba y la caiga el manejo de errores de run(), que ya
        manda freno.
        """
        assert self.near_ctrl is not None and self.pmap is not None and self.odometry is not None

        def get_pose() -> Pose:
            # BUGFIX: antes devolvia self.odometry.pose sin refrescar
            # telemetria, asi que la pose quedaba congelada durante todo
            # execute() (retroceso y giros). Eso hacia que recorrido/err_deg
            # nunca cambiaran y que ambos loops corrieran siempre hasta su
            # timeout en vez de cortar cuando correspondia. Hay que pedir
            # telemetria fresca y volver a integrar en cada llamada, igual
            # que hace el bucle normal en _step().
            telem = self.client.telemetry()
            return self.odometry.update(telem.raw)

        def capture_bev() -> tuple[np.ndarray, float]:
            """Frame fresco -> BEV + resolucion, para elegir de que lado
            girar DESPUES del retroceso. No usar antes de retroceder: a esa
            distancia el BEV todavia esta degradado (ver docstring de
            recovery.py)."""
            rgb, _ = self.client.front_frame()
            res = self.perception.process(rgb)
            return res.traversability, self.resolution

        def compute_clearance() -> float:
            rgb, _ = self.client.front_frame()
            res = self.perception.process(rgb)
            return front_clearance_m(res.traversability, self.resolution,
                                     max_check_m=self.near_ctrl.cfg.clearance_check_m)

        def request_intervention() -> None:
            print("[bridge] solicitando intervencion humana ...")
            self.client.start_intervention()

        resultado = self.near_ctrl.execute(
            pmap=self.pmap,
            get_pose=get_pose,
            capture_bev=capture_bev,
            compute_clearance=compute_clearance,
            send=self.send,
            request_intervention=request_intervention,
            stopped_flag=lambda: self._stop_requested,
        )
        self.stats.near_regime_activations += 1
        if resultado == "intervencion":
            self.stats.near_regime_interventions += 1
        self.heading_est.reset_track()
        self._consecutive_turns = 0
        self._turn_sign_history.clear()
        self._consecutive_empty = 0
        self._commit_side = 0
        self._current_plan_xy = None
        self._last_plan_pose = None
        print(f"[bridge] regimen cercano -> {resultado}")
    def _unstick(self) -> None:
        """Rompe el ciclo de giros sin avance.

        Avanza en linea recta un momento, ignorando el planner. Es seguro
        porque front_is_blocked ya se evaluo en esta misma iteracion y dio
        libre: si hubiera algo delante, no habriamos llegado hasta aca.
        """
        self.stats.unstucks += 1
        giros = self._consecutive_turns
        sentido = sum(self._turn_sign_history)
        print(f"[bridge] ATASCADO: {giros} giros seguidos sin avanzar "
              f"(sentido dominante {'izq' if sentido > 0 else 'der'}). "
              f"Fuerzo un avance de {self.unstick_forward_s:.1f} s.")

        # BUGFIX: antes se confiaba ciegamente en unstick_forward_s de tiempo
        # de motor sin verificar que el robot realmente se desplazara. Una
        # rueda patinando (o el robot trabado contra algo fuera del cono que
        # chequea front_is_blocked) mandaba "avance maximo" el tiempo entero
        # sin que nada lo notara. Si hay odometria, se verifica desplazamiento
        # real y se corta si no avanza tras darle un margen de arranque.
        start_pose = self.odometry.pose if (self.use_map and self.odometry is not None) else None
        if start_pose is not None:
            start_pose = Pose(start_pose.x, start_pose.y, start_pose.theta)
        stall_grace_s = min(0.6, self.unstick_forward_s / 2)

        self.send(DriveCommand(self.follower.max_linear, 0.0, "avance forzado"))
        t0 = time.time()
        while time.time() - t0 < self.unstick_forward_s and not self._stop_requested:
            time.sleep(0.15)
            try:
                rgb, _ = self.client.front_frame()
                res = self.perception.process(rgb)
                if front_is_blocked(res.traversability, self.resolution):
                    print("[bridge] obstaculo durante el avance forzado, corto")
                    break
                if start_pose is not None and (time.time() - t0) > stall_grace_s:
                    telem = self.client.telemetry()
                    pose = self.odometry.update(telem.raw)
                    avance = math.hypot(pose.x - start_pose.x, pose.y - start_pose.y)
                    if avance < 0.03:
                        print(f"[bridge] avance forzado sin desplazamiento real "
                              f"({avance*100:.1f} cm) -- posible rueda patinando "
                              "o robot trabado, corto")
                        break
            except Exception:
                break

        self.send(DriveCommand(0.0, 0.0, "fin del avance forzado"))
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

    def _recover(self) -> None:
        """Recuperacion minima: girar en el lugar para buscar salida.

        GeNIE usa un VLM para esto (mirar en 4 direcciones y elegir). Esta es la
        version sin VLM: gira a ciegas. Si te importa el puntaje del ERC, aca es
        donde conviene meter el modulo del paper.
        """
        print("[bridge] RECUPERACION: girando para buscar terreno transitable")
        self.send(DriveCommand(0.0, self.follower.angular_sign * self.follower.turn_speed,
                               "barrido de recuperacion"))
        time.sleep(self.recovery_turn_s)
        self.send(DriveCommand(0.0, 0.0, "fin del barrido"))
        self.heading_est.reset_track()  # el track GPS previo ya no dice el rumbo
        self._consecutive_empty = 0

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
        print(f"  activaciones regimen cercano: {s.near_regime_activations}")
        print(f"  intervenciones pedidas:       {s.near_regime_interventions}")
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