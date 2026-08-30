"""Bucle de control: Earth Rover SDK <-> SAM-TP <-> planner BEV de GeNIE.

Por defecto arranca en DRY-RUN: calcula todo e imprime los comandos sin
enviarlos. Para que el robot se mueva de verdad hace falta pasar --go.

    # simulacro, el robot no se mueve
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml

    # de verdad
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml --go \
        --start-mission --max-seconds 120 --debug-dir debug/run1

MERGE 30/08/2026: este archivo incorpora las mejoras del compañero
(replanificación por disparo espacial, banco de caminos cacheado, régimen
cercano con retroceso + recuperación informada por mapa/VLM) preservando lo
propio: la tabla de consola (MissionConsoleReporter), `uses_gps_checkpoints`
para que IndoorBridge siga funcionando sin tocarlo, y odometría alimentada
durante TODAS las maniobras ciegas (ver `_track_pose_during_maneuver`, el
fix del 24-25/08 para que _recover/_unstick no descarrilen la pose y el
PersistentMap). Ver claude/plan-integracion-mejoras-bridge-companero-indoor-30-08.md
en el proyecto para el detalle de qué se tocó y qué queda pendiente de
reconciliar a mano en indoor_bridge.py.
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

from genie_path_planner.path_sampling import (
    sample_paths_polynomial,
    sample_paths_uniform_fan,
)
from genie_path_planner.planner import PlannerConfig, _resize_pixel, plan_on_bev

from .navigation import (
    DriveCommand,
    HeadingEstimator,
    PathFollower,
    front_clearance_m,
    front_is_blocked,
    goal_from_gps,
    path_to_robot,
    path_to_world,
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
    near_regime_activations: int = 0
    retrocesos: int = 0
    recoveries_por_mapa: int = 0
    recoveries_por_vlm: int = 0
    recoveries_ciegas: int = 0


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
            min_linear_while_following=nav.get("min_linear_while_planned", 0.08),
        )
        self.goal_range_m = float(nav.get("goal_range_m", 3.5))
        self.claim_radius_m = float(nav.get("claim_radius_m", 8.0))

        # ---- replanificacion por disparo espacial -------------------------
        # plan_on_bev (GeNIE) no corre en cada frame: solo cuando el robot
        # avanzo replan_every_m desde el ultimo plan, o al camino cacheado le
        # queda menos de replan_min_remaining_m por delante, o pasaron
        # replan_max_s (red de seguridad temporal). Entre medio, el bridge
        # sigue el ultimo camino calculado, reproyectado a la pose actual con
        # path_to_robot(). Ver comentario en frodobot_rover.yaml.
        self.replan_every_m = float(nav.get("replan_every_m", 1.0))
        self.replan_min_remaining_m = float(nav.get("replan_min_remaining_m", 0.4))
        self.replan_max_s = float(nav.get("replan_max_s", 2.0))
        self._plan_path_world: np.ndarray | None = None
        self._plan_pose: Pose | None = None
        self._plan_t = 0.0

        self.planner_cfg = PlannerConfig(**cfg.get("planner", {}))
        # Banco de caminos candidatos, calculado una sola vez (ver _path_bank).
        self._bank: list[np.ndarray] | None = None
        self._bank_shape: tuple[int, int] | None = None
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

        # ---- regimen cercano ------------------------------------------------
        # Por debajo de ~0.6 m el BEV instantaneo deja de ser una fuente de
        # informacion valida (la proyeccion esta geometricamente degradada,
        # ver guia tecnica seccion 4): en vez de frenar para siempre, se usa
        # el mapa persistente -- que todavia tiene la observacion buena de
        # cuando el obstaculo estaba mas lejos -- para decidir si conviene
        # retroceder y hacia donde girar despues. Requiere memory.enabled.
        self.obstacle_persist_frames = int(safety.get("obstacle_persist_frames", 4))
        self.retroceso_min_libre_pct = float(safety.get("retroceso_min_libre_pct", 55.0))
        self.retroceso_min_cobertura_pct = float(safety.get("retroceso_min_cobertura_pct", 30.0))
        self.retroceso_max_m = float(safety.get("retroceso_max_m", 0.6))
        self.retroceso_paso_m = float(safety.get("retroceso_paso_m", 0.2))
        self.retroceso_linear = float(safety.get("retroceso_linear", -0.18))
        # Giro de recuperacion. Separado de follower.turn_speed a proposito:
        # aca conviene girar rapido (el robot esta parado esperando) mientras
        # que en el seguimiento de camino turn_speed es una velocidad de
        # maniobra. recovery_deg_per_s es la tasa REAL de giro del robot y es
        # lo que fija cuanto dura cada paso; medila cronometrando un giro de
        # 180 grados y ajustala si el robot se pasa o se queda corto.
        self.recovery_turn_speed = float(safety.get("recovery_turn_speed", 0.45))
        self.recovery_step_deg = float(safety.get("recovery_step_deg", 45.0))
        self.recovery_deg_per_s = float(safety.get("recovery_deg_per_s", 45.0))
        self.recovery_headings_deg = list(safety.get("recovery_headings_deg", [0.0, 90.0, -90.0, 180.0]))
        self.recovery_min_cobertura_pct = float(safety.get("recovery_min_cobertura_pct", 25.0))
        self.heading_search_radius_m = float(safety.get("heading_search_radius_m", 2.0))
        # Recuperacion con VLM (genie_rover.vlm_recovery): se prueba solo si
        # el mapa no encontro un rumbo confiable. Nunca lanza excepcion -- si
        # falla (sin credenciales, sin red, timeout) el bridge cae al barrido
        # ciego de siempre.
        self.use_vlm_recovery = bool(safety.get("use_vlm_recovery", False))
        self.vlm_recovery_timeout_s = float(safety.get("vlm_recovery_timeout_s", 4.0))
        self.vlm_recovery_min_confidence = float(safety.get("vlm_recovery_min_confidence", 0.35))
        self._consecutive_blocked = 0

        self.stats = LoopStats()
        # Muestras de (rumbo_gps - rumbo_brujula) para diagnosticar si el
        # magnetometro tiene un offset sistematico (corregible con
        # navigation.orientation_offset_deg) o si directamente es ruido.
        self._disagreements: list[float] = []
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

        # Histeresis de lado de esquive. Aunque ahora se replanifica por
        # disparo espacial (no cada frame), cada replanificacion sigue
        # partiendo de cero, asi que nada impide cambiar de "paso por
        # derecha" a "paso por izquierda" de una replanificacion a la
        # siguiente. Ese titubeo consume el margen que hacia falta para
        # cualquiera de los dos lados. Aca recordamos el lado elegido y
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
        tabla (ver _step()/_send_path_command() mas abajo), asi que
        imprimirla de nuevo aca duplicaba la misma info en dos lineas por
        iteracion. Los eventos raros (recuperacion, retroceso, desatasco,
        freno por error) siguen llamando con el default quiet=False, para
        que queden bien visibles fuera de la tabla -- mismo criterio que ya
        se uso en indoor_bridge.py.
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
                    except Exception as exc2:
                        print(f"[bridge] tambien fallo el freno de emergencia: {exc2}")
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

        # Integrar en el mapa persistente. Esto corre SIEMPRE, a la
        # frecuencia del frame: es lo que permite desacoplar la percepcion
        # (esto) de la decision de replanificar (mas abajo, disparo
        # espacial) y del regimen cercano.
        #
        # NOTA para IndoorBridge (RTAB-Map): la correccion de pose por
        # RTAB-Map se inyecta ANTES de la linea `self.odometry.update(...)`
        # de aca abajo, pisando self.odometry.pose con la pose corregida de
        # la relocalizacion. Esa linea sigue en la misma posicion relativa
        # que en la version anterior de este archivo (justo despues de
        # perception.process, dentro del bloque `if self.use_map...`), asi
        # que el override de _step() en indoor_bridge.py debe seguir
        # inyectando la correccion inmediatamente antes de este llamado.
        plan_bev, plan_obs = res.traversability, res.observed
        fwd, side = self.forward_range, self.side_range
        pose_now: Pose | None = None
        map_cells: int | None = None
        if self.use_map and self.odometry is not None and self.pmap is not None:
            pose_now = self.odometry.update(telem.raw)
            self.pmap.integrate(res.traversability, res.observed, pose_now,
                                self.forward_range, self.side_range, t=now)
            h, w = res.traversability.shape
            plan_bev, plan_obs = self.pmap.extract_bev(
                pose_now, self.plan_forward_m, self.plan_side_m, h, w)
            fwd, side = self.plan_forward_m, self.plan_side_m
            map_cells = self.pmap.stats()["celdas_vistas"]

        # Desacuerdo brujula vs GPS, para el diagnostico final (_print_summary).
        desacuerdo = self.heading_est.disagreement_deg()
        if desacuerdo is not None and self.heading_est.source == "gps_track":
            self._disagreements.append(float(desacuerdo))

        # Estado de la mision derivado del contexto de esta iteracion (Bridge
        # no tiene una FSM formal como ConeMissionFSM, pero igual conviene
        # avisar en su propia linea cuando cambia de fase) -- se afina mas
        # abajo segun el resultado del chequeo de obstaculo / del planner.
        bridge_state = "SIN_META" if target is None else (
            "CERCA_CHECKPOINT" if goal.distance_m < self.claim_radius_m * 1.5 else "NAVEGANDO")
        self._reporter.state_change(bridge_state, goal_desc)

        def _row(action: str, state: str = bridge_state) -> None:
            self._reporter.row(iteration=self.stats.iterations, state=state,
                               pose=pose_now, target_desc=goal_desc, map_cells=map_cells,
                               trav=res.traversability, action=action)

        # El chequeo de colision usa SIEMPRE la observacion fresca: si algo se
        # cruzo recien, no queremos que el promedio del mapa lo diluya. Esto
        # corre cada frame, sin esperar al disparo espacial de mas abajo.
        if front_is_blocked(res.traversability, self.resolution):
            self.stats.blocked += 1
            self._consecutive_blocked += 1
            if self._consecutive_blocked >= self.obstacle_persist_frames and self.use_map:
                _row(f"REGIMEN CERCANO ({self._consecutive_blocked} frames bloqueado)",
                     state="RECUPERANDO")
                self._retroceso_y_recover(res.traversability)
            else:
                self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"), quiet=True)
                _row(f"frenado: obstaculo al frente ({self._consecutive_blocked}"
                     f"/{self.obstacle_persist_frames})", state="OBSTACULO")
            return
        self._consecutive_blocked = 0

        # ---- disparo espacial: solo llamar a GeNIE si hace falta ----------
        need_replan = True
        if pose_now is not None and self._plan_path_world is not None and self._plan_pose is not None:
            avance = math.hypot(pose_now.x - self._plan_pose.x, pose_now.y - self._plan_pose.y)
            remanente = self._plan_remaining_m(path_to_robot(self._plan_path_world, pose_now))
            need_replan = (avance >= self.replan_every_m
                          or remanente <= self.replan_min_remaining_m
                          or (now - self._plan_t) >= self.replan_max_s)

        plan = None
        if need_replan:
            plan = plan_on_bev(
                bev_traversability=plan_bev,
                observed_mask=plan_obs,
                goal_x_m=float(goal.x_right_m),
                goal_y_m=float(goal.y_forward_m),
                bev_resolution_m=(2.0 * side) / plan_bev.shape[1],
                config=self.planner_cfg,
                candidate_path_bank=self._path_bank(
                    plan_bev.shape, (2.0 * side) / plan_bev.shape[1]),
            )

            path = plan.final_path_xy_m
            if path is None or len(path) < 2:
                self.stats.plans_empty += 1
                self._consecutive_empty += 1
                if self._consecutive_empty >= self.recovery_after_empty:
                    _row(f"RECUPERACION ({self._consecutive_empty} planes vacios seguidos)",
                         state="RECUPERANDO")
                    self._recover()
                else:
                    self.send(DriveCommand(0.0, 0.0, "el planner no encontro camino"), quiet=True)
                    _row(f"sin camino ({self._consecutive_empty}/{self.recovery_after_empty})",
                         state="SIN_CAMINO")
                return

            self._consecutive_empty = 0
            self.stats.plans_ok += 1

            if pose_now is not None:
                self._plan_path_world = path_to_world(path, pose_now)
                self._plan_pose = Pose(pose_now.x, pose_now.y, pose_now.theta)
                self._plan_t = now
            path_robot = path
        else:
            path_robot = path_to_robot(self._plan_path_world, pose_now)

        if self._send_path_command(path_robot, _row):
            if plan is not None:
                self._maybe_dump_debug(rgb, res, plan)

    def _path_bank(self, bev_shape: tuple[int, int],
                   bev_resolution_m: float) -> list[np.ndarray] | None:
        """Banco de caminos candidatos, calculado una sola vez por corrida.

        sample_paths_polynomial/sample_paths_uniform_fan NO miran la escena:
        sus entradas son la pose del robot en la grilla (siempre el
        centro-abajo), grid_size, y la semilla, todas constantes mientras no
        cambie la forma del BEV. Medido en una RTX 2080: recalcularlo
        costaba ~2.95 s de los ~3.15 s que tardaba plan_on_bev, o sea mas que
        el dead-man watchdog del SDK (CONTROL_WATCHDOG_S, 3 s por defecto) --
        el rover se frenaba solo entre comando y comando.

        Devuelve None (y entonces plan_on_bev lo calcula por su cuenta) si
        include_goal_in_path_bank esta activo: en ese caso el banco SI
        depende de la meta y cachearlo daria caminos de una meta vieja.
        """
        if bool(self.planner_cfg.include_goal_in_path_bank):
            return None
        if self._bank is None or self._bank_shape != bev_shape:
            grid = int(self.planner_cfg.grid_size)
            start = (bev_shape[0] - 1, bev_shape[1] // 2)
            t0 = time.time()
            if bool(self.planner_cfg.uniform_path_bank):
                self._bank = sample_paths_uniform_fan(
                    bev_shape=bev_shape,
                    bev_resolution_m=bev_resolution_m,
                    grid_size=grid,
                    max_angle_deg=float(self.planner_cfg.fan_max_angle_deg),
                    num_headings=int(self.planner_cfg.fan_num_headings),
                    num_samples=int(self.planner_cfg.path_num_samples),
                )
            else:
                self._bank = sample_paths_polynomial(
                    robot=_resize_pixel(start, bev_shape, grid),
                    num_goals=int(self.planner_cfg.num_goals),
                    num_mid_points_per_goal=int(self.planner_cfg.num_mid_points_per_goal),
                    num_samples=int(self.planner_cfg.path_num_samples),
                    grid_size=grid,
                    goal=None,
                    include_random_goals=bool(self.planner_cfg.include_random_goals),
                    random_seed=self.planner_cfg.random_seed,
                )
            self._bank_shape = bev_shape
            print(f"[bridge] banco de {len(self._bank)} caminos candidatos precalculado "
                  f"en {time.time() - t0:.1f} s (se reusa durante toda la corrida)")
        return self._bank

    def _plan_remaining_m(self, path_robot: np.ndarray) -> float:
        """Longitud del tramo del camino cacheado que todavia esta adelante
        del robot (y_forward >= 0), reproyectado a la pose actual."""
        p = np.asarray(path_robot, dtype=np.float64)
        ahead = p[p[:, 1] >= 0.0]
        if len(ahead) < 2:
            return 0.0
        seg = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
        return float(np.sum(seg))

    def _send_path_command(self, path_robot: np.ndarray, _row) -> bool:
        """Arma el comando a partir de un camino en marco robot (recien
        planificado o reproyectado de un plan cacheado), aplica anti-titubeo
        y anti-bucle, y lo manda a traves de la tabla de consola (`_row`, la
        clausura armada en `_step()`). Devuelve False si disparo _unstick(),
        que ya mando su propio comando y su propia linea de evento.

        Un comando con linear=0 y angular!=0 es "girar en el lugar". Si eso
        se repite, el robot esta atrapado: cada giro le muestra una escena
        que vuelve a pedir girar, y sin memoria no sale solo.
        """
        cmd = self.follower.command(path_robot, committed=True)
        cmd = self._apply_commit(cmd, path_robot)

        if cmd.linear == 0.0 and cmd.angular != 0.0:
            self._consecutive_turns += 1
            self._turn_sign_history.append(1.0 if cmd.angular > 0 else -1.0)
            if self._consecutive_turns >= self.max_consecutive_turns:
                _row(f"DESATASCO ({self._consecutive_turns} giros seguidos)",
                     state="DESATASCO")
                self._unstick()
                return False
        else:
            self._consecutive_turns = 0
            self._turn_sign_history.clear()

        self.send(cmd, quiet=True)
        _row(f"{cmd.linear:+.2f}m/s {cmd.angular:+.2f}rad/s  {cmd.reason}")
        return True

    # ------------------------------------------------------- odometria ciega

    def _track_pose_during_maneuver(self) -> None:
        """Pide telemetria y actualiza odometria durante una maniobra ciega
        (_unstick / _barrido_ciego / _retroceder / _girar_hacia).

        BUG que esto arregla (ver fix-recovery-odometria-bridge.md del
        proyecto): estas maniobras mueven al robot con send() + sleep(), y
        Odometry.update() tiene un guard de max_dt_s (0.5s por defecto): si
        el hueco entre dos muestras integradas supera eso, DESCARTA el
        intervalo entero en vez de integrarlo. Como estas maniobras duran
        1-3s tipicamente, sin alimentar telemetria en el medio el robot se
        mueve de verdad pero self.odometry.pose (y el PersistentMap, que
        ancla todo en ese marco) quedan exactamente igual que antes de la
        maniobra -- justo en el momento en que mas importa que no lo hagan
        (recuperandose de estar atascado o retrocediendo de un obstaculo).

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
        print(f"[bridge] ATASCADO: {giros} giros seguidos sin avanzar "
              f"(sentido dominante {'izq' if sentido > 0 else 'der'}). "
              f"Fuerzo un avance de {self.unstick_forward_s:.1f} s.")

        self.send(DriveCommand(self.follower.max_linear, 0.0, "avance forzado"))
        t0 = time.time()
        while time.time() - t0 < self.unstick_forward_s and not self._stop_requested:
            time.sleep(0.15)
            self._track_pose_during_maneuver()
            try:
                rgb, _ = self.client.front_frame()
                res = self.perception.process(rgb)
                if front_is_blocked(res.traversability, self.resolution):
                    print("[bridge] obstaculo durante el avance forzado, corto")
                    break
            except Exception:
                break

        self.send(DriveCommand(0.0, 0.0, "fin del avance forzado"))
        # Una muestra mas despues del freno, para capturar el movimiento
        # hasta que el robot realmente se detiene.
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

    def _recover(self) -> None:
        """Recuperacion tras varios planes vacios seguidos: buscar un rumbo
        transitable. Informada por mapa+VLM cuando hay memoria espacial
        (_recover_informado); si no, barrido ciego en el lugar.
        """
        print("[bridge] RECUPERACION: buscando rumbo transitable "
              f"({self._consecutive_empty} planes vacios seguidos)")
        if self.use_map and self.pmap is not None and self.odometry is not None:
            self._recover_informado()
        else:
            self._barrido_ciego()
        self.heading_est.reset_track()  # el track GPS previo ya no dice el rumbo
        self._consecutive_empty = 0

    def _barrido_ciego(self) -> None:
        """Girar en el lugar sin evaluar nada: el ultimo recurso cuando no
        hay mapa, ni el mapa ni el VLM dieron un rumbo confiable.

        Loop de polling (en vez de un unico time.sleep) para alimentar
        odometria durante todo el barrido (ver _track_pose_during_maneuver)
        y permitir que Ctrl-C corte la maniobra a mitad de camino.
        """
        self.stats.recoveries_ciegas += 1
        print("[bridge]   barrido ciego")
        self.send(DriveCommand(0.0, self.follower.angular_sign * self.follower.turn_speed,
                               "barrido de recuperacion"))
        t0 = time.time()
        while time.time() - t0 < self.recovery_turn_s and not self._stop_requested:
            time.sleep(0.15)
            self._track_pose_during_maneuver()
        self.send(DriveCommand(0.0, 0.0, "fin del barrido"))
        self._track_pose_during_maneuver()

    # ---------------------------------------------------------- regimen cercano

    def _retroceso_y_recover(self, bev: np.ndarray) -> None:
        """Regimen cercano (guia tecnica S04): por debajo de ~0.6 m el BEV
        instantaneo ya no es una fuente de informacion valida para planificar
        -- el obstaculo tapa la mayor parte del campo visual y la distorsion
        de lente es maxima justo donde estaria el hueco libre. En vez de
        seguir frenando para siempre, se usa el mapa persistente -- que
        todavia tiene la observacion buena de cuando el obstaculo estaba mas
        lejos -- para decidir si conviene retroceder y hacia donde girar.

        `bev` es la observacion fresca de la iteracion que disparo esto (la
        misma que ya evaluo front_is_blocked), solo para el log de clearance.
        """
        assert self.pmap is not None and self.odometry is not None
        self.stats.near_regime_activations += 1
        pose = self.odometry.pose
        clearance = front_clearance_m(bev, self.resolution, max_check_m=1.2)
        print(f"[bridge] REGIMEN CERCANO: {self._consecutive_blocked} frames bloqueado "
              f"seguidos, clearance={clearance:.2f} m")

        libre_pct, cobertura_pct = self._map_free_and_coverage(
            pose, heading_rel_deg=180.0, radius_m=self.retroceso_max_m)
        print(f"[bridge]   mapa detras del robot: libre={libre_pct:.0f}% cobertura={cobertura_pct:.0f}%")
        if libre_pct >= self.retroceso_min_libre_pct and cobertura_pct >= self.retroceso_min_cobertura_pct:
            self._retroceder()
        else:
            print("[bridge]   detras no parece seguro (o sin datos todavia), salteo el retroceso")

        self._recover_informado()

        self.heading_est.reset_track()
        self._consecutive_turns = 0
        self._turn_sign_history.clear()
        self._consecutive_empty = 0
        self._commit_side = 0
        self._consecutive_blocked = 0
        self._plan_path_world = None
        self._plan_pose = None

    def _map_free_and_coverage(self, pose: Pose, heading_rel_deg: float,
                               radius_m: float) -> tuple[float, float]:
        """Consulta el mapa persistente 'mirando' hacia heading_rel_deg
        (relativo al rumbo actual) hasta radius_m. Devuelve
        (libre_pct, cobertura_pct) sobre las celdas de esa ventana.

        Reutiliza PersistentMap.extract_bev con una pose sintetica rotada en
        vez de agregar geometria nueva: extract_bev ya sabe recortar una
        ventana del mapa en cualquier orientacion.
        """
        assert self.pmap is not None
        probe = Pose(pose.x, pose.y, pose.theta + math.radians(heading_rel_deg))
        out = max(8, int(round(radius_m / self.resolution)))
        bev, observed = self.pmap.extract_bev(probe, radius_m, radius_m * 0.6, out, out)
        mask = observed.astype(bool)
        cobertura_pct = float(np.mean(observed)) * 100.0
        if not np.any(mask):
            return 0.0, cobertura_pct
        libre_pct = float(np.mean(bev[mask] > 0.4)) * 100.0
        return libre_pct, cobertura_pct

    def _retroceder(self) -> None:
        """Retrocede en pasos cortos y verificados: es la unica accion que
        reduce la ocupacion angular de un obstaculo pegado al frente (girar
        en el lugar no alcanza, la zona ciega gira con el robot)."""
        assert self.odometry is not None
        self.stats.retrocesos += 1
        print(f"[bridge]   retrocediendo hasta {self.retroceso_max_m:.2f} m "
              f"en pasos de {self.retroceso_paso_m:.2f} m")
        start = self.odometry.pose
        start_pose = Pose(start.x, start.y, start.theta)
        step_s = self.retroceso_paso_m / max(abs(self.retroceso_linear), 1e-3)
        recorrido = 0.0

        while recorrido < self.retroceso_max_m and not self._stop_requested:
            self.send(DriveCommand(self.retroceso_linear, 0.0, "retroceso (regimen cercano)"))
            t0 = time.time()
            while time.time() - t0 < step_s and not self._stop_requested:
                time.sleep(0.1)
            self.send(DriveCommand(0.0, 0.0, "pausa de retroceso"))

            self._track_pose_during_maneuver()
            pose = self.odometry.pose
            recorrido = math.hypot(pose.x - start_pose.x, pose.y - start_pose.y)
            try:
                rgb, _ = self.client.front_frame()
                res = self.perception.process(rgb)
                if not front_is_blocked(res.traversability, self.resolution):
                    print(f"[bridge]   frente liberado tras retroceder {recorrido:.2f} m")
                    break
            except Exception:
                break

        self.send(DriveCommand(0.0, 0.0, "fin del retroceso"))

    def _recover_informado(self) -> None:
        """Elige un rumbo de escape en tres pasos, del mas barato al mas caro:
        mapa persistente (sin red, sin latencia) -> VLM (si el mapa no dio un
        candidato confiable) -> barrido ciego (ultimo recurso).
        """
        assert self.pmap is not None and self.odometry is not None
        pose = self.odometry.pose

        mejor_heading, mejor_libre = None, -1.0
        for h in self.recovery_headings_deg:
            libre_pct, cobertura_pct = self._map_free_and_coverage(pose, float(h), self.heading_search_radius_m)
            print(f"[bridge]   rumbo {h:+.0f} grados: libre={libre_pct:.0f}% cobertura={cobertura_pct:.0f}%")
            if cobertura_pct >= self.recovery_min_cobertura_pct and libre_pct > mejor_libre:
                mejor_heading, mejor_libre = float(h), libre_pct

        if mejor_heading is not None:
            print(f"[bridge]   elijo rumbo {mejor_heading:+.0f} grados por mapa (libre={mejor_libre:.0f}%)")
            self.stats.recoveries_por_mapa += 1
            self._girar_hacia(mejor_heading)
            return

        if self.use_vlm_recovery:
            decision = self._preguntar_vlm()
            if decision is not None:
                heading_deg = {"izquierda": -75.0, "derecha": 75.0,
                               "adelante": 0.0, "atras": 180.0}[decision.heading]
                print(f"[bridge]   VLM sugiere '{decision.heading}' "
                      f"(confianza {decision.confidence:.2f}): {decision.reason}")
                self.stats.recoveries_por_vlm += 1
                self._girar_hacia(heading_deg)
                return

        self._barrido_ciego()

    def _preguntar_vlm(self):
        try:
            from .vlm_recovery import ask_recovery_heading
        except Exception as exc:
            print(f"[bridge]   vlm_recovery no disponible ({exc}), sigo sin VLM")
            return None
        rgb, _ = self.client.front_frame()
        return ask_recovery_heading(rgb, min_confidence=self.vlm_recovery_min_confidence,
                                    timeout_s=self.vlm_recovery_timeout_s)

    def _girar_hacia(self, heading_rel_deg: float, step_deg: float | None = None) -> None:
        """Gira en pasos hacia heading_rel_deg (relativo al rumbo del robot al
        momento de llamar), re-verificando con el mapa despues de cada paso.
        Nunca se compromete de una sola vez a un angulo grande calculado de
        antemano, porque para el final del giro esos datos ya pueden estar
        viejos.

        La duracion de cada paso sale de recovery_deg_per_s, que es la tasa de
        giro REAL del robot en grados por segundo.
        """
        assert self.odometry is not None
        if abs(heading_rel_deg) < 1e-6:
            return
        step_deg = float(self.recovery_step_deg if step_deg is None else step_deg)
        ang = self.follower.angular_sign * math.copysign(self.recovery_turn_speed, heading_rel_deg)
        ang = float(np.clip(ang, -self.follower.max_angular, self.follower.max_angular))
        paso_s = step_deg / max(self.recovery_deg_per_s, 1e-3)

        girado = 0.0
        while abs(girado) < abs(heading_rel_deg) and not self._stop_requested:
            self.send(DriveCommand(0.0, ang, f"girando hacia {heading_rel_deg:+.0f} grados (regimen cercano)"))
            t0 = time.time()
            while time.time() - t0 < paso_s and not self._stop_requested:
                time.sleep(0.1)
            girado += math.copysign(step_deg, heading_rel_deg)

            self._track_pose_during_maneuver()
            pose = self.odometry.pose
            libre_pct, cobertura_pct = self._map_free_and_coverage(pose, 0.0, self.heading_search_radius_m)
            if cobertura_pct >= self.recovery_min_cobertura_pct and libre_pct >= self.retroceso_min_libre_pct:
                print(f"[bridge]   frente libre por mapa tras girar {girado:+.0f} grados, corto")
                break

        self.send(DriveCommand(0.0, 0.0, "fin del giro"))
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
        print(f"  regimen cercano:        {s.near_regime_activations} "
              f"(retrocesos: {s.retrocesos})")
        print(f"  recuperaciones:         mapa={s.recoveries_por_mapa}  "
              f"vlm={s.recoveries_por_vlm}  ciegas={s.recoveries_ciegas}")
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
        self._print_heading_diagnosis()

    def _print_heading_diagnosis(self) -> None:
        """Resume el desacuerdo brujula vs GPS y dice que hacer con el.

        Se usa estadistica CIRCULAR (promediar el vector unitario de cada
        angulo) y no un promedio comun: con angulos, +179 y -179 estan a 2
        grados uno del otro pero su media aritmetica da 0, que es el lado
        opuesto. R es la longitud del vector promedio: cerca de 1 significa
        que las muestras apuntan todas para el mismo lado (offset sistematico,
        corregible); cerca de 0 significa que estan repartidas (ruido).
        """
        muestras = self._disagreements
        print("\n  --- rumbo: brujula vs GPS ---")
        if len(muestras) < 5:
            print(f"  muestras utiles:        {len(muestras)} (pocas para concluir; "
                  "hace falta que el robot avance lo suficiente como para que "
                  "gps_track se active)")
            return
        radianes = np.radians(np.asarray(muestras, dtype=np.float64))
        media = math.degrees(math.atan2(np.mean(np.sin(radianes)), np.mean(np.cos(radianes))))
        r = float(np.hypot(np.mean(np.sin(radianes)), np.mean(np.cos(radianes))))
        dispersion = math.degrees(math.sqrt(-2.0 * math.log(r))) if r > 1e-9 else 180.0
        print(f"  muestras utiles:        {len(muestras)}")
        print(f"  desacuerdo medio:       {media:+.0f} grados")
        print(f"  concentracion R:        {r:.2f}  (dispersion ~{dispersion:.0f} grados)")
        if abs(media) < 10.0:
            print("  -> SIN OFFSET: las dos fuentes coinciden en promedio."
                  " No hay nada que calibrar.")
            if r < 0.9:
                print("     La dispersion que queda no es sesgo sino desacuerdo"
                      " momentaneo: el track GPS mide la CUERDA entre dos"
                      " muestras, asi que mientras el robot gira se queda"
                      " atras. Por eso trust_orientation: true.")
        elif r >= 0.7:
            print(f"  -> OFFSET SISTEMATICO. Pone en el config:")
            print(f"     navigation.orientation_offset_deg: {media:+.0f}")
            print("     Con eso las dos fuentes coinciden y desaparecen los saltos"
                  " de rumbo al alternar entre ellas.")
        elif r >= 0.4:
            print("  -> offset parcial: hay tendencia pero con mucha dispersion."
                  " Probar el offset ayuda, pero conviene la fusion con odometria.")
        else:
            print("  -> RUIDO, no offset. El magnetometro no es corregible con una"
                  " constante: hay que usar pose.theta de la odometria como rumbo"
                  " de corto plazo y el GPS solo como referencia absoluta.")


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
