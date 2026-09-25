"""Bucle de control: Earth Rover SDK <-> SAM-TP <-> planner BEV de GeNIE.

Por defecto arranca en DRY-RUN: calcula todo e imprime los comandos sin
enviarlos. Para que el robot se mueva de verdad hace falta pasar --go.

    # simulacro, el robot no se mueve
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml

    # de verdad
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml --go \
        --start-mission --max-seconds 120 --debug-dir debug/run1

    # con ruta grabada como guia secundaria entre checkpoints oficiales
    # (genie/rutas/mision2.json, ver genie_rover/route.py). Estos puntos son
    # solo apoyo de navegacion: nunca se reclaman en el SDK y nunca
    # condicionan el reached del checkpoint oficial, que sigue siendo lo
    # unico que hace avanzar la mision.
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml \
        --route genie/rutas/mision2.json
"""

from __future__ import annotations

import argparse
import csv
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
    check_checkpoint_reached,
    front_clearance_m,
    front_is_blocked,
    goal_from_gps,
    path_to_robot,
    path_to_world,
    wrap_deg,
)
from .gps_guard import GpsGuard, GpsGuardStatus
from .odometry import Odometry, OdometryConfig, Pose, estimate_roll_pitch, wrap_rad
from .perception import PerceptionPipeline
from .persistent_map import MapConfig, PersistentMap
from .route import (
    DASHBOARD_JSON_DEFAULT,
    DashboardOverlay,
    RouteConfig,
    cargar_rutas,
    gps_valido,
)
from .sdk_client import Checkpoint, RoverClient, RoverError
from .velocity_governor import GovernorConfig, VelocityGovernor


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
    escaneos_360: int = 0
    recoveries_por_vlm: int = 0
    recoveries_ciegas: int = 0
    governor_clamps: int = 0
    governor_stops: int = 0


def _safe_reset_recovery_state(b: Any) -> None:
    """Limpia todo el estado interno residual tras una maniobra de recuperacion,
    asegurando que la navegacion normal inicie desde un estado limpio."""
    if hasattr(b, "_reset_recovery_state") and callable(getattr(b, "_reset_recovery_state")):
        try:
            b._reset_recovery_state()
            return
        except Exception:
            pass
    if hasattr(b, "heading_est") and b.heading_est is not None:
        try:
            b.heading_est.reset_track()
        except Exception:
            pass
    b._consecutive_turns = 0
    if hasattr(b, "_turn_sign_history") and isinstance(b._turn_sign_history, list):
        b._turn_sign_history.clear()
    b._consecutive_empty = 0
    b._consecutive_empty_recoveries = 0
    b._commit_side = 0
    b._commit_until = 0.0
    b._consecutive_blocked = 0
    b._plan_path_world = None
    b._plan_pose = None


class Bridge:
    def __init__(self, cfg: dict, dry_run: bool = True, debug_dir: str | None = None,
                 log_tilt: bool = False, route: list[str] | str | None = None,
                 dashboard_json: str | None = None):
        self.cfg = cfg
        self.dry_run = bool(dry_run)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        debug_cfg = cfg.get("debug", {}) if isinstance(cfg.get("debug"), dict) else {}
        self.log_tilt = bool(log_tilt or debug_cfg.get("log_tilt", False))
        self._tilt_csv_file = None
        self._tilt_csv_writer = None
        self._tilt_csv_path: Path | None = None
        self._last_tilt_log_time = 0.0
        self._last_front_blocked = False
        self._last_cmd: DriveCommand | None = None

        if self.log_tilt:
            tilt_dir = self.debug_dir if self.debug_dir else Path("debug")
            tilt_dir.mkdir(parents=True, exist_ok=True)
            self._tilt_csv_path = tilt_dir / "tilt_log.csv"
            try:
                self._tilt_csv_file = open(self._tilt_csv_path, "w", newline="", buffering=1)
                self._tilt_csv_writer = csv.writer(self._tilt_csv_file)
                self._tilt_csv_writer.writerow(["timestamp", "pitch_deg", "roll_deg", "linear", "angular", "front_is_blocked"])
                print(f"[bridge] DIAG 2: log CSV de inclinacion activo en {self._tilt_csv_path}")
            except Exception as exc:
                print(f"[bridge] error al inicializar tilt_log.csv: {exc}")
                self._tilt_csv_file = None
                self._tilt_csv_writer = None

        self.client = RoverClient(cfg["rover"]["base_url"], timeout=cfg["rover"].get("timeout_s", 5.0))
        self.perception = PerceptionPipeline(cfg)

        nav = cfg["navigation"]
        self.heading_est = HeadingEstimator(
            min_displacement_m=nav.get("heading_min_displacement_m", 1.5),
            orientation_offset_deg=nav.get("orientation_offset_deg", 0.0),
            orientation_sign=nav.get("orientation_sign", 1.0),
            trust_orientation=nav.get("trust_orientation", False),
            use_ekf_udp=nav.get("use_ekf_udp", True),
            ekf_staleness_s=nav.get("ekf_staleness_s", 1.5),
            ekf_weight=nav.get("ekf_weight", 1.0),
            ekf_smooth_tau_s=float(nav.get("ekf_smooth_tau_s", 0.6)),  # ASUMIDO
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
        # checkpoint_reached_radius_m: Umbral de llegada al checkpoint (portado de ROS 2)
        # Origen: frodobot_rover.yaml (13.0m nominal, alineado con ERC y tolerancias GNSS de ROS 2)
        self.checkpoint_reached_radius_m = float(
            nav.get("checkpoint_reached_radius_m", nav.get("claim_radius_m", 13.0))
        )
        self.claim_radius_m = self.checkpoint_reached_radius_m
        self._base_checkpoint_radius_m = self.checkpoint_reached_radius_m
        self._current_checkpoint_radius_m = self.checkpoint_reached_radius_m
        self._last_target_sequence: int | None = None
        self.pre_claim_dwell_s = float(nav.get("pre_claim_dwell_s", 2.0))
        self._dwell_start_time: float | None = None
        self._dwell_target_seq: int | None = None

        # ---- telemetria de inclinacion y gate de ambiguedad ----------------
        self.front_far_m = float(nav.get("front_far_m", 1.25))
        self._dist_confiable_m = self.front_far_m
        self._tilt_ambiguo_px = 0

        # ---- ruta grabada como guia secundaria (apoyo sin claim) -------------
        # Los puntos de ruta (genie/rutas/*.json) son SOLO apoyo de navegacion
        # entre checkpoints oficiales: nunca se reclaman en el SDK y nunca
        # condicionan el "reached" del oficial. Lo unico que hace avanzar la
        # mision son los checkpoints oficiales (self._checkpoints /
        # claim_checkpoint), evaluados siempre en _step contra el checkpoint
        # real, sin pasar por la ruta. Ver _meta_con_ruta().
        route_cfg_raw = cfg.get("route", {}) or {}
        route_names = []
        if route:
            route_names = [route] if isinstance(route, str) else list(route)
        elif route_cfg_raw.get("files"):
            route_names = list(route_cfg_raw["files"])

        route_pt_cfg = RouteConfig(**{k: v for k, v in route_cfg_raw.get("config", {}).items()
                                      if k in RouteConfig.__dataclass_fields__})
        self.ruta = cargar_rutas(route_names, route_pt_cfg) if route_names else None
        # Con el oficial mas cerca que esto, se va directo a el.
        self.oficial_directo_m = float(route_cfg_raw.get("oficial_directo_m", 15.0))
        # Mas lejos de la ruta que esto (arranque lejos, rodeo enorme): se
        # ignora la ruta y se va al oficial. Se retoma sola al volver cerca.
        self.ruta_abandono_m = float(route_cfg_raw.get("abandono_m", 15.0))
        self._route_stats = {"metas_ruta": 0, "metas_oficial": 0, "ruta_ignorada": 0}
        self._en_directo = False
        if self.ruta is not None:
            print(f"[ruta] total {self.ruta.total_m:.1f} m, {len(self.ruta.puntos)} puntos "
                  "(apoyo sin reached), "
                  f"lookahead {route_pt_cfg.lookahead_m:.1f} m, oficial directo < {self.oficial_directo_m:.1f} m")

        dash = dashboard_json or route_cfg_raw.get("dashboard_path")
        if dash is None and DASHBOARD_JSON_DEFAULT.parent.is_dir():
            dash = DASHBOARD_JSON_DEFAULT
        self.dashboard = (DashboardOverlay(dash, float(route_cfg_raw.get("dashboard_period_s", 1.0)))
                          if dash else None)
        self._dash_state = "arrancando"
        self._dash_target: dict | None = None

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
                wheel_radius_m=float(odo_cfg.get("wheel_radius_m", 0.0475)),
                track_width_m=float(odo_cfg.get("track_width_m", 0.15)),
                left_rpm_indices=tuple(odo_cfg.get("left_rpm_indices", (0, 2))),
                right_rpm_indices=tuple(odo_cfg.get("right_rpm_indices", (1, 3))),
                rotation_sign=float(odo_cfg.get("rotation_sign", 1.0)),
                use_gyro_for_rotation=bool(odo_cfg.get("use_gyro_for_rotation", True)),
                gps_correction=bool(odo_cfg.get("gps_correction", True)),
                min_gps_displacement_m=float(odo_cfg.get("min_gps_displacement_m", 1.0)),
                gyro_yaw_index=int(odo_cfg.get("gyro_yaw_index", 2)),
                gyro_sign=float(odo_cfg.get("gyro_sign", 1.0)),
                gyro_yaw_bias_dps=float(odo_cfg.get("gyro_yaw_bias_dps", 1.2784)),
                gyro_deadband_dps=float(odo_cfg.get("gyro_deadband_dps", 0.5)),
                ekf_heading_correction=bool(odo_cfg.get("ekf_heading_correction", True)),
                ekf_heading_max_age_s=float(odo_cfg.get("ekf_heading_max_age_s", 1.5)),
                heading_blend=float(odo_cfg.get("heading_blend", 0.3)),
                heading_blend_tau_s=float(odo_cfg.get("heading_blend_tau_s", 4.0)),  # ASUMIDO
                accel_gate_norm_tol=float(odo_cfg.get("accel_gate_norm_tol", 0.08)),
                accel_gate_std_tol=float(odo_cfg.get("accel_gate_std_tol", 0.06)),
                gyro_bias_x_dps=float(odo_cfg.get("gyro_bias_x_dps", 0.0831)),
                gyro_bias_y_dps=float(odo_cfg.get("gyro_bias_y_dps", -0.0098)),
                use_tilt_projection=bool(odo_cfg.get("use_tilt_projection", True)),
                tilt_blend_start_deg=float(odo_cfg.get("tilt_blend_start_deg", 5.0)),
                tilt_blend_max_deg=float(odo_cfg.get("tilt_blend_max_deg", 20.0)),
                tilt_max_staleness_s=float(odo_cfg.get("tilt_max_staleness_s", 5.0)),
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

        # ---- chequeo frontal configurable (modo pendiente / falso obstaculo) ----
        self.front_near_m = float(safety.get("front_near_m", 0.32))
        self.front_far_m = float(safety.get("front_far_m", 1.25))
        self.front_half_width_m = float(safety.get("front_half_width_m", 0.22))
        self.front_traversable_thresh = float(safety.get("front_traversable_thresh", 0.28))
        self.front_min_free_ratio = float(safety.get("front_min_free_ratio", 0.40))
        self.allow_reverse = bool(safety.get("allow_reverse", False))

        # ---- Gobernador dinámico de velocidad por latencia P95 (Física de frenado) ----
        gov_enabled = bool(safety.get("governor_enabled", True))
        self.governor = VelocityGovernor(GovernorConfig(
            enabled=gov_enabled,
            window_size=int(safety.get("governor_window_size", 30)),
            a_brake=float(safety.get("governor_a_brake", 1.5)),
            cmd_latency_s=float(safety.get("governor_cmd_latency_s", 2.0)),
            d_horizon_m=float(safety.get("governor_d_horizon_m", self.front_far_m)),
            margin=float(safety.get("governor_margin", 1.2)),
            min_speed_mps=float(safety.get("governor_min_speed_mps", 0.10)),
            max_linear_speed_mps=float(safety.get("governor_max_linear_mps", 0.557)),
            alpha_up=float(safety.get("governor_alpha_up", 0.25)),
        ))

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
        self.recovery_min_libre_pct = float(safety.get("recovery_min_libre_pct", 30.0))
        self.recovery_goal_weight = float(safety.get("recovery_goal_weight", 1.2))
        self.recovery_clearance_weight = float(safety.get("recovery_clearance_weight", 1.0))
        self.heading_search_radius_m = float(safety.get("heading_search_radius_m", 2.0))
        self._last_goal = None
        # Latencia de arranque de hardware (tiempo muerto medido en Test A: 1.5 - 2.5 s)
        self.recovery_startup_latency_s = float(safety.get("recovery_startup_latency_s", 2.0))
        self.recovery_turn_tolerance_deg = float(safety.get("recovery_turn_tolerance_deg", 15.0))
        self.recovery_turn_timeout_s = float(safety.get("recovery_turn_timeout_s", 25.0))
        # Recuperacion con VLM (genie_rover.vlm_recovery): se prueba solo si
        # el mapa no encontro un rumbo confiable. Nunca lanza excepcion -- si
        # falla (sin credenciales, sin red, timeout) el bridge cae al barrido
        # ciego de siempre.
        self.use_vlm_recovery = bool(safety.get("use_vlm_recovery", False))
        self.vlm_recovery_timeout_s = float(safety.get("vlm_recovery_timeout_s", 4.0))
        self.vlm_recovery_min_confidence = float(safety.get("vlm_recovery_min_confidence", 0.35))
        self.recovery_tilt_veto_deg = float(safety.get("recovery_tilt_veto_deg", 8.0))
        self.vlm_recovery_max_retries = int(safety.get("vlm_recovery_max_retries", 2))
        self.vlm_recovery_cooldown_s = float(safety.get("vlm_recovery_cooldown_s", 10.0))
        self._vlm_consecutive_calls = 0
        self._last_vlm_call_time = 0.0
        self._consecutive_blocked = 0

        # ---- Escaneo 360° en recuperación (previo a VLM) ---------------------
        self.use_recovery_scan = bool(safety.get("use_recovery_scan", True))
        self.recovery_scan_deg_per_s = float(safety.get("recovery_scan_deg_per_s", 20.0))
        self.recovery_scan_turn_speed = float(safety.get("recovery_scan_turn_speed", getattr(self, "recovery_turn_speed", 0.75)))
        self.recovery_scan_early_exit_libre_pct = float(safety.get("recovery_scan_early_exit_libre_pct", 70.0))
        self.recovery_scan_max_per_stuck = int(safety.get("recovery_scan_max_per_stuck", 1))
        self.recovery_scan_min_disp_m = float(safety.get("recovery_scan_min_disp_m", 1.0))
        self.recovery_scan_cooldown_s = float(safety.get("recovery_scan_cooldown_s", 30.0))
        self.recovery_scan_timeout_s = float(safety.get("recovery_scan_timeout_s", 35.0))
        self._scan_count_at_stuck = 0
        self._last_scan_pose: Pose | None = None
        self._last_scan_time = 0.0

        # ---- Guarda contra GPS malo (Fase 1) --------------------------------
        gg_cfg = cfg.get("gps_guard", {})
        self.gps_guard = GpsGuard(
            enabled=bool(gg_cfg.get("enabled", True)),
            v_max_phys_m_s=float(gg_cfg.get("v_max_phys_m_s", 1.111)),
            gps_jump_noise_margin_m=float(gg_cfg.get("gps_jump_noise_margin_m", 1.5)),
            bad_fix_consecutive_thresh=int(gg_cfg.get("bad_fix_consecutive_thresh", 3)),
            degraded_linear_scale=float(gg_cfg.get("degraded_linear_scale", 0.6)),
            degraded_max_linear=float(gg_cfg.get("degraded_max_linear", 0.25)),
            degraded_min_linear=float(gg_cfg.get("degraded_min_linear", 0.20)),
            time_without_anchor_thresh_s=float(gg_cfg.get("time_without_anchor_thresh_s", 15.0)),
            min_fix_interval_s=float(gg_cfg.get("min_fix_interval_s", 0.8)),
            min_fix_quality=int(gg_cfg.get("min_fix_quality", 1)),
            hdop_reject=float(gg_cfg.get("hdop_reject", 0.080)),
        )

        self.stats = LoopStats()
        # Muestras de (rumbo_activo - curso_gps_confiable) para evaluar offset cinemático real
        self._disagreements_gps: list[float] = []
        # Muestras de (compás_crudo - referencia) para diagnóstico de perturbación magnética local
        self._disagreements_compass: list[float] = []
        self._disagreements = self._disagreements_gps  # alias para compatibilidad retrospectiva
        self._stop_requested = False
        self._checkpoints: list[Checkpoint] = []
        self._latest_scanned = 0
        self._last_frame_ts = 0.0
        self._last_frame_change = time.time()
        self._consecutive_errors = 0
        self._consecutive_empty = 0
        self._consecutive_empty_recoveries = 0
        # Deteccion de giro sin avance: el planner puede quedar en un ciclo
        # donde cada giro revela una escena que vuelve a pedir girar. Sin
        # memoria entre frames, eso no se rompe solo.
        self._consecutive_turns = 0
        self._turn_sign_history: list[float] = []
        self.max_consecutive_turns = int(safety.get("max_consecutive_turns", 6))
        self.unstick_forward_s = float(safety.get("unstick_forward_s", 1.2))
        self.unstick_min_clearance_m = float(safety.get("unstick_min_clearance_m", 0.9))  # ASUMIDO: clearance frontal mínimo requerido para avance forzado en _unstick

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

    # ------------------------------------------------------------------ ciclo

    def _is_front_blocked(self, bev: np.ndarray) -> bool:
        return front_is_blocked(
            bev,
            getattr(self, "resolution", 0.03),
            near_m=getattr(self, "front_near_m", 0.40),
            far_m=getattr(self, "front_far_m", 1.25),
            half_width_m=getattr(self, "front_half_width_m", 0.22),
            traversable_thresh=getattr(self, "front_traversable_thresh", 0.26),
            min_free_ratio=getattr(self, "front_min_free_ratio", 0.35),
        )

    def request_stop(self, *_a) -> None:
        print("\n[bridge] parada solicitada, frenando ...")
        self._stop_requested = True

    def _maybe_log_tilt(self, linear: float, angular: float, front_blocked: bool) -> None:
        """DIAG 2: Log CSV de inclinación y comandos a ~5 Hz."""
        if not getattr(self, "log_tilt", False) or getattr(self, "_tilt_csv_writer", None) is None:
            return
        now = time.time()
        # Rate limit a ~5 Hz (periodo nominal 0.20s; permitimos log si dt >= 0.18s) # ASUMIDO: intervalo de muestreo ~5 Hz
        if (now - getattr(self, "_last_tilt_log_time", 0.0)) < 0.18:
            return
        self._last_tilt_log_time = now

        pitch_deg = None
        roll_deg = None
        if getattr(self, "odometry", None) is not None and hasattr(self.odometry, "current_roll_pitch"):
            rp = self.odometry.current_roll_pitch()
            if rp is not None:
                roll_deg = round(math.degrees(rp[0]), 2)
                pitch_deg = round(math.degrees(rp[1]), 2)
            elif getattr(self.odometry, "last_pitch", None) is not None:
                pitch_deg = round(math.degrees(self.odometry.last_pitch), 2)
                roll_deg = round(math.degrees(self.odometry.last_roll), 2) if getattr(self.odometry, "last_roll", None) is not None else 0.0

        try:
            self._tilt_csv_writer.writerow([
                f"{now:.3f}",
                f"{pitch_deg:.2f}" if pitch_deg is not None else "",
                f"{roll_deg:.2f}" if roll_deg is not None else "",
                f"{linear:.2f}",
                f"{angular:.2f}",
                str(front_blocked),
            ])
            self._tilt_csv_file.flush()
        except Exception:
            pass

    def _close_tilt_log(self) -> None:
        if getattr(self, "_tilt_csv_file", None) is not None:
            try:
                self._tilt_csv_file.flush()
                self._tilt_csv_file.close()
            except Exception:
                pass
            self._tilt_csv_file = None
            self._tilt_csv_writer = None

    def __del__(self) -> None:
        self._close_tilt_log()

    def send(self, cmd: DriveCommand) -> None:
        self._last_cmd = cmd
        self._maybe_log_tilt(cmd.linear, cmd.angular, getattr(self, "_last_front_blocked", False))
        if hasattr(self, "gps_guard") and self.gps_guard is not None:
            if self.gps_guard.level == 3:
                cmd = DriveCommand(0.0, 0.0, f"[GPS_GUARD Nivel 3 Parada Emergencia] {cmd.reason}")
            elif self.gps_guard.level == 2:
                scaled_lin = float(np.clip(cmd.linear * self.gps_guard.degraded_linear_scale, -1.0, 1.0))
                cmd = DriveCommand(scaled_lin, cmd.angular, f"[GPS_GUARD Nivel 2 Degradado x{self.gps_guard.degraded_linear_scale:.1f}] {cmd.reason}")
        tag = "DRY-RUN" if getattr(self, "dry_run", True) else "ENVIADO"
        print(f"  [{tag}] linear={cmd.linear:+.2f} angular={cmd.angular:+.2f}  {cmd.reason}")
        if not getattr(self, "dry_run", True) and hasattr(self, "client"):
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

    def _meta_con_ruta(self, guard_status, heading, target, oficial_goal, goal, goal_desc):
        """Decide si la meta local sale de la ruta grabada (secundaria, de
        apoyo) o del checkpoint oficial. Devuelve (goal, goal_desc); sin ruta
        cargada devuelve los mismos que recibe.

        Se sigue la ruta mientras: no termino, el robot esta a menos de
        abandono_m de ella y el oficial (si hay) esta a mas de
        oficial_directo_m. El claim/reached NUNCA depende de esto: lo decide
        siempre el bloque del checkpoint oficial en _step con la distancia
        real al checkpoint.
        """
        lat, lon = guard_status.effective_lat, guard_status.effective_lon
        fix_ok = gps_valido(lat, lon)

        if self.ruta is not None and fix_ok:
            # El progreso solo necesita posicion: se actualiza aunque todavia
            # no haya rumbo, y tambien mientras se va directo al oficial.
            self.ruta.update(lat, lon, time.time())

        if target is not None:
            self._dash_target = {"lat": target.latitude, "lon": target.longitude, "kind": "oficial"}
            self._dash_state = f"yendo al checkpoint oficial #{target.sequence}"
        else:
            self._dash_target = None
            self._dash_state = "sin checkpoint pendiente"

        if self.ruta is None:
            pass
        elif not fix_ok or heading is None:
            self._dash_state = "esperando " + ("fix GPS" if not fix_ok else "rumbo")
        elif self.ruta.terminada:
            self._dash_state += " (ruta terminada)"
        elif self.ruta.desvio_m > self.ruta_abandono_m:
            self._route_stats["ruta_ignorada"] += 1
            self._dash_state += f" (lejos de la ruta: {self.ruta.desvio_m:.0f} m)"
            goal_desc += f" | ruta ignorada (desvio {self.ruta.desvio_m:.0f} m)"
        elif oficial_goal is not None and self._directo_al_oficial(oficial_goal.distance_m):
            goal_desc += " | directo al oficial"
        else:
            lat_t, lon_t = self.ruta.objetivo()
            goal = goal_from_gps(lat, lon, heading, lat_t, lon_t, self.goal_range_m)
            self._last_goal = goal   # el recovery alinea contra la meta que se sigue
            self._route_stats["metas_ruta"] += 1
            self._dash_target = {"lat": lat_t, "lon": lon_t, "kind": "ruta"}
            self._dash_state = "siguiendo la ruta"
            extra = f" | cp#{target.sequence} a {oficial_goal.distance_m:.0f} m" if oficial_goal else ""
            goal_desc = (f"{self.ruta.descripcion()}, rel {goal.relative_bearing_deg:+.0f} "
                         f"grados{extra}")

        if oficial_goal is not None and self._dash_target and self._dash_target["kind"] == "oficial":
            self._route_stats["metas_oficial"] += 1

        if self.dashboard is not None:
            self.dashboard.escribir(self.ruta, lat if fix_ok else None, lon if fix_ok else None,
                                    self._dash_state, self._dash_target, self._latest_scanned)
        return goal, goal_desc

    def _directo_al_oficial(self, dist_m: float) -> bool:
        """Histeresis: se entra a ir directo al oficial a oficial_directo_m y
        solo se sale al alejarse 3 m mas. Sin esto el ruido GPS alterna entre
        ruta y oficial en cada fix cuando la distancia ronda el umbral."""
        salida_m = self.oficial_directo_m + 3.0
        self._en_directo = dist_m <= (salida_m if self._en_directo else self.oficial_directo_m)
        return self._en_directo

    def run(self, max_seconds: float | None = None) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        self.refresh_checkpoints()
        target = self.current_target()
        if self.ruta is not None:
            print(f"[bridge] Guia de ruta activa (secundaria, sin reached): {len(self.ruta.puntos)} "
                  f"puntos ({self.ruta.total_m:.1f} m)")
            if target is not None:
                print(f"[bridge] Destino final: checkpoint oficial #{target.sequence} ({target.latitude}, {target.longitude})")
            else:
                print("[bridge] Guiando por ruta hasta que aparezca un checkpoint oficial.")
        elif target is not None:
            print(f"[bridge] Objetivo: checkpoint #{target.sequence} "
                  f"({target.latitude}, {target.longitude})")
        else:
            print("[bridge] No hay checkpoint pendiente. Voy a navegar solo evitando "
                  "obstaculos, con la meta fija derecho adelante.")

        if self.dashboard is not None:
            # Publicar la ruta en el mapa YA: si la camara o el GPS tardan,
            # igual se ven los puntos intermedios.
            self.dashboard.escribir(self.ruta, None, None, "arrancando", None,
                                    self._latest_scanned, force=True)
            print(f"[ruta] overlay del mapa en {self.dashboard.path}")

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
            if self.dashboard is not None:
                try:
                    self.dashboard.apagar()
                except Exception:
                    pass
            self._close_tilt_log()
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
        t_step_start = time.time()
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

        # Actualizar odometria antes de la guarda y la percepcion para alimentar
        # camera_pose con la estimacion mas fresca posible de pose, roll y pitch.
        pose_now: Pose | None = None
        roll_pitch: tuple[float, float] | None = None
        if self.odometry is not None:
            pose_now = self.odometry.update(
                telem.raw,
                ekf_heading=getattr(telem, "ekf_heading", None),
                ekf_timestamp=getattr(telem, "ekf_heading_time", None),
                now=now,
            )
            roll_pitch = self.odometry.current_roll_pitch(now=now)
        elif "accels" in telem.raw:
            tilt_res = estimate_roll_pitch(telem.raw.get("accels", []))
            if tilt_res is not None:
                roll_pitch = (tilt_res["roll_rad"], tilt_res["pitch_rad"])

        heading = self.heading_est.update(
            telem.latitude,
            telem.longitude,
            telem.orientation,
            telem.timestamp,
            ekf_heading=getattr(telem, "ekf_heading", None),
            ekf_timestamp=getattr(telem, "ekf_heading_time", None),
        )

        # Actualizar guarda contra GPS malo (Fase 1)
        has_anchor = (self.heading_est.source != "none")
        guard_status = self.gps_guard.update(
            telem=telem,
            odom_pose=self.odometry.pose if self.odometry is not None else None,
            heading_deg=heading if heading is not None else telem.orientation,
            has_heading_anchor=has_anchor,
            now=now,
        )

        if guard_status.level == 2:
            self.heading_est.reset_track()

        if guard_status.level == 3:
            self.send(DriveCommand(0.0, 0.0, f"[GPS_GUARD Nivel 3] Parada de emergencia: {guard_status.reason}"))
            return

        target = self.current_target()
        reached = False
        dist_to_cp = 0.0
        if target is not None and heading is not None:
            # Si cambió el target, restaurar el radio geodésico nominal/base
            if self._last_target_sequence != target.sequence:
                self._last_target_sequence = target.sequence
                self._current_checkpoint_radius_m = self._base_checkpoint_radius_m
                self._dwell_start_time = None
                self._dwell_target_seq = None

            reached, dist_to_cp = check_checkpoint_reached(
                guard_status.effective_lat,
                guard_status.effective_lon,
                target.latitude,
                target.longitude,
                self._current_checkpoint_radius_m,
            )
            goal = goal_from_gps(guard_status.effective_lat, guard_status.effective_lon, heading,
                                 target.latitude, target.longitude, self.goal_range_m)
            self._last_goal = goal
            if not reached or not guard_status.can_claim_checkpoints:
                self._dwell_start_time = None
                self._dwell_target_seq = None

            goal_desc = (f"cp#{target.sequence} a {goal.distance_m:.0f} m, "
                         f"rel {goal.relative_bearing_deg:+.0f} grados")
        else:
            self._dwell_start_time = None
            self._dwell_target_seq = None
            goal = type("G", (), {"x_right_m": 0.0, "y_forward_m": self.goal_range_m})()
            goal_desc = "derecho adelante (sin meta GPS)"

        # Ruta grabada como guia secundaria: solo puede reemplazar la meta
        # LOCAL (hacia donde apuntar ahora). El reclamo de checkpoint de mas
        # abajo (reached, dist_to_cp) ya quedo fijado arriba contra el
        # checkpoint oficial, sin pasar por la ruta.
        oficial_goal = goal if (target is not None and heading is not None) else None
        goal, goal_desc = self._meta_con_ruta(guard_status, heading, target, oficial_goal,
                                              goal, goal_desc)

        roll = roll_pitch[0] if roll_pitch is not None else None
        pitch = roll_pitch[1] if roll_pitch is not None else None
        res = self.perception.process(rgb, roll_rad=roll, pitch_rad=pitch)
        front_far = float(getattr(self, "front_far_m", 1.25))
        self._dist_confiable_m = float(res.stats.get("distancia_confiable_m", front_far))
        self._tilt_ambiguo_px = int(res.stats.get("tilt_ambiguo_px", 0))

        # Integrar en el mapa persistente y planificar sobre el acumulado.
        # Esto corre SIEMPRE, a la frecuencia del frame: es lo que permite
        # desacoplar la decision (mas abajo, disparo espacial) de la
        # percepcion (esto).
        plan_bev, plan_obs = res.traversability, res.observed
        fwd, side = self.forward_range, self.side_range
        nota_mapa = ""
        if self.use_map and self.odometry is not None and self.pmap is not None and pose_now is not None:
            self.pmap.integrate(res.traversability, res.observed, pose_now,
                                self.forward_range, self.side_range, t=now)
            h, w = res.traversability.shape
            plan_bev, plan_obs = self.pmap.extract_bev(
                pose_now, self.plan_forward_m, self.plan_side_m, h, w)
            fwd, side = self.plan_forward_m, self.plan_side_m
            st = self.pmap.stats()
            nota_mapa = (f"  mapa: {st['celdas_vistas']} celdas  "
                         f"pose=({pose_now.x:+.2f},{pose_now.y:+.2f},"
                         f"{math.degrees(pose_now.theta):+.0f}gr)")

        # Desacuerdo brujula vs GPS. Solo se registra cuando la fuente es
        # gps_track: ahi las dos lecturas salieron de la MISMA llamada a
        # update() y son comparables. Con orientation(fallback) el rumbo GPS
        # guardado puede ser viejo y el robot haber girado desde entonces, asi
        # Desacuerdo Heading Activo vs Curso GPS (Ground Truth en rectas).
        # Solo se registra cuando el curso GPS es confiable (Fase 2 / track con baja incertidumbre).
        nota_desac = ""
        desac_gps = self.heading_est.disagreement_deg()
        if desac_gps is not None:
            self._disagreements_gps.append(float(desac_gps))
            nota_desac = f"  desac_gps={desac_gps:+.0f}gr"

        # Distorsión magnética del compás crudo respecto a la referencia confiable (informativo)
        dist_mag = self.heading_est.compass_distortion_deg()
        if dist_mag is not None:
            self._disagreements_compass.append(float(dist_mag))
        nota_tilt = ""

        if self.odometry is not None and self.odometry.last_pitch is not None:
            p_deg = math.degrees(self.odometry.last_pitch)
            r_deg = math.degrees(self.odometry.last_roll) if self.odometry.last_roll is not None else 0.0
            if not self.odometry.tilt_gate_open:
                norm_str = f"{self.odometry.last_accel_norm:.2f}g" if self.odometry.last_accel_norm is not None else "?g"
                nota_tilt = f"  tilt=GATE_CLOSED(|a|={norm_str})"
            elif abs(p_deg) >= 3.0 or abs(r_deg) >= 3.0 or self.odometry.last_blend_effective < (self.odometry.last_blend_nominal - 1e-4):
                nota_tilt = f"  tilt={p_deg:+.1f}°p/{r_deg:+.1f}°r(blend={self.odometry.last_blend_effective:.2f})"

        nota_bev_tilt = ""
        if "pitch_deg" in res.stats and "roll_deg" in res.stats:
            nota_bev_tilt = f" [cam_tilt={res.stats['pitch_deg']:+.1f}°p/{res.stats['roll_deg']:+.1f}°r]"

        nota_dconf = ""
        dist_conf = getattr(self, "_dist_confiable_m", front_far)
        tilt_amb_px = getattr(self, "_tilt_ambiguo_px", 0)
        if tilt_amb_px > 0 or dist_conf < (front_far - 1e-3):
            nota_dconf = f"  tilt_amb={tilt_amb_px}px dconf={dist_conf:.2f}m"

        print(f"[{self.stats.iterations:04d}] rumbo={heading if heading is None else round(heading)} "
              f"({self.heading_est.source})  meta: {goal_desc}  "
              f"celdas BEV={res.stats['bev_observed_cells']:.0f}{nota_bev_tilt}{nota_mapa}{nota_desac}{nota_tilt}{nota_dconf}")


        # El chequeo de colision usa SIEMPRE la observacion fresca: si algo se
        # cruzo recien, no queremos que el promedio del mapa lo diluya. Esto
        # corre cada frame, sin esperar al disparo espacial de mas abajo.
        blocked = self._is_front_blocked(res.traversability)
        self._last_front_blocked = blocked

        if self.dashboard is not None:
            estado = "bloqueado" if blocked else ("dwell" if self._dwell_start_time is not None else self._dash_state)
            try:
                self.dashboard.escribir(
                    self.ruta,
                    guard_status.effective_lat,
                    guard_status.effective_lon,
                    estado,
                    self._dash_target,
                    self._latest_scanned,
                )
            except Exception:
                pass
        if blocked:
            self.stats.blocked += 1
            self._consecutive_blocked += 1
            # Si había un dwell pre-reclamo en curso, se aborta inmediatamente por presencia de obstáculo
            if self._dwell_start_time is not None:
                print(f"[bridge] Dwell pre-reclamo cp#{target.sequence if target else '?'} ABORTADO por obstáculo al frente")
                self._dwell_start_time = None
                self._dwell_target_seq = None
            if self._consecutive_blocked >= self.obstacle_persist_frames and self.use_map:
                self._retroceso_y_recover(res.traversability, rgb=rgb)
            else:
                self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"))
            return
        self._consecutive_blocked = 0

        # ---- Evaluación de llegada a Checkpoint y Dwell Pre-Reclamo ----
        if target is not None and reached:
            if not guard_status.can_claim_checkpoints:
                self._dwell_start_time = None
                self._dwell_target_seq = None
                print(f"[bridge] Reclamo de checkpoint #{target.sequence} BLOQUEADO por GpsGuard ({guard_status.reason})")
            else:
                # Frente despejado y rover dentro del radio de reclamo
                if self._dwell_start_time is None or self._dwell_target_seq != target.sequence:
                    self._dwell_start_time = now
                    self._dwell_target_seq = target.sequence
                    print(f"[bridge] Checkpoint #{target.sequence} alcanzado ({dist_to_cp:.1f} m <= {self._current_checkpoint_radius_m:.1f} m). "
                          f"Iniciando reposo pre-reclamo ({self.pre_claim_dwell_s:.1f} s)...")
                    self.send(DriveCommand(0.0, 0.0, f"[Pre-Claim] Frenando rover para estabilizacion cp#{target.sequence} (0.0s/{self.pre_claim_dwell_s:.1f}s)"))
                    if hasattr(self, "governor") and self.governor is not None:
                        self.governor.update(time.time() - t_step_start)
                    return

                dwell_elapsed = now - self._dwell_start_time
                if dwell_elapsed < self.pre_claim_dwell_s:
                    self.send(DriveCommand(0.0, 0.0, f"[Pre-Claim] Estabilizando cp#{target.sequence} ({dwell_elapsed:.1f}s/{self.pre_claim_dwell_s:.1f}s)"))
                    if hasattr(self, "governor") and self.governor is not None:
                        self.governor.update(time.time() - t_step_start)
                    return

                # Dwell cumplido con frente despejado: ejecutar reclamo
                print(f"[bridge] Reposo pre-reclamo ({dwell_elapsed:.1f} s >= {self.pre_claim_dwell_s:.1f} s) completado para cp#{target.sequence}. Reclamando...")
                self._dwell_start_time = None
                self._dwell_target_seq = None
                ok, msg = self.client.claim_checkpoint()
                if ok:
                    print(f"[bridge] ✓ checkpoint #{target.sequence} alcanzado "
                          f"({dist_to_cp:.1f} m <= {self._current_checkpoint_radius_m:.1f} m): {msg}")
                    # Al avanzar de checkpoint con éxito, se restaura la tolerancia base
                    self._current_checkpoint_radius_m = self._base_checkpoint_radius_m
                    self.refresh_checkpoints()
                    self.send(DriveCommand(0.0, 0.0, f"[Pre-Claim] Checkpoint #{target.sequence} reclamado con éxito"))
                    if hasattr(self, "governor") and self.governor is not None:
                        self.governor.update(time.time() - t_step_start)
                    return
                else:
                    # DETECTOR DE RECHAZO DEL SDK (portado de gps_waypoint_controller.py:396)
                    # Si el SDK rechaza el reclamo (ej. fuera de geocerca estricta o error 422),
                    # estrangular la tolerancia a la mitad (13.0 -> 6.5 -> 3.25m, piso 0.5m)
                    # para obligar al rover a acercarse más antes de volver a intentar,
                    # evitando saturar la API en cada frame.
                    old_rad = self._current_checkpoint_radius_m
                    self._current_checkpoint_radius_m = max(0.5, self._current_checkpoint_radius_m * 0.5)
                    print(f"[bridge] cerca del checkpoint ({dist_to_cp:.1f} m <= {old_rad:.1f} m) "
                          f"pero rechazado por SDK: {msg}. Estrangulando tolerancia geodésica a "
                          f"{self._current_checkpoint_radius_m:.2f} m.")


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
                    self._recover()
                else:
                    self.send(DriveCommand(0.0, 0.0, "el planner no encontro camino"))
                return

            self._consecutive_empty = 0
            self._consecutive_empty_recoveries = 0
            self.stats.plans_ok += 1

            if pose_now is not None:
                self._plan_path_world = path_to_world(path, pose_now)
                self._plan_pose = Pose(pose_now.x, pose_now.y, pose_now.theta)
                self._plan_t = now
            path_robot = path
        else:
            path_robot = path_to_robot(self._plan_path_world, pose_now)

        step_duration = time.time() - t_step_start
        if hasattr(self, "governor") and self.governor is not None:
            self.governor.update(step_duration)

        if self._send_path_command(path_robot):
            if plan is not None:
                self._maybe_dump_debug(rgb, res, plan)

    def _path_bank(self, bev_shape: tuple[int, int],
                   bev_resolution_m: float) -> list[np.ndarray] | None:
        """Banco de caminos candidatos, calculado una sola vez por corrida.

        sample_paths_polynomial NO mira la escena: sus entradas son la pose
        del robot en la grilla (siempre el centro-abajo), grid_size, y la
        semilla, todas constantes mientras no cambie la forma del BEV. Medido
        en una RTX 2080: recalcularlo costaba ~2.95 s de los ~3.15 s que
        tardaba plan_on_bev, o sea mas que el dead-man watchdog del SDK
        (CONTROL_WATCHDOG_S, 3 s por defecto) -- el rover se frenaba solo
        entre comando y comando.

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

    def _send_path_command(self, path_robot: np.ndarray) -> bool:
        """Arma el comando a partir de un camino en marco robot (recien
        planificado o reproyectado de un plan cacheado), aplica anti-titubeo
        y anti-bucle, y lo manda. Devuelve False si disparo _unstick(), que ya
        mando su propio comando.

        Jerarquía de prioridades de modulación de velocidad lineal:
        1. PRIORIDAD 1 (Parada Absoluta): Dwell pre-reclamo, Nivel 3 de GpsGuard, u Obstáculo frontal.
           Fuerzan comando de parada (0.0, 0.0) y bypass total de planificación/seguimiento de trayectoria.
        2. PRIORIDAD 2 (Guarda de GPS Degradada - Nivel 2): Acota el rango admisible [min_linear, max_linear]
           (e.g. <= 0.25) para prevenir pérdida de rumbo con GNSS ruidoso.
        3. PRIORIDAD 3 (Gobernador de Velocidad por Latencia): Aplica clamp descendente min(linear, v_safe_throttle)
           por física cuadrática de frenado y latencia P95 del ciclo. Si v_safe < umbral, corta a 0.0 (Stop & Wait).
        Composición: Pre-claim dwell anula el envío antes de llegar aquí. En navegación activa, rige la intersección
        más restrictiva entre GpsGuard y Gobernador (mínimo de ambos clamps), garantizando que ninguna guarda
        pueda acelerar por encima de lo exigido por la otra.

        Un comando con linear=0 y angular!=0 es "girar en el lugar". Si eso
        se repite, el robot esta atrapado: cada giro le muestra una escena
        que vuelve a pedir girar, y sin memoria no sale solo.
        """
        cmd = self.follower.command(path_robot, committed=True)
        cmd = self._apply_commit(cmd, path_robot)

        # Prioridad 2: Guarda de GPS en modo degradado (Nivel 2)
        if hasattr(self, "gps_guard") and self.gps_guard is not None:
            cmd.linear = self.gps_guard.apply_throttle(cmd.linear)

        # Prioridad 3: Gobernador de velocidad por latencia P95 (aplica clamp descendente min())
        if hasattr(self, "governor") and self.governor is not None and self.governor.cfg.enabled:
            linear_before = cmd.linear
            cmd.linear = self.governor.apply_throttle_limit(cmd.linear)
            if linear_before > 0.0 and cmd.linear == 0.0:
                self.stats.governor_stops += 1
                v_safe_val = self.governor.filtered_v_safe
                v_safe_str = f"{v_safe_val:.2f}m/s" if v_safe_val is not None else "0m/s"
                cmd.reason = f"[GOBERNADOR STOP&WAIT (v_safe={v_safe_str})] {cmd.reason}"
            elif linear_before > cmd.linear:
                self.stats.governor_clamps += 1
                cmd.reason = f"[GOBERNADOR ({linear_before:.2f}->{cmd.linear:.2f})] {cmd.reason}"

        if cmd.linear == 0.0 and cmd.angular != 0.0:
            self._consecutive_turns += 1
            self._turn_sign_history.append(1.0 if cmd.angular > 0 else -1.0)
            if self._consecutive_turns >= self.max_consecutive_turns:
                self._unstick()
                return False
        else:
            self._consecutive_turns = 0
            self._turn_sign_history.clear()
            self._vlm_consecutive_calls = 0

        self.send(cmd)
        return True

    def _unstick(self) -> None:
        """Rompe el ciclo de giros sin avance o avanza en salida frontal despejada.

        Verifica antes de avanzar y durante el avance que exista clearance frontal
        suficiente en observaciones frescas para evitar colisiones con obstáculos cercanos.
        """
        self.stats.unstucks += 1
        giros = getattr(self, "_consecutive_turns", 0)
        unstick_s = float(getattr(self, "unstick_forward_s", 1.2))
        min_clearance = float(getattr(self, "unstick_min_clearance_m", 0.9))

        # 1. Chequeo preventivo de clearance antes del avance forzado
        try:
            pose_now = None
            if self.odometry is not None:
                t_telem = self.client.telemetry()
                pose_now = self.odometry.update(
                    t_telem.raw,
                    ekf_heading=getattr(t_telem, "ekf_heading", None),
                    ekf_timestamp=getattr(t_telem, "ekf_heading_time", None),
                )
            rgb, _ = self.client.front_frame()
            roll_pitch = self.odometry.current_roll_pitch() if (self.odometry is not None and hasattr(self.odometry, "current_roll_pitch")) else None
            r = roll_pitch[0] if roll_pitch is not None else None
            p = roll_pitch[1] if roll_pitch is not None else None
            res = self.perception.process(rgb, roll_rad=r, pitch_rad=p)
            if getattr(self, "use_map", True) and self.pmap is not None and pose_now is not None and hasattr(res, "observed"):
                self.pmap.integrate(res.traversability, res.observed, pose_now,
                                    self.forward_range, self.side_range, t=time.time())

            init_clearance = front_clearance_m(
                res.traversability,
                getattr(self, "resolution", 0.03),
                near_m=getattr(self, "front_near_m", 0.40),
                max_check_m=1.2,
                half_width_m=getattr(self, "front_half_width_m", 0.22),
                traversable_thresh=getattr(self, "front_traversable_thresh", 0.26),
                min_free_ratio=getattr(self, "front_min_free_ratio", 0.35),
            )
            blocked = self._is_front_blocked(res.traversability)
            if blocked or init_clearance < min_clearance:
                print(f"[bridge] avance forzado CANCELADO: clearance={init_clearance:.2f} m")
                _safe_reset_recovery_state(self)
                return
        except Exception as e:
            print(f"[bridge] avance forzado CANCELADO: error al verificar clearance ({e})")
            _safe_reset_recovery_state(self)
            return

        if giros > 0:
            sentido = sum(self._turn_sign_history)
            print(f"[bridge] ATASCADO: {giros} giros seguidos sin avanzar "
                  f"(sentido dominante {'izq' if sentido > 0 else 'der'}). "
                  f"Fuerzo un avance de {unstick_s:.1f} s.")
        else:
            print(f"[bridge] RECUPERACION FRONTAL: frente despejado en memoria pero sin plan viable. "
                  f"Fuerzo un avance de {unstick_s:.1f} s para superar zona ciega.")

        self.send(DriveCommand(self.follower.max_linear, 0.0, "avance forzado"))
        t0 = time.time()
        while time.time() - t0 < unstick_s and not self._stop_requested:
            time.sleep(0.15)
            try:
                pose_now = None
                if self.odometry is not None:
                    t_telem = self.client.telemetry()
                    pose_now = self.odometry.update(
                        t_telem.raw,
                        ekf_heading=getattr(t_telem, "ekf_heading", None),
                        ekf_timestamp=getattr(t_telem, "ekf_heading_time", None),
                    )
                rgb, _ = self.client.front_frame()
                roll_pitch = self.odometry.current_roll_pitch() if (self.odometry is not None and hasattr(self.odometry, "current_roll_pitch")) else None
                r = roll_pitch[0] if roll_pitch is not None else None
                p = roll_pitch[1] if roll_pitch is not None else None
                res = self.perception.process(rgb, roll_rad=r, pitch_rad=p)
                if getattr(self, "use_map", True) and self.pmap is not None and pose_now is not None and hasattr(res, "observed"):
                    self.pmap.integrate(res.traversability, res.observed, pose_now,
                                        self.forward_range, self.side_range, t=time.time())

                loop_clearance = front_clearance_m(
                    res.traversability,
                    getattr(self, "resolution", 0.03),
                    near_m=getattr(self, "front_near_m", 0.40),
                    max_check_m=1.2,
                    half_width_m=getattr(self, "front_half_width_m", 0.22),
                    traversable_thresh=getattr(self, "front_traversable_thresh", 0.26),
                    min_free_ratio=getattr(self, "front_min_free_ratio", 0.35),
                )
                if self._is_front_blocked(res.traversability) or loop_clearance < min_clearance:
                    print(f"[bridge] obstaculo durante el avance forzado (clearance={loop_clearance:.2f} m), corto")
                    break
            except Exception:
                break

        self.send(DriveCommand(0.0, 0.0, "fin del avance forzado"))
        _safe_reset_recovery_state(self)

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

    def _reset_recovery_state(self) -> None:
        """Limpia todo el estado interno residual tras una maniobra de recuperación,
        asegurando que la navegación normal inicie desde un estado limpio y sin comandos
        ni setpoints arrastrados."""
        _safe_reset_recovery_state(self)

    def _get_goal_relative_bearing_deg(self) -> float:
        """Devuelve el rumbo relativo al checkpoint objetivo en grados [-180, 180).
        0° = exactamente al frente, +90° = derecha, -90° = izquierda, 180° = detrás.
        Si no hay meta disponible o determinable, asume 0.0° (frente).
        """
        if hasattr(self, "_goal_relative_bearing_deg") and self._goal_relative_bearing_deg is not None:
            return float(wrap_deg(self._goal_relative_bearing_deg))

        if hasattr(self, "_last_goal") and self._last_goal is not None:
            rel = getattr(self._last_goal, "relative_bearing_deg", None)
            if rel is not None:
                return float(wrap_deg(rel))

        if hasattr(self, "current_target"):
            try:
                target = self.current_target()
                if target is not None and getattr(self, "heading_est", None) is not None and hasattr(self, "client"):
                    heading = self.heading_est.heading()
                    if heading is not None:
                        t_telem = self.client.telemetry()
                        if getattr(t_telem, "latitude", 0.0) != 0.0 or getattr(t_telem, "longitude", 0.0) != 0.0:
                            fwd = self.plan_forward_m if getattr(self, "use_map", False) else getattr(self, "forward_range", 3.0)
                            g = goal_from_gps(
                                t_telem.latitude, t_telem.longitude, heading,
                                target.latitude, target.longitude, max_range_m=fwd,
                            )
                            return float(wrap_deg(g.relative_bearing_deg))
            except Exception:
                pass

        return 0.0

    def _recover(self) -> None:
        """Recuperacion tras varios planes vacios seguidos: buscar un rumbo
        transitable. Informada por mapa+VLM cuando hay memoria espacial
        (_recover_informado); si no, barrido ciego en el lugar.
        """
        self._consecutive_empty_recoveries = getattr(self, "_consecutive_empty_recoveries", 0) + 1
        print("[bridge] RECUPERACION: buscando rumbo transitable "
              f"({self._consecutive_empty} planes vacios seguidos, "
              f"intento #{self._consecutive_empty_recoveries})")
        if self.use_map and self.pmap is not None and self.odometry is not None:
            self._recover_informado()
        else:
            self._barrido_ciego()
        _safe_reset_recovery_state(self)

    def _get_estimated_tilt_deg(self) -> tuple[float, float] | None:
        """Devuelve (|pitch_deg|, |roll_deg|) si hay estimacion vigente, o None."""
        if self.odometry is None:
            return None
        if hasattr(self.odometry, "current_roll_pitch"):
            try:
                rp = self.odometry.current_roll_pitch()
                if rp is not None:
                    return abs(math.degrees(rp[1])), abs(math.degrees(rp[0]))
            except Exception:
                pass
        last_pitch = getattr(self.odometry, "last_pitch", None)
        if last_pitch is not None:
            p = abs(math.degrees(last_pitch))
            last_roll = getattr(self.odometry, "last_roll", None)
            r = abs(math.degrees(last_roll)) if last_roll is not None else 0.0
            return p, r
        return None

    def _is_tilt_too_steep_for_recovery(self) -> tuple[bool, str]:
        """Verifica si la inclinacion del rover supera el umbral seguro para
        maniobras de alto riesgo (giro de 180° o retroceso lineal)."""
        tilt = self._get_estimated_tilt_deg()
        if tilt is None:
            return False, "sin datos de inclinacion (asumo nivelado)"
        p_deg, r_deg = tilt
        thresh = getattr(self, "recovery_tilt_veto_deg", 8.0)
        if p_deg >= thresh or r_deg >= thresh:
            return True, (f"inclinacion excesiva (pitch={p_deg:.1f}°, roll={r_deg:.1f}° >= "
                          f"umbral {thresh:.1f}°)")
        return False, f"inclinacion segura (pitch={p_deg:.1f}°, roll={r_deg:.1f}°)"

    def _barrido_ciego(self) -> None:
        """Girar en el lugar: el ultimo recurso cuando no hay mapa, o ni el
        mapa ni el VLM dieron un rumbo confiable. Evalua la mitad del BEV con
        mas espacio libre para elegir lado de giro (barrido condicional)."""
        self.stats.recoveries_ciegas += 1
        print("[bridge]   barrido condicional...")

        # Evaluar la mitad con mas espacio libre para elegir lado de giro
        signo = self.follower.angular_sign
        try:
            rgb, _ = self.client.front_frame()
            roll_pitch = self.odometry.current_roll_pitch() if self.odometry is not None else None
            r = roll_pitch[0] if roll_pitch is not None else None
            p = roll_pitch[1] if roll_pitch is not None else None
            res = self.perception.process(rgb, roll_rad=r, pitch_rad=p)
            bev = res.traversability
            h, w = bev.shape
            left_half = bev[:, :w // 2]
            right_half = bev[:, w // 2:]
            left_free = (left_half > 0.4).sum()
            right_free = (right_half > 0.4).sum()

            # angular_sign: -1 significa giro a la izquierda con angulo positivo
            if left_free > right_free:
                signo = -1.0 if self.follower.angular_sign < 0 else 1.0
                print(f"[bridge]     mas espacio a la IZQUIERDA ({left_free} vs {right_free}), girando izq")
            else:
                signo = 1.0 if self.follower.angular_sign < 0 else -1.0
                print(f"[bridge]     mas espacio a la DERECHA ({right_free} vs {left_free}), girando der")
        except Exception as e:
            print(f"[bridge]     error en barrido condicional ({e}), usando signo por defecto")

        cmd = DriveCommand(0.0, signo * self.follower.turn_speed, "barrido condicional")
        self.send(cmd)

        t0 = time.time()
        while time.time() - t0 < self.recovery_turn_s and not self._stop_requested:
            time.sleep(0.1)

        self.send(DriveCommand(0.0, 0.0, "fin del barrido"))
        _safe_reset_recovery_state(self)

    # ---------------------------------------------------------- regimen cercano

    def _retroceso_y_recover(self, bev: np.ndarray, rgb: np.ndarray | None = None) -> None:
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
        clearance = front_clearance_m(
            bev,
            getattr(self, "resolution", 0.03),
            near_m=getattr(self, "front_near_m", 0.40),
            max_check_m=1.2,
            half_width_m=getattr(self, "front_half_width_m", 0.22),
            traversable_thresh=getattr(self, "front_traversable_thresh", 0.26),
            min_free_ratio=getattr(self, "front_min_free_ratio", 0.35),
        )

        # DIAG 1: Log de activación del régimen cercano (clearance, mapa -90..+90, mitades BEV)
        rumbos_diag = [-90.0, -45.0, 0.0, 45.0, 90.0]
        mapa_strs = []
        radius_m = float(getattr(self, "heading_search_radius_m", 1.5))
        for h_deg in rumbos_diag:
            try:
                l_pct, c_pct = self._map_free_and_coverage(pose, heading_rel_deg=h_deg, radius_m=radius_m)
            except Exception:
                l_pct, c_pct = 0.0, 0.0
            mapa_strs.append(f"{h_deg:+.0f}°(lib={l_pct:.0f}%/cob={c_pct:.0f}%)")
        mapa_diag_str = " ".join(mapa_strs)

        h, w = bev.shape
        res_m = float(getattr(self, "resolution", 0.03))
        # ASUMIDO: Franja cercana de 0.4 m a 1.25 m para diagnóstico de cordones y paso lateral
        r_near = h - 1 - int(round(0.40 / res_m))
        r_far = max(0, h - 1 - int(round(1.25 / res_m)))
        r0 = max(0, min(h - 1, min(r_far, r_near)))
        r1 = max(1, min(h, max(r_far, r_near) + 1))
        band = bev[r0:r1, :]
        mid = w // 2
        trav_thresh = float(getattr(self, "front_traversable_thresh", 0.26))
        left_half = band[:, :mid]
        right_half = band[:, mid:]
        bev_izq_pct = float(np.mean(left_half > trav_thresh)) * 100.0 if left_half.size > 0 else 0.0
        bev_der_pct = float(np.mean(right_half > trav_thresh)) * 100.0 if right_half.size > 0 else 0.0

        print(f"[bridge] REGIMEN CERCANO: {self._consecutive_blocked} frames bloqueado seguidos, clearance={clearance:.2f} m | "
              f"mapa: [{mapa_diag_str}] | "
              f"bev_fresco[0.4-1.25m]: izq={bev_izq_pct:.0f}% der={bev_der_pct:.0f}%")

        # DIAG 1: Guardar frame RGB y BEV del instante en debug_dir
        if getattr(self, "debug_dir", None):
            try:
                from PIL import Image
                n = getattr(self.stats, "iterations", 0)
                debug_path = Path(self.debug_dir)
                debug_path.mkdir(parents=True, exist_ok=True)
                if rgb is None and hasattr(self, "client") and hasattr(self.client, "front_frame"):
                    try:
                        rgb, _ = self.client.front_frame()
                    except Exception:
                        rgb = None
                if rgb is not None:
                    Image.fromarray(rgb).save(debug_path / f"{n:05d}_rgb.jpg", quality=80)
                np.save(debug_path / f"{n:05d}_bev.npy", bev)
            except Exception as exc:
                print(f"[bridge] no pude escribir debug de regimen cercano: {exc}")

        veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()

        libre_pct, cobertura_pct = self._map_free_and_coverage(
            pose, heading_rel_deg=180.0, radius_m=self.retroceso_max_m)

        # Si la cobertura trasera es baja y no estamos vetados por pendiente/config,
        # consultamos la camara trasera para actualizar el mapa detras del robot
        if self.allow_reverse and not veto_tilt and cobertura_pct < self.retroceso_min_cobertura_pct:
            print("[bridge]   cobertura insuficiente detras, tomando foto de camara trasera para mapear...")
            try:
                rgb_rear, _ = self.client.rear_frame()
                roll_pitch = self.odometry.current_roll_pitch() if self.odometry is not None else None
                # Mirando hacia atras: pitch y roll se invierten respecto a los ejes de camara frontal
                r_rear = -roll_pitch[0] if roll_pitch is not None else None
                p_rear = -roll_pitch[1] if roll_pitch is not None else None
                res_rear = self.perception.process(rgb_rear, roll_rad=r_rear, pitch_rad=p_rear)

                # Pose virtual mirando hacia atras (+pi)
                rear_pose = Pose(pose.x, pose.y, pose.theta + math.pi)
                self.pmap.integrate(res_rear.traversability, res_rear.observed, rear_pose,
                                    self.forward_range, self.side_range, t=time.time())

                libre_pct, cobertura_pct = self._map_free_and_coverage(
                    pose, heading_rel_deg=180.0, radius_m=self.retroceso_max_m)
                print(f"[bridge]   mapa actualizado detras con camara trasera: libre={libre_pct:.0f}% cobertura={cobertura_pct:.0f}%")
            except Exception as e:
                print(f"[bridge]   error usando camara trasera: {e}")

        print(f"[bridge]   mapa detras del robot: libre={libre_pct:.0f}% cobertura={cobertura_pct:.0f}%")

        if not self.allow_reverse:
            print("[bridge]   retroceso desactivado por configuracion (allow_reverse=false), salteo")
        elif veto_tilt:
            print(f"[bridge]   VETO DE RETROCESO POR PENDIENTE: {razon_tilt}, salteo el retroceso")
        elif libre_pct >= self.retroceso_min_libre_pct and cobertura_pct >= self.retroceso_min_cobertura_pct:
            self._retroceder()
        else:
            print("[bridge]   detras no parece seguro (o sin datos suficientes), salteo el retroceso")

        try:
            self._recover_informado(excluir_frente=True)
        except TypeError:
            self._recover_informado()
        _safe_reset_recovery_state(self)

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

        veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()
        if veto_tilt:
            print(f"[bridge]   BLOQUEO DEFENSIVO: aborto retroceso por {razon_tilt}")
            return

        self.stats.retrocesos += 1
        print(f"[bridge]   retrocediendo hasta {self.retroceso_max_m:.2f} m "
              f"en pasos de hasta {self.retroceso_paso_m:.2f} m")
        start = self.odometry.pose
        start_pose = Pose(start.x, start.y, start.theta)
        recorrido = 0.0

        # Durante la maniobra de retroceso, invalidamos cualquier plan cacheado y
        # reseteamos el commit lateral: el comando angular se fija estrictamente en 0.0
        # (recto). El PathFollower no interviene en este tramo.
        self._plan_path_world = None
        self._plan_pose = None
        self._commit_side = 0
        self._commit_until = 0.0

        while recorrido < self.retroceso_max_m and not self._stop_requested:
            falta_m = self.retroceso_max_m - recorrido
            step_m = min(self.retroceso_paso_m, falta_m)
            if step_m < 0.05:
                break
            step_s = step_m / max(abs(self.retroceso_linear), 1e-3)

            # Chequeo continuo de inclinacion durante retroceso
            veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()
            if veto_tilt:
                print(f"[bridge]   inclinacion peligrosa detectada durante retroceso ({razon_tilt}), corto maniobra")
                break

            self.send(DriveCommand(self.retroceso_linear, 0.0, "retroceso (regimen cercano)"))
            t0 = time.time()
            while time.time() - t0 < step_s and not self._stop_requested:
                time.sleep(0.1)
            self.send(DriveCommand(0.0, 0.0, "pausa de retroceso"))

            t_telem = self.client.telemetry()
            pose = self.odometry.update(
                t_telem.raw,
                ekf_heading=getattr(t_telem, "ekf_heading", None),
                ekf_timestamp=getattr(t_telem, "ekf_heading_time", None),
            )

            recorrido = math.hypot(pose.x - start_pose.x, pose.y - start_pose.y)
            try:
                rgb, _ = self.client.front_frame()
                roll_pitch = self.odometry.current_roll_pitch() if self.odometry is not None else None
                r = roll_pitch[0] if roll_pitch is not None else None
                p = roll_pitch[1] if roll_pitch is not None else None
                res = self.perception.process(rgb, roll_rad=r, pitch_rad=p)

                # Integrar foto nueva al mapa para datos frescos
                if self.use_map and self.pmap is not None:
                    self.pmap.integrate(res.traversability, res.observed, pose,
                                        self.forward_range, self.side_range, t=time.time())

                if not self._is_front_blocked(res.traversability):
                    print(f"[bridge]   frente liberado tras retroceder {recorrido:.2f} m")
                    break
            except Exception:
                break

        self.send(DriveCommand(0.0, 0.0, "fin del retroceso"))

    def _evaluar_candidatos_recovery_mapa(self, veto_tilt: bool, razon_tilt: str, excluir_frente: bool = False) -> dict | None:
        """Evalúa los rumbos candidatos en self.recovery_headings_deg usando el mapa
        persistente y la ponderación bilateral hacia la meta (Score = w_clearance * libre + w_goal * align).
        Devuelve el mejor candidato (dict) o None si ninguno es viable.
        """
        assert self.pmap is not None and self.odometry is not None
        pose = self.odometry.pose

        consecutive_recoveries = getattr(self, "_consecutive_empty_recoveries", 0)
        goal_rel_deg = self._get_goal_relative_bearing_deg() if hasattr(self, "_get_goal_relative_bearing_deg") else 0.0

        min_libre = float(getattr(self, "recovery_min_libre_pct", 30.0))
        min_cob = float(getattr(self, "recovery_min_cobertura_pct", 25.0))
        w_goal = float(getattr(self, "recovery_goal_weight", 1.2))
        w_clear = float(getattr(self, "recovery_clearance_weight", 1.0))

        candidatos_evaluados = []

        for h in self.recovery_headings_deg:
            h_flt = float(h)
            is_180 = abs(abs(h_flt) - 180.0) < 1.0

            # Si hay inclinacion peligrosa, vetamos el rumbo 180° (atras) por riesgo de vuelco
            if veto_tilt and is_180:
                print(f"[bridge]   rumbo 180° VETADO por pendiente ({razon_tilt})")
                continue

            # Excluir rumbo 0° si el recovery viene de un frente bloqueado (régimen cercano)
            if excluir_frente and abs(h_flt) < 1e-6:
                print("[bridge]   rumbo 0° OMITIDO (recovery por frente bloqueado)")
                continue

            # Anti-bucle para rumbo 0°: Si ya tuvimos recuperaciones consecutivas por planes vacios
            # sin avance, vetamos 0° para obligar a una rotacion real que cambie la perspectiva
            if consecutive_recoveries >= 2 and abs(h_flt) < 1e-6:
                print("[bridge]   rumbo 0° OMITIDO (reintentos consecutivos de recuperacion sin avance)")
                continue

            libre_pct, cobertura_pct = self._map_free_and_coverage(pose, h_flt, self.heading_search_radius_m)

            # Ponderación hacia la meta:
            diff_deg = abs(wrap_deg(h_flt - goal_rel_deg))
            goal_align = (math.cos(math.radians(diff_deg)) + 1.0) / 2.0
            clear_score = libre_pct / 100.0
            score = (w_clear * clear_score) + (w_goal * goal_align)

            has_clearance = (cobertura_pct >= min_cob and libre_pct >= min_libre)

            print(f"[bridge]   rumbo {h_flt:+.0f} grados: libre={libre_pct:.0f}% cobertura={cobertura_pct:.0f}% "
                  f"(align_meta={goal_align:.2f}, score={score:.2f})")

            candidatos_evaluados.append({
                "heading": h_flt,
                "is_180": is_180,
                "libre_pct": libre_pct,
                "cobertura_pct": cobertura_pct,
                "has_clearance": has_clearance,
                "goal_align": goal_align,
                "score": score,
            })

        # Selección:
        # Prioridad 1: Rumbos frontales / laterales (no 180°) con clearance suficiente
        viables_laterales = [c for c in candidatos_evaluados if not c["is_180"] and c["has_clearance"]]

        if viables_laterales:
            mejor = max(viables_laterales, key=lambda c: (c["score"], c["libre_pct"]))
            print(f"[bridge]   elijo rumbo lateral/frontal {mejor['heading']:+.0f} grados por mapa "
                  f"(libre={mejor['libre_pct']:.0f}%, align_meta={mejor['goal_align']:.2f}, score={mejor['score']:.2f})")
            return mejor

        # Prioridad 2: 180° SOLO como último recurso si ningún rumbo lateral tiene clearance
        candidatos_180 = [c for c in candidatos_evaluados if c["is_180"] and c["has_clearance"]]
        if candidatos_180:
            mejor = candidatos_180[0]
            print(f"[bridge]   ningun rumbo lateral con clearance suficiente; elijo rumbo 180° como ultimo recurso "
                  f"(libre={mejor['libre_pct']:.0f}%)")
            return mejor

        return None

    def _escanear_360(self) -> bool:
        """Ejecuta una maniobra continua de giro de 360° sobre el propio eje,
        integrando observaciones en el mapa persistente para repoblarlo en todas
        las direcciones.

        Comportamiento:
        1. Sentido de giro: Determinado por el bearing relativo al checkpoint objetivo
           (antihorario si bearing < 0, horario si bearing >= 0), asegurando que los
           primeros grados cubran la zona más prometedora.
        2. Sincronización frame/pose crítica: Cada frame capturado se proyecta al BEV
           usando el theta exacto del instante de captura (pose_at(frame_ts)), evitando
           la distorsión angular acumulada por la latencia del pipeline (~283 ms @ 20°/s).
        3. Corte anticipado: Tras superar recovery_startup_latency_s y un giro inicial,
           evalúa continuamente el rumbo frontal (0° relativo). Si encuentra una salida
           claramente despejada (libre >= recovery_scan_early_exit_libre_pct y
           cobertura >= recovery_min_cobertura_pct), corta el giro de inmediato y arranca
           hacia allí con _unstick(), devolviendo True.
        4. Si completa los 360° sin corte anticipado, frena y devuelve False para que
           el invocador reintente la evaluación bilateral de mapa con datos frescos.
        """
        assert self.odometry is not None and self.pmap is not None

        # Veto de inclinación defensivo
        veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()
        if veto_tilt:
            print(f"[bridge]   escaneo 360° VETADO por pendiente ({razon_tilt})")
            return False

        if hasattr(self, "stats") and hasattr(self.stats, "escaneos_360"):
            self.stats.escaneos_360 += 1

        # 1. Sentido de giro hacia la meta (Paso 1)
        goal_rel_deg = self._get_goal_relative_bearing_deg() if hasattr(self, "_get_goal_relative_bearing_deg") else 0.0
        turn_dir = -1.0 if goal_rel_deg < 0.0 else 1.0
        sentido_str = "antihorario (hacia meta izq)" if turn_dir < 0.0 else "horario (hacia meta der)"
        print(f"[bridge] INICIO ESCANEO 360°: sentido {sentido_str} (bearing_meta={goal_rel_deg:+.1f}°)")

        base_speed = float(getattr(self, "recovery_scan_turn_speed", getattr(self, "recovery_turn_speed", 0.75)))
        cmd_ang = self.follower.angular_sign * math.copysign(base_speed, turn_dir)
        cmd_ang = float(np.clip(cmd_ang, -1.0, 1.0))

        startup_lat_s = float(getattr(self, "recovery_startup_latency_s", 2.0))
        deg_per_s = max(float(getattr(self, "recovery_scan_deg_per_s", 20.0)), 1.0)
        early_exit_libre_thresh = float(getattr(self, "recovery_scan_early_exit_libre_pct", 70.0))
        min_cob = float(getattr(self, "recovery_min_cobertura_pct", 25.0))

        target_mag_deg = 360.0
        expected_s = startup_lat_s + (target_mag_deg / deg_per_s)
        timeout_s = float(getattr(self, "recovery_scan_timeout_s", max(25.0, expected_s * 1.5)))

        t_start = time.time()
        girado_real_deg = 0.0
        last_theta = self.odometry.pose.theta

        while not self._stop_requested:
            now = time.time()
            elapsed_s = now - t_start

            # Métricas efectivas de giro (sin contar latencia de arranque) para calibración en campo
            t_efectivo_s = max(0.001, elapsed_s - startup_lat_s) if elapsed_s > startup_lat_s else 0.001
            omega_real_dps = girado_real_deg / t_efectivo_s

            # Chequeo continuo de inclinación
            veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()
            if veto_tilt:
                print(f"[bridge]   inclinacion peligrosa detectada durante escaneo 360° ({razon_tilt}), aborto (girado real: {girado_real_deg:.1f}°, omega_real: {omega_real_dps:.1f}°/s)")
                break

            # Timeout de seguridad
            if elapsed_s >= timeout_s:
                print(f"[bridge]   TIMEOUT de escaneo 360° ({elapsed_s:.1f}s >= {timeout_s:.1f}s, girado real: {girado_real_deg:.1f}°, omega_real: {omega_real_dps:.1f}°/s en {t_efectivo_s:.1f}s efectivos)")
                break

            # Comando angular continuo sostenido (Paso 1)
            progreso_str = f"escaneo 360° [{girado_real_deg:.1f}°/360°] @ {deg_per_s:.0f}°/s (medida={omega_real_dps:.1f}°/s)"
            self.send(DriveCommand(0.0, cmd_ang, progreso_str))

            # Capturar frame y telemetría (Paso 2)
            try:
                rgb, frame_ts = self.client.front_frame()
            except Exception as exc:
                print(f"[bridge]   error capturando frame durante escaneo: {exc}")
                break

            t_telem = self.client.telemetry()
            pose_now = self.odometry.update(
                t_telem.raw,
                ekf_heading=getattr(t_telem, "ekf_heading", None),
                ekf_timestamp=getattr(t_telem, "ekf_heading_time", None),
                now=now,
            )

            # Sincronización crítica de pose (Paso 2):
            # Usar la pose del instante de captura (frame_ts) para que la proyeccion BEV
            # coincida con la orientacion real de la camara, no la retrasada por inferencia.
            if hasattr(self.odometry, "pose_at") and frame_ts > 0.0:
                pose_capture = self.odometry.pose_at(frame_ts)
            else:
                pose_capture = Pose(pose_now.x, pose_now.y, pose_now.theta)

            # Acumular rotación física medida por odometría
            d_th = wrap_rad(pose_now.theta - last_theta)
            last_theta = pose_now.theta
            girado_real_deg += math.degrees(abs(d_th))

            # Procesar percepción BEV con roll y pitch actuales
            roll_pitch = self.odometry.current_roll_pitch(now=now) if hasattr(self.odometry, "current_roll_pitch") else None
            r = roll_pitch[0] if roll_pitch is not None else None
            p = roll_pitch[1] if roll_pitch is not None else None
            res = self.perception.process(rgb, roll_rad=r, pitch_rad=p)

            # Integrar observación en el mapa persistente con pose_capture
            if self.use_map and self.pmap is not None:
                self.pmap.integrate(res.traversability, res.observed, pose_capture,
                                    self.forward_range, self.side_range, t=frame_ts)

            # Chequeo de corte anticipado ante salida claramente buena (Paso 3):
            # Solo tras superar la latencia de arranque del hardware y haber iniciado rotacion real
            if elapsed_s >= startup_lat_s and girado_real_deg >= 15.0:
                # Evaluamos lo que el robot tiene AL FRENTE (rumbo 0° relativo a pose_now)
                libre_pct, cobertura_pct = self._map_free_and_coverage(pose_now, 0.0, self.heading_search_radius_m)
                if cobertura_pct >= min_cob and libre_pct >= early_exit_libre_thresh:
                    print(f"[bridge]   CORTE ANTICIPADO de escaneo 360° tras girar {girado_real_deg:.1f}° en {t_efectivo_s:.1f}s efectivos "
                          f"(omega_real={omega_real_dps:.1f}°/s): frente claramente libre (libre={libre_pct:.1f}% >= {early_exit_libre_thresh:.1f}%, "
                          f"cobertura={cobertura_pct:.1f}%). Arrancando hacia la salida.")
                    self.send(DriveCommand(0.0, 0.0, "corte anticipado de escaneo 360"))
                    self._unstick()
                    return True

            # Condición de fin de vuelta completa (360° alcanzados dentro de tolerancia)
            tol_deg = float(getattr(self, "recovery_turn_tolerance_deg", 15.0))
            if girado_real_deg >= (target_mag_deg - tol_deg) and elapsed_s >= (startup_lat_s * 0.5):
                print(f"[bridge]   escaneo 360° completado por odometria: {girado_real_deg:.1f}° girados en {elapsed_s:.1f}s "
                      f"({t_efectivo_s:.1f}s efectivos de giro, omega_real={omega_real_dps:.1f}°/s)")
                break

        t_efectivo_s = max(0.001, (time.time() - t_start) - startup_lat_s)
        omega_final = girado_real_deg / t_efectivo_s
        print(f"[bridge]   fin de escaneo 360: {girado_real_deg:.1f}° acumulados (omega_real_final={omega_final:.1f}°/s)")
        self.send(DriveCommand(0.0, 0.0, "fin de escaneo 360"))
        return False

    def _recover_informado(self, excluir_frente: bool = False) -> None:
        """Cascada de recuperación informada en 4 niveles:
        (1) Mapa persistente inicial (bilateral ponderado hacia la meta)
        (2) Escaneo 360° continuo con corte anticipado -> reintento de mapa con datos frescos
        (3) VLM (orientación semántica multimodal)
        (4) Barrido ciego condicional (último recurso).
        """
        assert self.pmap is not None and self.odometry is not None

        veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()

        # NIVEL 1: Mapa persistente inicial
        eval_fn = getattr(self, "_evaluar_candidatos_recovery_mapa", None)
        if eval_fn is not None:
            try:
                candidato = eval_fn(veto_tilt, razon_tilt, excluir_frente=excluir_frente)
            except TypeError:
                candidato = eval_fn(veto_tilt, razon_tilt)
        else:
            candidato = Bridge._evaluar_candidatos_recovery_mapa(self, veto_tilt, razon_tilt, excluir_frente=excluir_frente)

        if candidato is not None:
            mejor_heading = candidato["heading"]
            mejor_libre = candidato["libre_pct"]
            print(f"[bridge] [NIVEL 1] elijo rumbo {mejor_heading:+.0f} grados por mapa persistente (libre={mejor_libre:.0f}%, score={candidato['score']:.2f})")
            self.stats.recoveries_por_mapa += 1
            if abs(mejor_heading) < 1e-6:
                self._unstick()
            else:
                self._girar_hacia(mejor_heading)
            return

        print("[bridge] [NIVEL 1] mapa persistente sin rumbo viable (falta cobertura o clearance)")

        # NIVEL 1.5: Escaneo 360° para repoblar mapa persistente (Paso 1 - 5)
        puede_escanear = False
        if getattr(self, "use_recovery_scan", True) and not veto_tilt:
            # Chequeo anti-bucle (Paso 5)
            # Resetear contador si el rover avanzó la distancia mínima requerida
            if self.odometry is not None and self._last_scan_pose is not None:
                disp = math.hypot(self.odometry.pose.x - self._last_scan_pose.x,
                                  self.odometry.pose.y - self._last_scan_pose.y)
                if disp >= getattr(self, "recovery_scan_min_disp_m", 1.0):
                    self._scan_count_at_stuck = 0

            now = time.time()
            max_per_stuck = getattr(self, "recovery_scan_max_per_stuck", 1)
            cooldown_s = getattr(self, "recovery_scan_cooldown_s", 30.0)
            scan_count = getattr(self, "_scan_count_at_stuck", 0)
            last_scan_time = getattr(self, "_last_scan_time", 0.0)

            if scan_count >= max_per_stuck:
                print(f"[bridge] [ESCANEO 360°] OMITIDO: limite alcanzado ({scan_count}/{max_per_stuck}) "
                      f"en este atascamiento. Salteo directo a VLM.")
            elif (now - last_scan_time) < cooldown_s:
                print(f"[bridge] [ESCANEO 360°] OMITIDO: en cooldown ({now - last_scan_time:.1f}s < {cooldown_s:.1f}s). "
                      f"Salteo directo a VLM.")
            else:
                puede_escanear = True
        elif veto_tilt:
            print(f"[bridge] [ESCANEO 360°] VETADO por pendiente ({razon_tilt}), salteo directo a VLM")

        if puede_escanear:
            self._scan_count_at_stuck = getattr(self, "_scan_count_at_stuck", 0) + 1
            self._last_scan_time = time.time()
            if self.odometry is not None:
                self._last_scan_pose = Pose(self.odometry.pose.x, self.odometry.pose.y, self.odometry.pose.theta)

            scan_fn = getattr(self, "_escanear_360", None)
            corte_anticipado = scan_fn() if scan_fn is not None else Bridge._escanear_360(self)
            if corte_anticipado:
                print("[bridge] [ESCANEO 360°] recuperacion exitosa por corte anticipado")
                return

            # Reintento del mapa con datos frescos (Paso 4)
            print("[bridge] [REINTENTO MAPA] evaluando rumbos con mapa persistente repoblado tras escaneo 360°...")
            if eval_fn is not None:
                try:
                    candidato_reintento = eval_fn(veto_tilt, razon_tilt, excluir_frente=excluir_frente)
                except TypeError:
                    candidato_reintento = eval_fn(veto_tilt, razon_tilt)
            else:
                candidato_reintento = Bridge._evaluar_candidatos_recovery_mapa(self, veto_tilt, razon_tilt, excluir_frente=excluir_frente)
            if candidato_reintento is not None:
                mejor_heading = candidato_reintento["heading"]
                mejor_libre = candidato_reintento["libre_pct"]
                print(f"[bridge] [REINTENTO MAPA] EXITOSO: elijo rumbo {mejor_heading:+.0f} grados (libre={mejor_libre:.0f}%, score={candidato_reintento['score']:.2f})")
                self.stats.recoveries_por_mapa += 1
                if abs(mejor_heading) < 1e-6:
                    self._unstick()
                else:
                    self._girar_hacia(mejor_heading)
                return

            print("[bridge] [REINTENTO MAPA] tampoco encontro salida viable tras repoblar mapa. Avanzando a VLM...")

        if self.use_vlm_recovery:
            decision = self._preguntar_vlm()
            if decision is not None:
                # Regla semantica on_road:
                # Si on_road=False y sugiere ir de frente ('adelante'), es una contradiccion directa
                # con la presencia de obstaculos/foso/escaleras enfrente: se descarta.
                if not decision.on_road and decision.heading == "adelante":
                    print(f"[bridge]   VLM sugiere 'adelante' pero on_road=false ({decision.reason}), "
                          "inconsistente: descarto sugerencia frontal")
                    decision = None
                elif not decision.on_road:
                    print(f"[bridge]   ALERTA VLM: rover off-road ({decision.reason}), ejecutando escape hacia '{decision.heading}'")

            if decision is not None:
                heading_deg = {"izquierda": -75.0, "derecha": 75.0,
                               "adelante": 0.0, "atras": 180.0}[decision.heading]

                # Veto de 180° ("atras") por inclinacion
                if veto_tilt and abs(heading_deg - 180.0) < 1.0:
                    print(f"[bridge]   VLM sugiere 'atras' (180°), pero fue VETADO por pendiente ({razon_tilt}). "
                          "Alternativa: recurro a barrido condicional acotado")
                    self._barrido_ciego()
                    return

                print(f"[bridge]   VLM sugiere '{decision.heading}' "
                      f"(confianza {decision.confidence:.2f}): {decision.reason}")
                self.stats.recoveries_por_vlm += 1
                if abs(heading_deg) < 1e-6:
                    self._unstick()
                else:
                    self._girar_hacia(heading_deg)
                return

        self._barrido_ciego()

    def _preguntar_vlm(self):
        now = time.time()
        # Cooldown y limite de reintentos consecutivos para evitar bucle de llamadas caras
        if self._vlm_consecutive_calls >= self.vlm_recovery_max_retries:
            tiempo_espera = now - self._last_vlm_call_time
            if tiempo_espera < self.vlm_recovery_cooldown_s:
                print(f"[bridge]   VLM en cooldown ({tiempo_espera:.1f}s < {self.vlm_recovery_cooldown_s:.1f}s, "
                      f"{self._vlm_consecutive_calls} llamadas consecutivas), salteo llamada a Gemini")
                return None
            else:
                self._vlm_consecutive_calls = 0

        try:
            from .vlm_recovery import ask_recovery_heading
        except Exception as exc:
            print(f"[bridge]   vlm_recovery no disponible ({exc}), sigo sin VLM")
            return None

        print("[bridge]   consultando VLM (Gemini) para orientacion de escape...")
        self._last_vlm_call_time = now
        self._vlm_consecutive_calls += 1

        rgb, _ = self.client.front_frame()
        return ask_recovery_heading(rgb, min_confidence=self.vlm_recovery_min_confidence,
                                    timeout_s=self.vlm_recovery_timeout_s)

    def _girar_hacia(self, heading_rel_deg: float, step_deg: float | None = None) -> None:
        """Gira hacia heading_rel_deg (relativo al rumbo del robot al momento de llamar).

        Opera en LAZO CERRADO monitoreando la rotación física real integrada por la odometría,
        contemplando la latencia de arranque del hardware (1.5 a 2.5 s) antes de evaluar
        criterios de corte.

        Criterios de fin de giro:
        1. Primario (Lazo Cerrado): El giro real medido alcanza el objetivo dentro de
           recovery_turn_tolerance_deg (o >= 80% del ángulo pedido).
        2. Anticipado (Mapa Despejado): El mapa detecta frente libre, pero SOLO si ya transcurrió
           la latencia de arranque y hubo rotación real significativa (>=20° o >=40% del objetivo).
        3. Seguridad / Fallback: Timeout de seguridad y modelo teórico si la odometría no se mueve.
        """
        assert self.odometry is not None
        if abs(heading_rel_deg) < 1e-6:
            return

        # Veto defensivo de giro 180° si la inclinacion supera el umbral
        if abs(abs(heading_rel_deg) - 180.0) < 1.0:
            veto_tilt, razon_tilt = self._is_tilt_too_steep_for_recovery()
            if veto_tilt:
                print(f"[bridge]   BLOQUEO DEFENSIVO: intento de giro 180° cancelado por {razon_tilt}")
                return

        target_mag_deg = abs(heading_rel_deg)
        step_deg = float(self.recovery_step_deg if step_deg is None else step_deg)
        base_speed = float(getattr(self, "recovery_turn_speed", 0.75))
        ang = self.follower.angular_sign * math.copysign(base_speed, heading_rel_deg)
        ang = float(np.clip(ang, -1.0, 1.0))

        startup_lat_s = float(getattr(self, "recovery_startup_latency_s", 0.0))
        deg_per_s = max(self.recovery_deg_per_s, 1.0)
        tol_deg = float(getattr(self, "recovery_turn_tolerance_deg", 15.0))
        target_reached_deg = max(target_mag_deg * 0.80, target_mag_deg - tol_deg)

        # Duración teórica esperada y timeout de seguridad
        expected_s = startup_lat_s + (target_mag_deg / deg_per_s)
        timeout_s = float(getattr(self, "recovery_turn_timeout_s", max(20.0, expected_s * 1.5)))
        paso_s = min(1.0, step_deg / deg_per_s)

        t_start = time.time()
        girado_real_deg = 0.0
        girado_teorico_deg = 0.0
        last_theta = self.odometry.pose.theta
        comandos_enviados = 0

        while not self._stop_requested:
            now = time.time()
            elapsed_s = now - t_start

            # Chequeo de timeout de seguridad
            if elapsed_s >= timeout_s:
                print(f"[bridge]   TIMEOUT de giro ({elapsed_s:.1f}s >= {timeout_s:.1f}s, "
                      f"girado real: {girado_real_deg:.1f}° / obj: {heading_rel_deg:+.0f}°)")
                break

            # Boost dinámico de par si el rover lucha contra fricción lateral en pendiente o pasto alto
            if elapsed_s >= (startup_lat_s + 1.0) and girado_real_deg < (target_mag_deg * 0.25):
                boost = min(0.20, 0.08 * (elapsed_s - (startup_lat_s + 0.5)))
                effective_speed = min(0.95, base_speed + boost)
                ang = self.follower.angular_sign * math.copysign(effective_speed, heading_rel_deg)
                ang = float(np.clip(ang, -1.0, 1.0))

            progreso_str = f"girando hacia {heading_rel_deg:+.0f} grados (regimen cercano) [{girado_real_deg:.1f}°/{target_mag_deg:.0f}°]"
            self.send(DriveCommand(0.0, ang, progreso_str))
            comandos_enviados += 1
            girado_teorico_deg += step_deg

            t0 = time.time()
            while time.time() - t0 < paso_s and not self._stop_requested:
                time.sleep(0.05)

            t_telem = self.client.telemetry()
            pose = self.odometry.update(
                t_telem.raw,
                ekf_heading=getattr(t_telem, "ekf_heading", None),
                ekf_timestamp=getattr(t_telem, "ekf_heading_time", None),
            )

            # Acumular rotación física medida (invariante a wrap [-pi, pi])
            d_th = wrap_rad(pose.theta - last_theta)
            last_theta = pose.theta
            girado_real_deg += math.degrees(abs(d_th))

            # 1. Criterio Lazo Cerrado: Giro físico completado
            if girado_real_deg >= target_reached_deg and elapsed_s >= (startup_lat_s * 0.5):
                print(f"[bridge]   giro completado por odometría: {girado_real_deg:.1f}° girados "
                      f"(objetivo: {target_mag_deg:.0f}°, tol: {tol_deg:.0f}°) en {elapsed_s:.1f}s ({comandos_enviados} cmds)")
                break

            # 2. Criterio Despeje por Mapa: Solo tras superar latencia de arranque Y giro mínimo real
            if elapsed_s >= startup_lat_s and (girado_real_deg >= min(20.0, target_mag_deg * 0.4)):
                libre_pct, cobertura_pct = self._map_free_and_coverage(pose, 0.0, self.heading_search_radius_m)
                if cobertura_pct >= self.recovery_min_cobertura_pct and libre_pct >= self.retroceso_min_libre_pct:
                    print(f"[bridge]   frente libre por mapa tras girar {girado_real_deg:.1f}° reales en {elapsed_s:.1f}s, corto")
                    break

            # 3. Criterio de compatibilidad/fallback si la odometría no se mueve (stubs offline o sensor sin giro)
            if girado_real_deg < 1.0 and girado_teorico_deg >= target_mag_deg and elapsed_s >= expected_s:
                print(f"[bridge]   corte por modelo abierto/fallback ({elapsed_s:.1f}s >= {expected_s:.1f}s, "
                      f"sin rotación de odometría detectada)")
                break

        self.send(DriveCommand(0.0, 0.0, "fin del giro"))

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
        escaneos = getattr(s, "escaneos_360", 0)
        print(f"  recuperaciones:         mapa={s.recoveries_por_mapa}  "
              f"escaneo_360={escaneos}  "
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
            print(f"  correcciones EKF:       {self.odometry.heading_corrections}")
            if self.odometry.last_pitch is not None:
                p_deg = math.degrees(self.odometry.last_pitch)
                r_deg = math.degrees(self.odometry.last_roll) if self.odometry.last_roll is not None else 0.0
                print(f"  inclinacion final:      pitch={p_deg:+.1f}°, roll={r_deg:+.1f}° (blend={self.odometry.last_blend_effective:.2f})")

        if hasattr(self, "governor") and self.governor is not None and self.governor.cfg.enabled:
            print(f"  --- gobernador de velocidad ---")
            print(f"  t_plan_p95:             {self.governor.compute_t_plan_p95():.3f} s")
            v_safe = self.governor.filtered_v_safe
            v_str = f"{v_safe:.2f} m/s" if v_safe is not None else "N/A"
            th_lim = self.governor.filtered_throttle_limit
            th_str = f"{th_lim:.2f}" if th_lim is not None else "N/A"
            print(f"  v_safe / throttle lim:  {v_str} / {th_str}")
            print(f"  recortes de velocidad:  {getattr(s, 'governor_clamps', 0)}")
            print(f"  cortes Stop & Wait:     {getattr(s, 'governor_stops', 0)}")

        if self.ruta is not None:
            rs = self._route_stats
            print(f"  --- ruta (secundaria, checkpoints intermedios) ---")
            print(f"  {self.ruta.descripcion()}  reenganches={self.ruta.reenganches}  "
                  f"terminada={self.ruta.terminada}")
            print(f"  metas: ruta={rs['metas_ruta']}  oficial={rs['metas_oficial']}  "
                  f"ruta ignorada por desvio={rs['ruta_ignorada']}")

        print(f"  errores:                {s.errors}")
        if getattr(self, "log_tilt", False) and getattr(self, "_tilt_csv_path", None):
            print(f"  log de inclinacion (CSV): {self._tilt_csv_path}")
        self._print_heading_diagnosis()

    def _print_heading_diagnosis(self) -> None:
        """Resume el diagnóstico de rumbo al final de la corrida.

        Compara:
        1. Rumbo Activo vs Curso GPS Confiable: Fuente válida para evaluar offset cinemático.
        2. Compás Crudo SDK vs Referencia: Diagnóstico informativo de perturbación magnética
           ambiental (estructuras metálicas), con advertencia explícita de NO aplicar al config.
        """
        print("\n  --- diagnóstico de rumbo ---")

        # 1. Heading Activo vs Curso GPS Confiable
        muestras_gps = self._disagreements_gps
        print("  [1] Rumbo Activo vs Curso GPS (Ground Truth en rectas):")
        if len(muestras_gps) < 5:
            print(f"      muestras útiles:        {len(muestras_gps)} (pocas para concluir; "
                  "hace falta avance rectilíneo continuo para activar curso GPS confiable)")
        else:
            rad = np.radians(np.asarray(muestras_gps, dtype=np.float64))
            media_gps = math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
            r_gps = float(np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
            disp = math.degrees(math.sqrt(-2.0 * math.log(r_gps))) if r_gps > 1e-9 else 180.0
            print(f"      muestras útiles:        {len(muestras_gps)}")
            print(f"      desacuerdo medio:       {media_gps:+.0f} grados")
            print(f"      concentración R:        {r_gps:.2f}  (dispersión ~{disp:.0f} grados)")
            if abs(media_gps) < 10.0:
                print("      -> SIN OFFSET: el rumbo activo y el curso GPS coinciden en promedio."
                      " No hay nada que calibrar.")
            elif r_gps >= 0.7:
                print(f"      -> OFFSET SISTEMÁTICO CINEMÁTICO DETECTADO contra curso GPS.")
                print(f"         navigation.orientation_offset_deg: {media_gps:+.0f}")
                print("         (Alineación cinemática del chasis respecto a trayectoria GNSS)")
            else:
                print("      -> Dispersión transitoria en el track GPS (curvas o baja velocidad).")

        # 2. Compás Crudo del SDK vs Referencia Confiable (Interferencia Magnética Ambiental)
        muestras_mag = self._disagreements_compass
        if len(muestras_mag) >= 5:
            rad_m = np.radians(np.asarray(muestras_mag, dtype=np.float64))
            media_mag = math.degrees(math.atan2(np.mean(np.sin(rad_m)), np.mean(np.cos(rad_m))))
            r_mag = float(np.hypot(np.mean(np.sin(rad_m)), np.mean(np.cos(rad_m))))
            print(f"\n  [2] Compás Crudo SDK vs Referencia (Diagnóstico de Interferencia Magnética):")
            print(f"      muestras registradas:   {len(muestras_mag)}")
            print(f"      desvío compás crudo:    {media_mag:+.0f} grados (R={r_mag:.2f})")
            print("      -> AVISO: Esta discrepancia refleja perturbación magnética local (metal en el sitio).")
            print("         NO aplicar este valor a navigation.orientation_offset_deg: el EKF ya descarta")
            print("         el compás saturado mediante gating y se ancla al giróscopo debiasado y GPS.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--go", action="store_true",
                    help="enviar comandos de verdad (sin esto es simulacro)")
    ap.add_argument("--start-mission", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="cortar despues de N segundos (usalo siempre las primeras veces)")
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--log-tilt", action="store_true",
                    help="activar logging CSV de inclinacion y comandos a ~5 Hz (DIAG 2)")
    ap.add_argument("--route", "--rutas", nargs="+", default=None,
                    help="uno o mas archivos de rutas grabadas (en genie/rutas/), en orden. "
                         "Son SOLO apoyo de navegacion entre checkpoints oficiales: nunca se "
                         "reclaman en el SDK y nunca condicionan el reached del oficial. Sin "
                         "esto se usa route.files del config")
    ap.add_argument("--dashboard-json", default=None,
                    help="donde escribir el overlay de ruta para el mapa del SDK "
                         "(default: earth-rovers-sdk/static/genie_waypoints.json)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _check_placeholders(cfg)

    bridge = Bridge(
        cfg,
        dry_run=not args.go,
        debug_dir=args.debug_dir,
        log_tilt=args.log_tilt,
        route=args.route,
        dashboard_json=args.dashboard_json,
    )

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
