"""Bucle de control: Earth Rover SDK <-> SAM-TP <-> planner BEV de GeNIE.

Por defecto arranca en DRY-RUN: calcula todo e imprime los comandos sin
enviarlos. Para que el robot se mueva de verdad hace falta pasar --go.

    # simulacro, el robot no se mueve
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml

    # de verdad
    python -m genie_rover.bridge --config configs/frodobot_rover.yaml --go \
        --start-mission --max-seconds 120 --debug-dir debug/run1


QUE CAMBIO EN ESTA VERSION (ver la Guia Tecnica UNLP, secciones 03 y 04)
=======================================================================

1) REPLANIFICACION POR DISPARO ESPACIAL, no por frame.
   Antes: cada iteracion llamaba a plan_on_bev() y armaba un comando nuevo.
   A 0.25 m/s y 5 Hz eso son 5 cm de avance por decision, con celdas de 3 cm:
   dos frames seguidos ven casi la misma escena y lo unico que cambia entre
   ellos es RUIDO (jitter del borde de mascara, cabeceo de la suspension).
   Cuando dos rutas tienen costo parecido, el ruido da vuelta la eleccion --
   ese es el titubeo.

   Ahora el bucle esta desacoplado:

       cada frame (5 Hz):  percepcion -> integrar al mapa -> chequeo frontal
       replanificar SOLO si:  avance >= navigation.replan_every_m
                              o al plan le queda < replan_min_remaining_m
                              o el chequeo frontal veto el plan
                              o pasaron > navigation.replan_max_age_s (red de
                                seguridad; tambien es el unico disparador
                                cuando no hay odometria)
       entre replanificaciones: seguir el plan comprometido, reproyectado a
                                la pose actual (_transform_path_xy mas abajo)

   El chequeo de colision frontal SIGUE corriendo a 5 Hz: el compromiso es
   con la ruta, nunca con "no frenar".

   Esto reemplaza el parche de histeresis (_apply_commit) por el primitivo
   correcto, que es compromiso real. _apply_commit se conserva pero ahora
   casi nunca actua: con un plan que dura 1 m no hay de que titubear. Su
   unico rol es el frame en que efectivamente se replanifica.

2) SE SACO EL VLM POR COMPLETO.
   _ask_vlm_heading(), use_vlm_recovery y vlm_recovery_* ya no existen. No
   se importa .vlm_recovery ni se necesita GEMINI_API_KEY. Si el config
   todavia trae use_vlm_recovery: true se avisa una vez y se ignora.

3) LA RECUPERACION USA EL SCORING DE CORREDORES DE rover_traversability.
   En vez de "girar a ciegas" o de preguntarle a un VLM, _recover() ahora:

     a. frena y, si la memoria dice que atras hay lugar, RETROCEDE
        (retroceso_max_m en pasos de retroceso_paso_m). Retroceder es la
        unica accion que reduce la ocupacion angular de un obstaculo
        cercano: a 0.3 m una maceta tapa 60 grados del campo visual y los
        12 caminos candidatos del planner la atraviesan, giremos hacia
        donde giremos, porque la zona ciega gira con el robot.
     b. elige rumbo cruzando DOS fuentes:
          - suggest_command() de rover_traversability.policy sobre la
            mascara de imagen fresca (res.image_traversability): puntua
            corredores verticales y dice cual tiene mas terreno transitable.
            Solo ve lo que entra en el campo visual (~+-46 grados).
          - el PersistentMap, que cubre los 360 grados y ademas conserva la
            observacion buena de cuando el obstaculo estaba a 1.5 m y se
            veia bien. Es la unica fuente valida por debajo de 0.6 m.
        Con cobertura de mapa suficiente manda el mapa; si no, mandan los
        corredores; si no hay ninguno, barrido ciego como antes.
     c. gira hacia ese rumbo con realimentacion de odometria.
     d. bloquea el re-disparo por recovery_block_s para no encadenar
        recuperaciones sobre datos identicos.

4) _unstick() YA NO FUERZA UN AVANCE RECTO.
   Forzar avance estando trabado de frente es exactamente lo peor que se
   puede hacer: en el mejor caso choca suave y el operador interviene. Los
   giros repetidos sin avance ahora entran a la misma rutina de
   recuperacion (retroceso + rumbo informado), que es la respuesta correcta
   al mismo sintoma.

5) ESCALAMIENTO. Recuperaciones consecutivas sin haber avanzado en el medio
   escalan el retroceso (suave -> fuerte, igual que MissionRunner en
   traversability/) y, a la tercera, se pide intervencion en vez de seguir
   golpeando la pared.

Config nuevo/leido por primera vez (todo con default, nada obligatorio):

    navigation:
      replan_every_m: 1.0          # ya estaba declarado, ahora SI se lee
      replan_min_remaining_m: 0.4  # idem
      replan_max_age_s: 2.0        # nuevo: red de seguridad temporal

    safety:
      obstacle_persist_frames: 4   # ya estaba declarado, ahora SI se lee
      retroceso_max_m: 0.6         # ya estaban declarados, ahora SI se leen
      retroceso_paso_m: 0.2
      retroceso_linear: -0.18
      retroceso_min_libre_pct: 55.0
      retroceso_min_cobertura_pct: 30.0
      recovery_block_s: 2.0        # nuevo
      recovery_escalate_after: 2   # nuevo: retroceso fuerte a partir de aca
      recovery_give_up_after: 3    # nuevo: pedir intervencion
      corridors:                   # nuevo, opcional: pisa PolicyConfig
        num_corridors: 9
        drivable_thresh: 0.5
      traversability_path: null    # nuevo, opcional: donde esta el paquete
                                   # rover_traversability si no esta en el
                                   # PYTHONPATH (ej: "../traversability")
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


# --------------------------------------------------------- geometria de caminos
#
# El camino que devuelve el planner esta en el marco del robot EN EL MOMENTO
# DE PLANIFICAR: columnas (x_right, y_forward), x_right positivo = derecha.
# Para poder seguirlo varios frames despues hay que reexpresarlo en el marco
# del robot AHORA. Las dos funciones de abajo hacen ese ida y vuelta pasando
# por el marco mundo de odometry.py (x adelante, y izquierda, theta antihorario
# -- la misma convencion que usa PersistentMap.extract_bev y el codigo de
# frontera en Indoor/mission.py).
#
# Se definen ACA a proposito, y no en navigation.py: asi este archivo se puede
# pegar en el repo sin depender de que navigation.py tenga o no un
# transform_path_xy (el comentario de los yaml lo menciona, pero no esta en
# todas las copias del repo).

def _local_to_world(x_right: np.ndarray, y_forward: np.ndarray,
                    pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(y_forward, dtype=np.float64)
    left = -np.asarray(x_right, dtype=np.float64)
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return (pose.x + c * forward - s * left,
            pose.y + s * forward + c * left)


def _world_to_local(wx: np.ndarray, wy: np.ndarray,
                    pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    dx = np.asarray(wx, dtype=np.float64) - pose.x
    dy = np.asarray(wy, dtype=np.float64) - pose.y
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    forward = c * dx + s * dy
    left = -s * dx + c * dy
    return -left, forward          # (x_right, y_forward)


def _transform_path_xy(path_xy: np.ndarray, pose_from: Pose, pose_to: Pose) -> np.ndarray:
    """Reexpresa un camino del marco de `pose_from` al marco de `pose_to`."""
    p = np.asarray(path_xy, dtype=np.float64)
    wx, wy = _local_to_world(p[:, 0], p[:, 1], pose_from)
    nx, ny = _world_to_local(wx, wy, pose_to)
    return np.stack([nx, ny], axis=1)


def _path_remaining_m(path_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Recorta lo que ya quedo atras y devuelve (camino_util, metros_restantes).

    "Atras" es y_forward <= 0 en el marco actual. Un camino cuyo ultimo punto
    quedo atras esta agotado: devuelve un array vacio y 0.0, y quien llama
    replanifica.
    """
    p = np.asarray(path_xy, dtype=np.float64)
    if p.ndim != 2 or len(p) < 2:
        return p[:0], 0.0
    adelante = np.nonzero(p[:, 1] > 0.0)[0]
    if adelante.size < 2:
        return p[:0], 0.0
    p = p[adelante[0]:]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    # Se suma la distancia del robot al primer punto vivo: si no, un camino
    # que arranca 80 cm adelante parece mas corto de lo que es.
    return p, float(np.linalg.norm(p[0])) + float(seg.sum())


@dataclass
class LoopStats:
    iterations: int = 0
    plans_ok: int = 0
    plans_empty: int = 0
    plans_reused: int = 0      # iteraciones que siguieron el plan comprometido
    blocked: int = 0
    errors: int = 0
    recoveries: int = 0
    retrocesos: int = 0


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

        # ---- replanificacion por disparo espacial (guia, seccion 03) ---------
        self.replan_every_m = float(nav.get("replan_every_m", 1.0))
        self.replan_min_remaining_m = float(nav.get("replan_min_remaining_m", 0.4))
        self.replan_max_age_s = float(nav.get("replan_max_age_s", 2.0))
        self._plan_path: np.ndarray | None = None   # marco del robot al planificar
        self._plan_pose: Pose | None = None
        self._plan_t = 0.0
        self._plan_id = 0
        self._plan_reason = "sin plan"

        # ---- memoria espacial ------------------------------------------------
        # Sin esto el planner ve una foto de 2x2 m que se descarta en cada
        # frame: no recuerda el obstaculo que acaba de salir de cuadro, ni que
        # ya giro buscando salida. Ademas es la UNICA fuente valida en el
        # regimen cercano (< 0.6 m), donde el BEV instantaneo esta
        # estructuralmente degradado por la geometria de proyeccion.
        mem = cfg.get("memory", {})
        self.use_map = bool(mem.get("enabled", True))
        self.odometry = None
        self.pmap = None
        self.plan_forward_m = self.forward_range
        self.plan_side_m = self.side_range
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

        # --- recuperacion informada, sin VLM ---------------------------------
        # Rumbos candidatos, en grados relativos al rumbo actual. 0 = adelante
        # (por si el bloqueo ya se disolvio), 180 = atras. Se evaluan contra el
        # PersistentMap; si no hay memoria util, deciden los corredores de
        # rover_traversability sobre la mascara fresca.
        self.recovery_headings_deg = [
            float(v) for v in safety.get("recovery_headings_deg", [0.0, 90.0, -90.0, 180.0])
        ] or [90.0]
        # Cobertura minima (%) que tiene que tener un rumbo en la memoria para
        # que su puntaje se tome en serio. Por debajo, "no se sabe" != "libre".
        self.recovery_min_cobertura_pct = float(
            safety.get("recovery_min_cobertura_pct", 25.0))
        self.recovery_block_s = float(safety.get("recovery_block_s", 2.0))
        self.recovery_escalate_after = int(safety.get("recovery_escalate_after", 2))
        self.recovery_give_up_after = int(safety.get("recovery_give_up_after", 3))
        self._recovery_block_until = 0.0
        self._recoveries_since_progress = 0

        # --- retroceso (config que estaba declarada y nunca se leia) ----------
        self.retroceso_max_m = float(safety.get("retroceso_max_m", 0.6))
        self.retroceso_paso_m = float(safety.get("retroceso_paso_m", 0.2))
        self.retroceso_linear = float(safety.get("retroceso_linear", -0.18))
        self.retroceso_min_libre_pct = float(safety.get("retroceso_min_libre_pct", 55.0))
        self.retroceso_min_cobertura_pct = float(
            safety.get("retroceso_min_cobertura_pct", 30.0))

        # --- corredores de rover_traversability ------------------------------
        self._policy_cfg = None          # PolicyConfig o None si no se pudo importar
        self._suggest_command = None
        self._corridors_warned = False
        self._load_corridor_policy(safety)

        if safety.get("use_vlm_recovery"):
            print("[bridge] AVISO: safety.use_vlm_recovery esta en true pero esta "
                  "version no usa VLM. Se ignora (sacalo del config).")

        self.stats = LoopStats()
        self._stop_requested = False
        self._checkpoints: list[Checkpoint] = []
        self._latest_scanned = 0
        self._last_frame_ts = 0.0
        self._last_frame_change = time.time()
        self._consecutive_errors = 0
        self._consecutive_empty = 0
        # Deteccion de giro sin avance: el planner puede quedar en un ciclo
        # donde cada giro revela una escena que vuelve a pedir girar.
        self._consecutive_turns = 0
        self._turn_sign_history: list[float] = []
        self.max_consecutive_turns = int(safety.get("max_consecutive_turns", 6))
        # Obstaculo persistente al frente: cuantos frames SEGUIDOS de "front_is
        # _blocked" antes de recuperar. No reaccionar a un bloqueo de un solo
        # frame (alguien que cruzo caminando) pero tampoco quedarse frenado
        # para siempre contra una maceta. Contador propio, separado de
        # max_consecutive_turns: frenar de frente y pivotear son sintomas
        # distintos, con umbrales distintos.
        self.obstacle_persist_frames = int(safety.get("obstacle_persist_frames", 4))
        self._consecutive_blocked = 0

        # Histeresis de lado de esquive. Con replanificacion por distancia esto
        # casi no actua (solo en el frame en que se replanifica), pero se
        # conserva porque sigue siendo correcto ahi.
        self._commit_side = 0          # -1 izquierda, +1 derecha, 0 sin compromiso
        self._commit_until = 0.0
        self.commit_hold_s = float(nav.get("commit_hold_s", 2.0))
        self.commit_min_deg = float(nav.get("commit_min_deg", 8.0))
        self.commit_override_deg = float(nav.get("commit_override_deg", 30.0))

        mission_cfg = cfg.get("mission", {}) or {}
        self._reporter = MissionConsoleReporter(
            target_label="checkpoint",
            enable_color=mission_cfg.get("console_color"))

    # ------------------------------------------------- corredores (traversability)

    def _load_corridor_policy(self, safety: dict) -> None:
        """Importa suggest_command/PolicyConfig del paquete rover_traversability.

        El paquete vive fuera de genie/ (es el hermano traversability/ del
        repo), asi que puede no estar en el PYTHONPATH del proceso. Si el
        import directo falla se prueba con safety.traversability_path (o los
        candidatos habituales relativos al repo). Si tampoco, se sigue sin
        corredores: la recuperacion cae al PersistentMap y al barrido ciego,
        que es exactamente el comportamiento anterior. Nunca es fatal.
        """
        def _try_import():
            from rover_traversability.policy import PolicyConfig, suggest_command
            return PolicyConfig, suggest_command

        try:
            PolicyConfig, suggest_command = _try_import()
        except Exception:
            aqui = Path(__file__).resolve()
            candidatos = []
            extra = safety.get("traversability_path")
            if extra:
                candidatos.append(Path(extra).expanduser())
            # genie/genie_rover/bridge.py -> repo/ -> repo/traversability
            candidatos += [p / "traversability" for p in aqui.parents[:4]]
            for cand in candidatos:
                if not cand.is_dir():
                    continue
                sys.path.insert(0, str(cand))
                try:
                    PolicyConfig, suggest_command = _try_import()
                    break
                except Exception:
                    sys.path.pop(0)
            else:
                print("[bridge] AVISO: no encontre el paquete rover_traversability. "
                      "La recuperacion va a usar solo el mapa persistente. "
                      "Poné safety.traversability_path en el config si lo tenés "
                      "en otro lado.")
                return

        overrides = dict(safety.get("corridors", {}) or {})
        validos = set(PolicyConfig.__dataclass_fields__)
        desconocidos = [k for k in overrides if k not in validos]
        if desconocidos:
            print(f"[bridge] AVISO: safety.corridors ignora {desconocidos} "
                  f"(no son campos de PolicyConfig)")
        overrides = {k: v for k, v in overrides.items() if k in validos}
        # El angular lo pone _turn_towards con turn_speed/angular_sign del
        # config de navegacion, asi que de PolicyConfig solo nos interesa el
        # scoring de corredores, no sus limites de velocidad.
        self._policy_cfg = PolicyConfig(**overrides)
        self._suggest_command = suggest_command

    def _corridor_heading_deg(self, image_mask) -> tuple[float | None, float, str]:
        """Rumbo relativo sugerido por los corredores, su score, y una glosa.

        Devuelve (None, 0.0, motivo) si no hay corredores disponibles o si
        ninguno supera min_corridor_score. El angulo sale del indice del
        corredor elegido mapeado sobre el campo visual horizontal
        (PolicyConfig.hfov_deg), con el signo de la convencion de este
        archivo: positivo = izquierda, igual que recovery_headings_deg.
        """
        if self._suggest_command is None or image_mask is None:
            return None, 0.0, "sin corredores"
        try:
            d = self._suggest_command(np.asarray(image_mask, dtype=np.float32),
                                      self._policy_cfg, goal_offset_deg=None)
        except Exception as exc:
            if not self._corridors_warned:
                self._corridors_warned = True
                print(f"[bridge] AVISO: suggest_command fallo ({exc}); "
                      f"sigo sin corredores el resto de la corrida")
                self._suggest_command = None
            return None, 0.0, "corredores fallaron"

        if d.best_corridor is None or d.best_corridor < 0 or not d.corridor_scores:
            return None, 0.0, f"corredores: {d.reason}"

        n = len(d.corridor_scores)
        score = float(d.corridor_scores[d.best_corridor])
        # Centro del corredor elegido en x normalizado [-1, 1], y de ahi a
        # grados. imagen-derecha = mundo-derecha = angulo NEGATIVO en la
        # convencion de recovery_headings_deg (positivo = izquierda).
        centro_norm = ((d.best_corridor + 0.5) / n) * 2.0 - 1.0
        media_fov = math.radians(self._policy_cfg.hfov_deg / 2.0)
        deg = -math.degrees(math.atan(centro_norm * math.tan(media_fov)))
        return deg, score, (f"corredores: {d.reason}, banda {d.best_corridor + 1}/{n} "
                            f"({score * 100:.0f}%)")

    # ------------------------------------------------------------------ ciclo

    def request_stop(self, *_a) -> None:
        print("\n[bridge] parada solicitada, frenando ...")
        self._stop_requested = True

    def send(self, cmd: DriveCommand, quiet: bool = False) -> None:
        """Envia el comando (o lo simula, en dry-run)."""
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

        if self.uses_gps_checkpoints:
            self.refresh_checkpoints()
            target = self.current_target()
            if target is None:
                print("[bridge] No hay checkpoint pendiente. Voy a navegar solo evitando "
                      "obstaculos, con la meta fija derecho adelante.")
            else:
                print(f"[bridge] Objetivo: checkpoint #{target.sequence} "
                      f"({target.latitude}, {target.longitude})")

        if self.odometry is None:
            print(f"[bridge] sin odometria: replanifico solo por tiempo "
                  f"(cada {self.replan_max_age_s:.1f} s), no por distancia.")
        else:
            print(f"[bridge] replanificacion por distancia: cada "
                  f"{self.replan_every_m:.2f} m, o si al plan le quedan menos de "
                  f"{self.replan_min_remaining_m:.2f} m, o cada "
                  f"{self.replan_max_age_s:.1f} s como red de seguridad.")

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
                    self._invalidate_plan("error en la iteracion")
                    try:
                        self.send(DriveCommand(0.0, 0.0, "freno por error"))
                    except Exception as exc_freno:
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

    # ------------------------------------------------------- gestion del plan

    def _invalidate_plan(self, motivo: str) -> None:
        self._plan_path = None
        self._plan_pose = None
        self._plan_reason = motivo

    def _plan_age_s(self) -> float:
        return time.time() - self._plan_t

    def _advance_since_plan_m(self, pose: Pose | None = None) -> float:
        """Metros recorridos desde que se planifico.

        Se mide contra la pose QUE SE PASA, no contra self.odometry.pose: son
        la misma cosa en el bucle normal, pero atarlo al argumento hace que la
        funcion sea testeable y que no dependa de en que momento del frame se
        la llame.
        """
        if self._plan_pose is None:
            return 0.0
        p = pose if pose is not None else (
            self.odometry.pose if self.odometry is not None else None)
        if p is None:
            return 0.0
        return math.hypot(p.x - self._plan_pose.x, p.y - self._plan_pose.y)

    def _needs_replan(self, pose: Pose | None) -> tuple[bool, str, np.ndarray | None, float]:
        """(hace_falta, motivo, camino_reproyectado, metros_restantes).

        El camino reproyectado solo viene cuando NO hace falta replanificar --
        es justamente el plan comprometido listo para seguir.
        """
        if self._plan_path is None or len(self._plan_path) < 2:
            return True, "sin plan vigente", None, 0.0

        edad = self._plan_age_s()
        if edad > self.replan_max_age_s:
            return True, f"plan viejo ({edad:.1f} s)", None, 0.0

        # Sin odometria no hay forma de medir avance: el unico disparador es la
        # edad, que ya se evaluo arriba. Se sigue el plan tal cual (el error de
        # no reproyectarlo lo absorbe replan_max_age_s siendo corto).
        if pose is None or self._plan_pose is None:
            return False, "sin odometria: sigo el plan por tiempo", self._plan_path, 0.0

        avance = self._advance_since_plan_m(pose)
        if avance >= self.replan_every_m:
            return True, f"avance de {avance:.2f} m", None, 0.0

        camino, restante = _path_remaining_m(
            _transform_path_xy(self._plan_path, self._plan_pose, pose))
        if len(camino) < 2:
            return True, "plan agotado", None, 0.0
        if restante < self.replan_min_remaining_m:
            return True, f"al plan le quedan {restante:.2f} m", None, restante
        return False, f"sigo plan #{self._plan_id} ({restante:.2f} m)", camino, restante

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
                    # Meta nueva: el plan viejo apuntaba al checkpoint anterior.
                    self._invalidate_plan("checkpoint reclamado")
                else:
                    self._reporter.event(
                        "CERCA DEL CHECKPOINT",
                        f"{goal.distance_m:.1f} m, rechazado: {msg}", "yellow")
            goal_desc = (f"cp#{target.sequence} a {goal.distance_m:.0f} m, "
                         f"rel {goal.relative_bearing_deg:+.0f} grados")
        else:
            goal = type("G", (), {"x_right_m": 0.0, "y_forward_m": self.goal_range_m})()
            goal_desc = "derecho adelante (sin meta GPS)"

        # --- percepcion e integracion al mapa: SIEMPRE, a la frecuencia del
        # bucle. Lo que se desacopla es la DECISION, no la percepcion.
        res = self.perception.process(rgb)

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

        bridge_state = "SIN_META" if target is None else (
            "CERCA_CHECKPOINT" if goal.distance_m < self.claim_radius_m * 1.5 else "NAVEGANDO")
        self._reporter.state_change(bridge_state, goal_desc)

        def _row(action: str, state: str = bridge_state) -> None:
            self._reporter.row(iteration=self.stats.iterations, state=state,
                               pose=pose, target_desc=goal_desc, map_cells=map_cells,
                               trav=res.traversability, action=action)

        # --- chequeo de colision frontal: a 5 Hz, SIEMPRE, con la observacion
        # fresca. El compromiso es con la ruta, nunca con "no frenar". Este
        # chequeo tambien VETA el plan vigente: es uno de los disparadores de
        # replanificacion de la guia.
        if front_is_blocked(res.traversability, self.resolution):
            self.stats.blocked += 1
            self._consecutive_blocked += 1
            self._invalidate_plan("veto del chequeo frontal")
            self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"), quiet=True)
            if self._consecutive_blocked >= self.obstacle_persist_frames:
                _row(f"RECUPERACION ({self._consecutive_blocked} frames bloqueado)",
                     state="RECUPERANDO")
                self._recover(res, motivo=f"obstaculo al frente durante "
                                          f"{self._consecutive_blocked} frames")
            else:
                _row(f"frenado: obstaculo al frente "
                     f"({self._consecutive_blocked}/{self.obstacle_persist_frames})",
                     state="OBSTACULO")
            return
        self._consecutive_blocked = 0

        # --- decision: seguir el plan comprometido, o replanificar -----------
        replanificar, motivo, camino_vigente, _restante = self._needs_replan(pose)

        if not replanificar and camino_vigente is not None:
            self.stats.plans_reused += 1
            cmd = self.follower.command(camino_vigente)
            self._track_turn_streak(cmd)
            if self._consecutive_turns >= self.max_consecutive_turns:
                _row(f"RECUPERACION ({self._consecutive_turns} giros seguidos)",
                     state="RECUPERANDO")
                self._recover(res, motivo="giros repetidos sin avanzar")
                return
            self.send(cmd, quiet=True)
            _row(f"{cmd.linear:+.2f}m/s {cmd.angular:+.2f}rad/s  {motivo}")
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
            self._invalidate_plan("el planner no encontro camino")
            if self._consecutive_empty >= self.recovery_after_empty:
                _row(f"RECUPERACION ({self._consecutive_empty} planes vacios seguidos)",
                     state="RECUPERANDO")
                self._recover(res, motivo=f"{self._consecutive_empty} planes vacios")
            else:
                self.send(DriveCommand(0.0, 0.0, "el planner no encontro camino"), quiet=True)
                _row(f"sin camino ({self._consecutive_empty}/{self.recovery_after_empty})",
                     state="SIN_CAMINO")
            return

        self._consecutive_empty = 0
        self.stats.plans_ok += 1
        self._plan_path = np.asarray(path, dtype=np.float64)
        self._plan_pose = pose
        self._plan_t = time.time()
        self._plan_id += 1

        cmd = self.follower.command(path)
        cmd = self._apply_commit(cmd, path)
        self._track_turn_streak(cmd)
        if self._consecutive_turns >= self.max_consecutive_turns:
            _row(f"RECUPERACION ({self._consecutive_turns} giros seguidos)",
                 state="RECUPERANDO")
            self._recover(res, motivo="giros repetidos sin avanzar")
            return

        self.send(cmd, quiet=True)
        _row(f"{cmd.linear:+.2f}m/s {cmd.angular:+.2f}rad/s  plan #{self._plan_id} "
             f"({motivo})")
        self._maybe_dump_debug(rgb, res, plan)

    def _track_turn_streak(self, cmd: DriveCommand) -> None:
        """Cuenta giros en el lugar consecutivos (linear=0, angular!=0).

        Girar en el lugar es la unica accion que no produce informacion nueva
        sobre lo que hay adelante ni avance hacia la meta: es la definicion
        operativa de estar trabado.
        """
        if cmd.linear == 0.0 and cmd.angular != 0.0:
            self._consecutive_turns += 1
            self._turn_sign_history.append(1.0 if cmd.angular > 0 else -1.0)
        else:
            if cmd.linear > 0.0:
                # Hubo avance real: se considera progreso y se reinicia el
                # escalamiento de recuperaciones.
                self._recoveries_since_progress = 0
            self._consecutive_turns = 0
            self._turn_sign_history.clear()

    # --------------------------------------------------------- odometria ciega

    def _track_pose_during_maneuver(self) -> None:
        """Pide telemetria y actualiza odometria durante una maniobra ciega.

        _recover()/_retroceder()/_turn_towards() mueven al robot con send() +
        sleep. Odometry.update() descarta intervalos mayores a max_dt_s (0.5 s),
        asi que sin estas muestras intermedias la pose y el PersistentMap se
        desalinean del mundo real justo en el momento en que mas importa que no
        lo hagan (recuperandose de estar atascado).
        """
        if not self.use_map or self.odometry is None:
            return
        try:
            telem = self.client.telemetry()
            self.odometry.update(telem.raw)
        except Exception as exc:
            print(f"[bridge] no pude actualizar odometria durante la maniobra: {exc}")

    def _apply_commit(self, cmd: DriveCommand, path: np.ndarray) -> DriveCommand:
        """Evita cambiar de lado de esquive a mitad de maniobra.

        Con replanificacion por distancia esto casi no actua: solo corre en el
        frame en que efectivamente se replanifica. Se conserva porque ahi sigue
        siendo correcto, pero el compromiso REAL ahora lo da el plan que dura
        replan_every_m metros, no esta histeresis.
        """
        target = self.follower.lookahead_point(path)
        if target is None:
            return cmd

        error_deg = math.degrees(math.atan2(float(target[0]), float(target[1])))
        now = time.time()

        if abs(error_deg) < self.commit_min_deg:
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

        if abs(error_deg) >= self.commit_override_deg:
            lado = "derecha" if side > 0 else "izquierda"
            print(f"[bridge] cambio de lado justificado ({error_deg:+.0f} grados "
                  f"hacia {lado})")
            self._commit_side = side
            self._commit_until = now + self.commit_hold_s
            return cmd

        return DriveCommand(cmd.linear, 0.0,
                            f"mantengo el rumbo (evito titubeo, {error_deg:+.0f} grados)")

    # ------------------------------------------------- recuperacion informada

    def _score_heading_from_memory(self, heading_rad: float) -> tuple[float, float]:
        """(fraccion_libre, cobertura) de la ventana de memoria en ese rumbo."""
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

    def _pick_recovery_heading(self, res=None) -> tuple[float | None, str]:
        """Elige cuantos grados girar (relativos al rumbo actual) y por que.

        Orden de preferencia:
          1. memoria espacial (PersistentMap): cubre los 360 grados y conserva
             la observacion buena de cuando el obstaculo estaba lejos y se veia
             bien. Es la unica fuente valida en el regimen cercano.
          2. corredores de rover_traversability sobre la mascara de imagen
             fresca: solo ve el campo visual (~+-46 grados), pero es mucho mas
             fino que el mapa dentro de el, y funciona aunque no haya memoria
             (mapa recien arrancado, use_map en false).
          3. barrido ciego de siempre.

        Devuelve (None, motivo) para pedir el barrido ciego.

        Sesgo deliberado contra "adelante" (~0 grados): a _recover() se llega
        JUSTAMENTE porque el rumbo actual no funciona, y el planner ya
        planifica sobre el mapa. "La memoria dice que adelante esta libre" no
        es informacion nueva: elegirlo seria no hacer nada y volver a entrar en
        recuperacion en la proxima iteracion.
        """
        MARGEN_ADELANTE = 0.15   # cuanto mejor tiene que ser "adelante" para ganar
        SCORE_CORREDOR_MIN = 0.45

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
            # El mapa insiste en que adelante esta mucho mejor que cualquier
            # giro. No nos quedamos quietos (eso vuelve a fallar): barrido
            # corto y que el planner reintente con datos frescos.
            return None, (f"la memoria prefiere seguir derecho "
                          f"({mejor_recto_libre * 100:.0f}% libre), hago un barrido corto")

        if memoria_util:
            return (mejor_giro,
                    f"memoria: {mejor_giro:+.0f} grados, {mejor_giro_libre * 100:.0f}% libre "
                    f"({mejor_giro_cob * 100:.0f}% visto)")

        # La memoria no dio una respuesta util (mapa vacio, o poca cobertura):
        # decide el scoring de corredores sobre la mascara fresca.
        mask = getattr(res, "image_traversability", None) if res is not None else None
        corr_deg, corr_score, corr_desc = self._corridor_heading_deg(mask)

        if corr_deg is not None and corr_score >= SCORE_CORREDOR_MIN:
            if abs(corr_deg) < 15.0:
                return None, f"{corr_desc}; el mejor corredor es el central, barrido corto"
            return corr_deg, f"{corr_desc} -> giro {corr_deg:+.0f} grados"

        if corr_deg is not None:
            return None, (f"{corr_desc}: ningun corredor supera "
                          f"{SCORE_CORREDOR_MIN * 100:.0f}%, barrido ciego")

        return None, f"barrido ciego ({corr_desc}, memoria sin cobertura util)"

    def _hay_lugar_atras(self) -> tuple[bool, str]:
        """Consulta a la MEMORIA (no hay camara trasera) si conviene retroceder."""
        if self.pmap is None or self.odometry is None:
            return False, "sin memoria: no retrocedo a ciegas"
        theta = self.odometry.pose.theta
        libre, cob = self._score_heading_from_memory(theta + math.pi)
        if cob * 100.0 < self.retroceso_min_cobertura_pct:
            return False, (f"atras solo {cob * 100:.0f}% visto "
                           f"(<{self.retroceso_min_cobertura_pct:.0f}%): no retrocedo")
        if libre * 100.0 < self.retroceso_min_libre_pct:
            return False, (f"atras solo {libre * 100:.0f}% libre "
                           f"(<{self.retroceso_min_libre_pct:.0f}%): no retrocedo")
        return True, f"atras {libre * 100:.0f}% libre ({cob * 100:.0f}% visto)"

    def _retroceder(self, metros: float) -> float:
        """Retrocede hasta `metros` con realimentacion de odometria.

        Devuelve los metros efectivamente retrocedidos (0.0 en dry-run sin
        odometria). Es la unica accion que reduce la ocupacion angular de un
        obstaculo cercano: girar no lo aleja, porque la zona ciega gira con el
        robot.

        OJO: retroceso_linear negativo = reversa. Ese signo NO esta verificado
        empiricamente en este repo (no hay un tools/ equivalente a
        check_angular_sign.py para reversa). Confirmalo a mano la primera vez,
        con espacio de sobra detras del robot.
        """
        if metros <= 0.0:
            return 0.0
        self.stats.retrocesos += 1
        p0 = self.odometry.pose if self.odometry is not None else None
        # Tope de tiempo: metros / |velocidad| con margen, acotado para que una
        # rueda patinando no deje al robot en reversa indefinidamente.
        v = max(abs(self.retroceso_linear), 0.05)
        tope_s = min(10.0, (metros / v) * 2.5 + 1.0)

        self.send(DriveCommand(self.retroceso_linear, 0.0,
                               f"retroceso de {metros:.2f} m"))
        t0 = time.time()
        recorrido = 0.0
        while time.time() - t0 < tope_s and not self._stop_requested:
            time.sleep(0.15)
            self._track_pose_during_maneuver()
            if p0 is not None and self.odometry is not None:
                p = self.odometry.pose
                recorrido = math.hypot(p.x - p0.x, p.y - p0.y)
                if recorrido >= metros:
                    break
        self.send(DriveCommand(0.0, 0.0, "fin del retroceso"))
        self._track_pose_during_maneuver()
        return recorrido

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

    def _recover(self, res=None, motivo: str = "") -> None:
        """Recuperacion: retroceder si se puede, y girar al rumbo con mas lugar.

        Secuencia (guia tecnica, seccion 04 -- regimen cercano):

          1. frenar;
          2. retroceder (si la memoria dice que atras hay lugar). Escala a un
             retroceso mas largo si esta es la 2da+ recuperacion sin haber
             avanzado en el medio -- mismo patron que _iniciar_retroceso() de
             traversability/rover_traversability/mission.py;
          3. elegir rumbo: memoria -> corredores de traversability -> ciego;
          4. girar hacia ahi con realimentacion de odometria;
          5. bloquear el re-disparo por recovery_block_s.

        A la recovery_give_up_after-esima recuperacion consecutiva sin avanzar,
        pide intervencion en vez de seguir golpeando la pared: cuesta MPI, pero
        cuesta mucho menos que un stall.
        """
        if time.time() < self._recovery_block_until:
            return

        self.stats.recoveries += 1
        self._recoveries_since_progress += 1
        self._invalidate_plan("recuperacion")

        self.send(DriveCommand(0.0, 0.0, "freno para decidir la recuperacion"), quiet=True)
        self._track_pose_during_maneuver()

        if self._recoveries_since_progress >= self.recovery_give_up_after:
            self._reporter.event(
                "INTERVENCION",
                f"{self._recoveries_since_progress} recuperaciones seguidas sin "
                f"avanzar ({motivo}). El robot no se esta destrabando solo: "
                f"conviene intervenir a mano.", "red")

        # --- 1) retroceso -----------------------------------------------------
        puede, por_que = self._hay_lugar_atras()
        if puede:
            objetivo = self.retroceso_paso_m
            if self._recoveries_since_progress >= self.recovery_escalate_after:
                objetivo = self.retroceso_max_m     # escalamiento suave -> fuerte
            objetivo = min(objetivo, self.retroceso_max_m)
            self._reporter.event(
                "RETROCESO",
                f"{objetivo:.2f} m — {por_que} (recuperacion "
                f"#{self._recoveries_since_progress})", "magenta")
            hecho = self._retroceder(objetivo)
            if self.odometry is not None:
                self._reporter.event("RETROCESO", f"retrocedi {hecho:.2f} m")
        else:
            self._reporter.event("RETROCESO", f"omitido — {por_que}", "yellow")

        # --- 2) rumbo ---------------------------------------------------------
        delta_deg, por_que_rumbo = self._pick_recovery_heading(res)
        self._reporter.event("RECUPERACION",
                             f"{motivo or 'buscando salida'} — {por_que_rumbo}", "magenta")

        if delta_deg is None:
            self._recover_blind()
        else:
            self._turn_towards(delta_deg)

        self.heading_est.reset_track()  # el track GPS previo ya no dice el rumbo
        self._consecutive_empty = 0
        self._consecutive_turns = 0
        self._consecutive_blocked = 0
        self._turn_sign_history.clear()
        self._recovery_block_until = time.time() + self.recovery_block_s

    def _recover_blind(self) -> None:
        """El barrido de siempre: girar un tiempo fijo hacia el lado del follower."""
        ang = float(np.clip(self.follower.angular_sign * self.follower.turn_speed,
                            -self.follower.max_angular, self.follower.max_angular))
        self.send(DriveCommand(0.0, ang, "barrido de recuperacion"))

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
        print(f"  replanificaciones:      {self._plan_id}")
        print(f"  planes exitosos:        {s.plans_ok}")
        print(f"  iters siguiendo plan:   {s.plans_reused}")
        print(f"  planes vacios:          {s.plans_empty}")
        print(f"  frenadas por obstaculo: {s.blocked}")
        print(f"  recuperaciones:         {s.recoveries}")
        print(f"  retrocesos:             {s.retrocesos}")
        if self.pmap is not None and self.odometry is not None:
            st = self.pmap.stats()
            p = self.odometry.pose
            dist = self.odometry.distance_travelled
            print(f"  --- memoria espacial ---")
            print(f"  celdas del mapa:        {st['celdas_vistas']}")
            print(f"  recentrados:            {st['recentrados']}")
            print(f"  pose final:             ({p.x:+.2f}, {p.y:+.2f}) "
                  f"{math.degrees(p.theta):+.0f} grados")
            print(f"  distancia recorrida:    {dist:.2f} m")
            print(f"  correcciones GPS:       {self.odometry.gps_corrections}")
            # La metrica que la guia pide construir antes que cualquier otra
            # cosa: cuantos metros hace el rover entre intervenciones. Aca no
            # se puede contar la intervencion humana, pero si las
            # recuperaciones, que son su proxy mas cercano.
            if s.recoveries > 0:
                print(f"  metros por recuperacion: {dist / s.recoveries:.2f} m")
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
    ap.add_argument("--replan-every-m", type=float, default=None,
                    help="pisa navigation.replan_every_m del config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _check_placeholders(cfg)
    if args.replan_every_m is not None:
        cfg.setdefault("navigation", {})["replan_every_m"] = args.replan_every_m

    bridge = Bridge(cfg, dry_run=not args.go, debug_dir=args.debug_dir)

    if args.start_mission:
        print("[bridge] iniciando mision ...")
        print(f"   {bridge.client.start_mission()}")

    if args.go:
        print("\n" + "=" * 62)
        print("  MODO REAL: el rover se va a mover. Ctrl-C frena.")
        print("  Tene el robot a la vista y espacio libre alrededor.")
        print("  Esta version RETROCEDE durante la recuperacion: dejale")
        print("  espacio libre TAMBIEN detras.")
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
