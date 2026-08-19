"""Simple GPS-checkpoint mission runner: traversability + goal bearing.

The loop each step:
1. Read telemetry (lat/lon/speed/orientation) and the front frame.
2. Estimate heading via GPS course-over-ground (see geo.py for why the
   magnetometer is not trusted).
3. Compute the bearing offset AND distance to the next checkpoint.
4. Always run SAM-TP first — traversability never gets skipped, even when
   the goal is out of the camera's field of view. The mask feeds
   ``suggest_command``, whose goal bias is soft (``PolicyConfig.
   goal_bias_floor``): traversability dominates, the goal only breaks ties.
5. The one case that overrides that soft bias with an aggressive turn is
   "goal out of view AND close" (``align_urgent_m``): far away, a few
   degrees of heading error cost nothing, so there is no reason to spin
   toward a distant point blind to what is underfoot — just drive the safest
   opening and let the bearing converge naturally as ground truth updates.
   Close to the checkpoint, missing it because the goal fell outside the
   frame *does* matter, so an arc-turn is worth it — but only after
   confirming (same mask, no goal bias) that what is directly ahead isn't
   blocked; a checkpoint 3 m to the side is not a reason to drive through
   whatever is in front right now.
6. Within ``arrive_attempt_m`` of the checkpoint, try POST /checkpoint-reached;
   the backend enforces the real radius and rejects with 400 while too far —
   that is the designed retry loop, keep driving.
7. Getting stuck is expected: ``suggest_command`` picks a corridor from a
   single noisy frame with no memory, so two similar-scoring corridors can
   swap places from one frame to the next and the rover spins in place
   without ever completing the turn. Two mitigations, both stateful at the
   MissionRunner level (``suggest_command`` itself stays a pure function):
   - Corridor hysteresis: the previously-chosen corridor gets a small score
     bonus (``PolicyConfig.corridor_hysteresis``) via
     ``suggest_command(..., prev_best_corridor=...)``, so it keeps winning
     close calls instead of flip-flopping.
   - Stuck escalation: if ``stuck_frames_backup`` consecutive steps produce
     no forward motion, back up blindly (no rear camera, same caveat as the
     GeNIE bridge) for a short stretch and retry from a different spot/angle.
     If that still doesn't unstick it, the next backup is longer and
     stronger (``backup_linear_strong``/``backup_frames_strong``) rather than
     repeating the same short nudge forever.
8. Right after a backup finishes, the next step is NOT a normal ``_decide()``
   call — it's ``_realinear()``: a fresh, goal-agnostic look (no checkpoint
   bias at all) at wherever has the most drivable space, so the rover
   commits to open ground before the goal starts pulling it again. If that
   first look still finds nothing usable, it "cabecea" (peeks): a blind turn
   to look left, then — if that's no better — a blind turn to look right,
   each followed by a fresh prediction, comparing which side actually has
   more room instead of just reacting to whatever's dead ahead. If NEITHER
   side has anything usable either, that's not a heading problem anymore —
   it escalates to another (stronger) backup instead of continuing to spin
   through headings that have already been checked.

This is deliberately the *simple* mission solution. The roadmap upgrade path
(docs/ROADMAP.md) replaces step 5 with BEV projection + the vendored
genie_path_planner.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from math import degrees
from typing import Callable, Optional

from .client import RoverClient
from .geo import HeadingEstimator, gps_bearing_and_distance, wrap_angle_deg
from .policy import CommandDecision, PolicyConfig, suggest_command

log = logging.getLogger(__name__)

import logging

class SoloMissionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "mission" or record.name.endswith(".mission")

logging.basicConfig(
    level=logging.INFO,   # usá logging.DEBUG si querés más detalle
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger().handlers[0].addFilter(SoloMissionFilter())


@dataclass
class Checkpoint:
    id: int
    sequence: int
    latitude: float
    longitude: float


@dataclass
class MissionResult:
    completed: bool
    checkpoints_reached: int
    steps: int
    reason: str
    history: list = field(default_factory=list)


def _parse_checkpoints(body: dict) -> tuple[list[Checkpoint], int | None]:
    """Parse /checkpoints-list. Lat/lon arrive as STRINGS — cast them."""
    raw = body.get("checkpoints_list") or []
    cps = [
        Checkpoint(
            id=int(c.get("id", i)),
            sequence=int(c.get("sequence", i + 1)),
            latitude=float(c["latitude"]),
            longitude=float(c["longitude"]),
        )
        for i, c in enumerate(raw)
    ]
    cps.sort(key=lambda c: c.sequence)
    latest = body.get("latest_scanned_checkpoint")
    return cps, (int(latest) if latest is not None else None)


class MissionRunner:
    def __init__(
        self,
        client: RoverClient | None = None,
        predictor=None,
        policy: PolicyConfig | None = None,
        arrive_attempt_m: float = 14.0,
        interval_s: float = 0.5,
        turn_in_place_deg: float = 70.0,
        arc_turn_linear: float = 0.15,
        align_urgent_m: float = 20.0,
        stuck_frames_backup: int = 4,
        backup_linear_soft: float = -0.3,
        backup_frames_soft: int = 4,
        backup_linear_strong: float = -0.5,
        backup_frames_strong: int = 7,
        cabeceo_turn_frames: int = 3,
        max_steps: int | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> None:
        self.client = client or RoverClient()
        if predictor is None:
            from .predictor import TraversabilityPredictor

            predictor = TraversabilityPredictor()
        self.predictor = predictor
        self.policy = policy or PolicyConfig()
        self.arrive_attempt_m = float(arrive_attempt_m)
        self.interval_s = float(interval_s)
        self.turn_in_place_deg = float(turn_in_place_deg)
        self.arc_turn_linear = float(arc_turn_linear)
        # Distancia por debajo de la cual vale la pena forzar el arc-turn
        # agresivo hacia un checkpoint fuera de campo de vision. Mas lejos
        # que esto, unos grados de error de rumbo no cuestan nada — prioriza
        # terreno seguro y deja que el rumbo converja solo. Por defecto un
        # poco mas alla de arrive_attempt_m (14 m): en ese rango ya vale la
        # pena alinear bien para no pasar de largo el checkpoint.
        self.align_urgent_m = float(align_urgent_m)
        # Cuantos steps SEGUIDOS sin lograr "forward" (bloqueado o girando en
        # el lugar sin resultado) antes de retroceder a ciegas para reintentar
        # desde otro angulo/posicion. Sin memoria ni camara trasera aca (a
        # diferencia del bridge de GeNIE), asi que el primer retroceso es
        # corto y lento; si vuelve a trabarse sin haber logrado avanzar en el
        # medio, el siguiente retroceso escala a mas fuerte/largo en vez de
        # repetir el mismo empujoncito indefinidamente.
        self.stuck_frames_backup = int(stuck_frames_backup)
        self.backup_linear_soft = float(backup_linear_soft)
        self.backup_frames_soft = int(backup_frames_soft)
        self.backup_linear_strong = float(backup_linear_strong)
        self.backup_frames_strong = int(backup_frames_strong)
        # Cuantos steps de giro ciego dura cada "mirada" del cabeceo (a la
        # velocidad angular maxima de policy). No hace falta odometria para
        # volver a un rumbo exacto: si un costado resulta prometedor, el
        # cabeceo simplemente sigue girando hacia ahi en vez de "volver" a
        # ningun lado.
        self.cabeceo_turn_frames = int(cabeceo_turn_frames)
        self.max_steps = max_steps
        self.on_step = on_step
        self.heading = HeadingEstimator()
        # Estado de atasco: corredor elegido el step anterior (para la
        # histeresis de suggest_command), racha de steps sin avanzar, y
        # cuantos frames de retroceso quedan por enviar / a que velocidad.
        self._prev_best_corridor: int | None = None
        self._non_forward_streak = 0
        self._backup_frames_left = 0
        self._backup_linear_active = 0.0
        self._consecutive_backups = 0
        # Tras un retroceso: hay que realinear antes de retomar el flujo
        # normal (ver _realinear). _cabeceo_stage no-None mientras esta a
        # mitad de girar/mirar hacia un costado.
        self._pending_realign = False
        self._cabeceo_stage: str | None = None
        self._cabeceo_turn_sign = 0.0
        self._cabeceo_frames_left = 0
        self._cabeceo_score_izq: float | None = None

    # ------------------------------------------------------------------ public

    def run(self) -> MissionResult:
        """Drive checkpoints until the mission completes. Always stops the rover."""
        try:
            return self._run_inner()
        finally:
            self.client.stop()

    # ----------------------------------------------------------------- internal

    def _next_checkpoint(self) -> tuple[Optional[Checkpoint], int]:
        body = self.client.get_checkpoints_list()
        if not body:
            return None, 0
        cps, latest = _parse_checkpoints(body)
        done = latest or 0
        for cp in cps:
            if cp.sequence > done:
                return cp, done
        return None, done

    def _run_inner(self) -> MissionResult:
        target, done_count = self._next_checkpoint()
        if target is None:
            return MissionResult(
                completed=done_count > 0,
                checkpoints_reached=done_count,
                steps=0,
                reason="no pending checkpoints (mission not started, or already complete)",
            )
        log.info("next checkpoint: seq %s at (%s, %s)",
                 target.sequence, target.latitude, target.longitude)

        steps = 0
        reached = done_count
        history: list = []

        while True:
            if self.max_steps is not None and steps >= self.max_steps:
                return MissionResult(False, reached, steps, "max_steps reached", history)
            steps += 1
            t_start = time.perf_counter()

            data = self.client.get_data() or {}
            lat, lon = data.get("latitude"), data.get("longitude")
            heading_deg = self.heading.update(
                lat, lon, data.get("speed"), data.get("orientation")
            )

            goal_offset_deg: float | None = None
            distance_m: float | None = None
            if lat is not None and lon is not None:
                bearing_rad, distance_m = gps_bearing_and_distance(
                    float(lat), float(lon), target.latitude, target.longitude
                )
                if heading_deg is not None:
                    goal_offset_deg = wrap_angle_deg(degrees(bearing_rad) - heading_deg)


            # Arrival attempt — backend enforces the true radius.
            if distance_m is not None and distance_m <= self.arrive_attempt_m:
                log.warning("🎯 [INTENTO REACHED] Distancia de %.2fm <= umbral (%.2fm). Enviando POST /checkpoint-reached...", distance_m, self.arrive_attempt_m)
                res = self.client.checkpoint_reached()
                if res.accepted:
                    reached += 1
                    body = res.body or {}
                    log.warning("✅ [REACHED ACEPTADO] ¡Checkpoint alcanzado con éxito! Respuesta: %s", body)

                    if body.get("mission_completed"):
                        return MissionResult(True, reached, steps, "mission completed", history)
                    
                    # 1. Guardamos el número del checkpoint que acabamos de completar para el log
                    completed_seq = target.sequence if target else "desconocido"
                    log.warning(f"¡Checkpoint {completed_seq} confirmado!") # <--- Tu log en el lugar correcto

                    # 2. Buscamos el siguiente checkpoint
                    target, _ = self._next_checkpoint()
                    
                    # 3. Verificamos si ya no quedan objetivos
                    if target is None:
                        return MissionResult(True, reached, steps, "all checkpoints done", history)
                    
                    log.warning("🔄 [CAMBIO] Siguiente objetivo asignado: seq %s", target.sequence)
                    continue
                else:
                    log.warning("⏳ [REACHED RECHAZADO] El servidor indicó que aún está fuera del radio real. Siguiendo..." )

            if self._backup_frames_left > 0:
                nivel = "fuerte" if self._consecutive_backups > 1 else "suave"
                decision = CommandDecision(
                    self._backup_linear_active, 0.0, False,
                    f"retrocediendo_atascado_{nivel}",
                )
                self._backup_frames_left -= 1
                if self._backup_frames_left == 0:
                    self._pending_realign = True  # no retomar _decide() todavia: realinear primero
            elif self._cabeceo_frames_left > 0:
                lado = "izq" if self._cabeceo_turn_sign > 0 else "der"
                decision = CommandDecision(0.0, self._cabeceo_turn_sign, False, f"cabeceo_girando_{lado}")
                self._cabeceo_frames_left -= 1
            elif self._pending_realign:
                decision = self._realinear()
            else:
                decision = self._decide(goal_offset_deg, distance_m)
                if decision.reason == "forward":
                    self._non_forward_streak = 0
                    self._consecutive_backups = 0
                else:
                    self._non_forward_streak += 1
                    if self._non_forward_streak >= self.stuck_frames_backup:
                        self._non_forward_streak = 0
                        decision = self._iniciar_retroceso("atascado sin avanzar")

            if decision.stop:
                self.client.stop()
            else:
                self.client.send_command(decision.linear, decision.angular)

            step_info = {
                "step": steps,
                "distance_m": distance_m,
                "heading_deg": heading_deg,
                "goal_offset_deg": goal_offset_deg,
                "decision": decision,
                "target_checkpoint": target,
            }
            history.append(step_info)
            if self.on_step is not None:
                self.on_step(step_info)

            elapsed = time.perf_counter() - t_start
            if elapsed < self.interval_s:
                time.sleep(self.interval_s - elapsed)

    def _predict_mask(self):
        """Frame + SAM-TP, o None si no se pudo (frame faltante / prediccion
        fallida) junto con el CommandDecision de stop correspondiente."""
        frame = self.client.get_front_frame()
        if frame is None:
            return None, CommandDecision(0.0, 0.0, True, "no_frame")
        try:
            result = self.predictor.predict(frame)
        except Exception as exc:
            log.error("prediction failed: %s", exc)
            return None, CommandDecision(0.0, 0.0, True, "predict_error")
        return result, None

    def _iniciar_retroceso(self, motivo: str) -> CommandDecision:
        """Arranca (o escala) un retroceso ciego: la primera vez corto y
        lento; si hace falta de nuevo sin haber logrado avanzar en el medio,
        mas fuerte y largo en vez de repetir el mismo empujoncito para
        siempre. Llamado tanto por la racha de steps sin avanzar como por el
        cabeceo cuando no encuentra espacio a ningun lado."""
        self._consecutive_backups += 1
        if self._consecutive_backups == 1:
            self._backup_linear_active = self.backup_linear_soft
            self._backup_frames_left = self.backup_frames_soft
            reason = "retrocediendo_atascado_suave"
        else:
            self._backup_linear_active = self.backup_linear_strong
            self._backup_frames_left = self.backup_frames_strong
            reason = "retrocediendo_atascado_fuerte"
        self._prev_best_corridor = None
        self._cabeceo_stage = None
        log.warning(
            "🔙 [RETROCESO] %s -> nivel %d (%d steps a %.2f)",
            motivo, self._consecutive_backups, self._backup_frames_left, self._backup_linear_active,
        )
        decision = CommandDecision(self._backup_linear_active, 0.0, False, reason)
        self._backup_frames_left -= 1  # este step ya cuenta como el primero
        if self._backup_frames_left == 0:
            self._pending_realign = True
        return decision

    def _realinear(self) -> CommandDecision:
        """Tras un retroceso, antes de retomar el flujo normal de _decide()
        (que podria volver a entrar de a poco en un turning_to_corridor),
        evalua DESDE CERO y SIN sesgo de meta a donde hay mas espacio para
        circular. Si de frente no hay nada aprovechable, cabecea: gira a
        mirar a la izquierda y, si tampoco, a la derecha, comparando cual
        costado tiene realmente mas lugar en vez de reaccionar solo a lo que
        hay justo adelante. Si ningun lado sirve, no es un problema de
        angulo — escala a otro retroceso en lugar de seguir girando por
        rumbos que ya se comprobaron vacios."""
        result, err = self._predict_mask()
        if err is not None:
            return err
        chequeo = suggest_command(result.mask, self.policy, goal_offset_deg=None)

        if self._cabeceo_stage is None:
            if chequeo.reason != "blocked":
                self._pending_realign = False
                self._consecutive_backups = 0
                self._prev_best_corridor = chequeo.best_corridor if chequeo.best_corridor >= 0 else None
                log.info("↔️ [REALINEANDO] hay espacio (%s) tras el retroceso, sin cabecear", chequeo.reason)
                return CommandDecision(chequeo.linear, chequeo.angular, chequeo.stop,
                                        "realineando", chequeo.corridor_scores, chequeo.best_corridor)
            log.info("👀 [CABECEO] nada aprovechable de frente, girando a mirar a la izquierda")
            self._cabeceo_stage = "mirando_izq"
            self._cabeceo_turn_sign = self.policy.max_angular
            self._cabeceo_frames_left = self.cabeceo_turn_frames
            self._cabeceo_frames_left -= 1  # este step ya cuenta como el primer giro
            return CommandDecision(0.0, self._cabeceo_turn_sign, False, "cabeceo_girando_izq")

        if self._cabeceo_stage == "mirando_izq":
            self._cabeceo_score_izq = float(max(chequeo.corridor_scores)) if chequeo.corridor_scores else 0.0
            if chequeo.reason != "blocked":
                self._pending_realign = False
                self._cabeceo_stage = None
                self._consecutive_backups = 0
                self._prev_best_corridor = chequeo.best_corridor if chequeo.best_corridor >= 0 else None
                log.info("👀 [CABECEO] a la izquierda hay lugar, alineando ahi")
                return CommandDecision(chequeo.linear, chequeo.angular, chequeo.stop,
                                        "cabeceo_izq_ok", chequeo.corridor_scores, chequeo.best_corridor)
            log.info("👀 [CABECEO] tampoco a la izquierda (max %.2f), girando a mirar a la derecha",
                     self._cabeceo_score_izq)
            self._cabeceo_stage = "mirando_der"
            self._cabeceo_turn_sign = -self.policy.max_angular
            self._cabeceo_frames_left = self.cabeceo_turn_frames * 2  # cruza el centro
            self._cabeceo_frames_left -= 1
            return CommandDecision(0.0, self._cabeceo_turn_sign, False, "cabeceo_girando_der")

        # self._cabeceo_stage == "mirando_der"
        score_der = float(max(chequeo.corridor_scores)) if chequeo.corridor_scores else 0.0
        self._cabeceo_stage = None
        if chequeo.reason != "blocked":
            self._pending_realign = False
            self._consecutive_backups = 0
            self._prev_best_corridor = chequeo.best_corridor if chequeo.best_corridor >= 0 else None
            log.info("👀 [CABECEO] a la derecha hay lugar (max %.2f), alineando ahi", score_der)
            return CommandDecision(chequeo.linear, chequeo.angular, chequeo.stop,
                                    "cabeceo_der_ok", chequeo.corridor_scores, chequeo.best_corridor)
        # Ni de frente, ni a la izquierda, ni a la derecha: no es cuestion de
        # angulo, hace falta mas distancia.
        return self._iniciar_retroceso(
            f"cabeceo sin espacio (izq {self._cabeceo_score_izq:.2f}, der {score_der:.2f})"
        )

    def _decide(self, goal_offset_deg: float | None, distance_m: float | None) -> CommandDecision:
        # Traversability SIEMPRE corre primero. Nunca se decide un movimiento
        # (ni siquiera "girar hacia la meta") sin haber mirado la mascara de
        # SAM-TP: un checkpoint lejano o fuera de campo de vision no es motivo
        # para avanzar o girar a ciegas sobre lo que sea que haya adelante.
        result, err = self._predict_mask()
        if err is not None:
            return err

        fuera_de_vista = goal_offset_deg is not None and abs(goal_offset_deg) > self.turn_in_place_deg
        urgente = distance_m is not None and distance_m <= self.align_urgent_m

        if fuera_de_vista and urgente:
            # Meta fuera de campo de vision pero CERCA: vale la pena forzar
            # el giro para no pasarla de largo (motion, no spin puro, para
            # que el GPS-COG siga dando heading). Antes de avanzar, un
            # chequeo de transitabilidad SIN sesgo de meta: si lo que hay
            # adelante esta bloqueado, un checkpoint a upa no es excusa para
            # meterse ahi.
            chequeo = suggest_command(result.mask, self.policy, goal_offset_deg=None)
            if chequeo.stop:
                return CommandDecision(0.0, 0.0, True, "goal_fuera_de_vista_bloqueado")
            turn = -self.policy.max_angular if goal_offset_deg > 0 else self.policy.max_angular
            self._prev_best_corridor = None  # ruta forzada, no corredor "elegido" que fijar
            return CommandDecision(
                linear=self.arc_turn_linear,
                angular=turn,
                stop=False,
                reason="arc_turn_to_goal",
            )

        # Meta dentro de campo de vision, o fuera pero todavia lejos: manejo
        # normal. suggest_command ya hace que la transitabilidad domine y el
        # sesgo hacia la meta (si hay goal_offset_deg) sea suave —
        # PolicyConfig.goal_bias_floor mantiene los corredores no-preferidos
        # con la mayor parte de su puntaje. prev_best_corridor agrega
        # histeresis: sin esto, dos corredores laterales con puntaje parecido
        # pueden intercambiar el primer puesto de frame a frame solo por
        # ruido del modelo, y el rover nunca completa el giro (se queda
        # oscilando ang=+/- sin ganar terreno).
        decision = suggest_command(
            result.mask, self.policy, goal_offset_deg=goal_offset_deg,
            prev_best_corridor=self._prev_best_corridor,
        )
        self._prev_best_corridor = decision.best_corridor if decision.best_corridor >= 0 else None
        return decision
