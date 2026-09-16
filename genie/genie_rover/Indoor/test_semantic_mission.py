"""Autoprueba del modo semantico de mission.py (tramos rectos + hitos VLM).

No necesita robot, camara, SAM-TP, GPU ni VLM: el VLM se reemplaza por un
guion de VlmObservation y el robot por un simulador de 20 lineas que gira
hacia la meta y avanza si linear_scale > 0. Lo que SI corre de verdad es
todo mission.py: HeadingLock, corridor_hint_from_bev, RouteHint,
ConePhotoLog y SemanticMissionFSM completa.

    python -m genie_rover.Indoor.test_semantic_mission
"""

from __future__ import annotations

import math

import numpy as np

from ..odometry import Pose
from .mission import (
    ConePhotoLog,
    CorridorHint,
    HeadingLock,
    RouteHint,
    SemanticMissionConfig,
    SemanticMissionFSM,
    SemanticState,
    VlmObservation,
    corridor_hint_from_bev,
)

SEGMENTS = [
    {"id": "f1_entrada",
     "milestone": {"id": "piano", "prompt": "¿se ve un piano?", "confirm_hits": 3,
                   "min_confidence": 0.6},
     "on_milestone": {"side": "left", "look_for": "una puerta o vano"},
     "max_distance_m": 25.0, "on_timeout": "sweep"},
    {"id": "f2_pasillo_central",
     "milestone": {"id": "tres_sillas", "prompt": "¿tres sillas en hilera?",
                   "confirm_hits": 3, "min_confidence": 0.6},
     "on_milestone": {"side": "left", "look_for": "una apertura lateral"},
     "max_distance_m": 30.0, "on_timeout": "sweep"},
    {"id": "f3_pasillo_final",
     "milestone": {"id": "marca_azul", "prompt": "¿una tuerca azul?",
                   "confirm_hits": 2, "min_confidence": 0.55},
     "on_milestone": {"side": "left", "look_for": "la continuacion del pasillo"},
     "max_distance_m": 30.0, "on_timeout": "advance"},
    {"id": "f4_retorno",
     "milestone": {"id": "fin_pasillo", "prompt": "¿el pasillo termina?",
                   "confirm_hits": 2, "min_confidence": 0.6},
     "on_milestone": {"side": "right", "look_for": "un cono naranja"},
     "max_distance_m": 30.0, "on_timeout": "stop", "final": True},
]


def _cfg(**kw):
    d = {"segments": SEGMENTS, "segment_lookahead_m": 2.5,
         "corridor_centering_weight": 0.6, "milestone_cooldown_s": 1.0}
    d.update(kw)
    return SemanticMissionConfig.from_dict(d)


def test_heading_lock():
    print("=== HeadingLock: carril, no punto fijo ===")
    lock = HeadingLock(Pose(0, 0, 0))
    # La meta se desliza: mas lejos el robot, mas lejos la meta.
    g1 = lock.goal_world(Pose(0, 0, 0), 2.5)
    g2 = lock.goal_world(Pose(10, 0, 0), 2.5)
    print(f"  meta en x=0: {g1}   meta en x=10: {g2}")
    assert abs(g1[0] - 2.5) < 1e-6 and abs(g2[0] - 12.5) < 1e-6, "la meta no se desliza"

    # Desvio lateral: la meta vuelve al carril (y=0), no queda adelante del robot.
    desviado = Pose(4.0, 0.8, 0.0)
    gx, gy = lock.goal_world(desviado, 2.5)
    print(f"  robot desviado 0.8 m -> meta ({gx:.2f}, {gy:.2f})")
    assert abs(gy) < 1e-6, "la meta deberia estar SOBRE el carril"
    assert abs(lock.lateral_error_m(desviado) - 0.8) < 1e-6
    assert abs(lock.progress_m(desviado) - 4.0) < 1e-6

    # Carril a 90 grados (despues de un giro a la izquierda).
    lock90 = HeadingLock(Pose(5, 5, math.pi / 2))
    gx, gy = lock90.goal_world(Pose(5, 7, math.pi / 2), 2.0)
    print(f"  carril a 90 grados -> meta ({gx:.2f}, {gy:.2f})")
    assert abs(gx - 5.0) < 1e-6 and abs(gy - 9.0) < 1e-6


def test_corridor_hint():
    print("\n=== corridor_hint_from_bev ===")
    filas, cols, res = 100, 80, 0.05     # 5 m de fondo, 4 m de ancho
    bev = np.zeros((filas, cols), dtype=np.float32)
    # Pasillo libre de 1 m de ancho, corrido 0.5 m a la IZQUIERDA del robot.
    # Izquierda = y_left positivo = columnas por DEBAJO del centro (x_right < 0).
    centro_col = (cols - 1) / 2.0
    c0 = int(centro_col - 0.5 / res - 10)
    c1 = int(centro_col - 0.5 / res + 10)
    bev[:, c0:c1] = 1.0
    hint = corridor_hint_from_bev(bev, None, res, lookahead_m=2.5)
    print(f"  pasillo corrido a la izq -> offset={hint.lateral_offset_m:+.2f} m  "
          f"despeje={hint.clearance_m:.2f} m")
    assert hint.valid
    assert hint.lateral_offset_m > 0.3, "deberia ver el centro libre a la izquierda"
    # Con el pasillo corrido, la franja CENTRAL esta medio tapada -> despeje 0.
    # Eso es correcto: es justo lo que impide que TURN_ALIGN se de por alineado
    # antes de terminar de girar.
    assert hint.clearance_m < 0.5, "el frente no esta despejado si el pasillo esta corrido"

    centrado = np.zeros((filas, cols), dtype=np.float32)
    centrado[:, int(centro_col - 12):int(centro_col + 12)] = 1.0
    h2 = corridor_hint_from_bev(centrado, None, res, lookahead_m=2.5)
    print(f"  pasillo centrado          -> offset={h2.lateral_offset_m:+.2f} m  "
          f"despeje={h2.clearance_m:.2f} m")
    assert abs(h2.lateral_offset_m) < 0.1 and h2.clearance_m > 2.0

    vacio = corridor_hint_from_bev(np.zeros((filas, cols), np.float32), None, res, 2.5)
    print(f"  BEV todo bloqueado -> valido={vacio.valid}")
    assert not vacio.valid, "sin espacio libre el hint no puede ser valido"


def test_route_hint():
    print("\n=== RouteHint: guia, nunca obligacion ===")
    hint = RouteHint([(5.0, 1.0)], max_bias_deg=25.0)
    th, motivo = hint.bias_heading(Pose(0, 0, 0), 0.0)
    print(f"  correccion chica -> {math.degrees(th):+.1f} grados ({motivo})")
    assert 0 < math.degrees(th) < 25

    lejos = RouteHint([(0.5, 5.0)], max_bias_deg=25.0)
    th2, motivo2 = lejos.bias_heading(Pose(0, 0, 0), 0.0)
    print(f"  correccion grande -> {math.degrees(th2):+.1f} grados ({motivo2})")
    assert abs(th2) < 1e-9, "una sugerencia muy distinta tiene que ignorarse"


def test_cone_photo_log():
    print("\n=== ConePhotoLog: el cono es evento, no meta ===")
    log = ConePhotoLog(revisit_radius_m=1.0)
    assert log.should_photograph(3.0, 1.0)
    log.record(3.0, 1.0)
    assert not log.should_photograph(3.4, 1.2), "mismo cono, no deberia repetir foto"
    assert log.should_photograph(8.0, 1.0), "cono nuevo, si"
    print(f"  fotos registradas: {log.photos}")


class _Sim:
    """Robot de mentira: gira hacia la meta y avanza si linear_scale > 0."""

    def __init__(self, fsm, cfg, dist_hito):
        self.fsm, self.cfg = fsm, cfg
        self.pose = Pose(0, 0, 0)
        self.t = 0.0
        self.dt = 0.5
        self.dist_hito = dist_hito      # a que avance del tramo aparece el hito
        self.turn_theta0 = None
        self.turn_id = None
        self.estados = []

    def _obs(self, q):
        if q is None:
            return None
        if q.id.endswith("__apertura"):
            if self.turn_id != q.id:
                self.turn_id, self.turn_theta0 = q.id, self.pose.theta
            girado = abs(math.degrees(self.pose.theta - self.turn_theta0))
            return VlmObservation(q.id, present=girado >= 75.0, confidence=0.8,
                                  position="centro", t=self.t)
        lock = self.fsm.lane
        avance = lock.progress_m(self.pose) if lock else 0.0
        return VlmObservation(q.id, present=avance >= self.dist_hito, confidence=0.75,
                              position="centro", t=self.t)

    def run(self, max_steps=800):
        hint = CorridorHint(lateral_offset_m=0.03, clearance_m=3.0, valid=True)
        for _ in range(max_steps):
            q = self.fsm.current_query()
            goal = self.fsm.update(self.pose, self.t, vlm=self._obs(q), corridor=hint)
            if not self.estados or self.estados[-1] != goal.state:
                self.estados.append(goal.state)
                print(f"    t={self.t:5.1f}s  {goal.state:<12} {goal.reason}")
            if goal.mission_done:
                return goal
            rumbo = math.atan2(-goal.x_right_m, goal.y_forward_m)
            paso = max(-math.radians(15), min(math.radians(15), rumbo))
            self.pose = Pose(self.pose.x, self.pose.y, self.pose.theta + paso)
            if goal.linear_scale > 0.05:
                d = 0.5 * goal.linear_scale
                self.pose = Pose(self.pose.x + d * math.cos(self.pose.theta),
                                 self.pose.y + d * math.sin(self.pose.theta),
                                 self.pose.theta)
            self.t += self.dt
        raise AssertionError("la mision no termino en max_steps")


def test_circuito_completo():
    print("\n=== SemanticMissionFSM: circuito de 4 tramos de punta a punta ===")
    cfg = _cfg()
    fsm = SemanticMissionFSM(cfg)
    sim = _Sim(fsm, cfg, dist_hito=6.0)
    goal = sim.run()
    print(f"  final: {goal.state} | tramos completados={fsm.segments_done} "
          f"| fallo={fsm.failed_reason}")
    assert goal.mission_done and fsm.state == SemanticState.DONE
    assert fsm.failed_reason is None, "no deberia terminar por fallo"
    assert fsm.segments_done == 4, f"esperaba 4 tramos, hubo {fsm.segments_done}"
    for s in ("RUN_SEGMENT", "TURN_SEARCH", "TURN_ALIGN", "DONE"):
        assert s in sim.estados, f"nunca paso por {s}"
    # Giro real de ~90 grados por tramo: el rumbo final no puede ser el inicial.
    print(f"  rumbo final: {math.degrees(sim.pose.theta):.0f} grados")


def test_un_hito_solo_no_cambia_de_fase():
    print("\n=== Histeresis: una sola confirmacion NO cambia de fase ===")
    cfg = _cfg(milestone_cooldown_s=0.0)
    fsm = SemanticMissionFSM(cfg)
    pose = Pose(0, 0, 0)
    fsm.update(pose, 0.0)
    q = fsm.current_query()
    g = fsm.update(pose, 1.0, vlm=VlmObservation(q.id, True, 0.9, t=1.0))
    print(f"  tras 1 confirmacion: {g.state}")
    assert g.state == "RUN_SEGMENT", "un solo frame no puede disparar el giro"
    g = fsm.update(pose, 1.5, vlm=VlmObservation(q.id, True, 0.9, t=1.5))
    g = fsm.update(pose, 2.0, vlm=VlmObservation(q.id, True, 0.9, t=2.0))
    print(f"  tras 3 confirmaciones seguidas: {g.state}")
    assert g.state == "TURN_SEARCH"

    # Una confianza baja en el medio resetea la cuenta.
    fsm2 = SemanticMissionFSM(_cfg(milestone_cooldown_s=0.0))
    fsm2.update(pose, 0.0)
    q2 = fsm2.current_query()
    fsm2.update(pose, 1.0, vlm=VlmObservation(q2.id, True, 0.9, t=1.0))
    fsm2.update(pose, 1.5, vlm=VlmObservation(q2.id, False, 0.9, t=1.5))
    g2 = fsm2.update(pose, 2.0, vlm=VlmObservation(q2.id, True, 0.9, t=2.0))
    print(f"  con un 'no' en el medio: {g2.state}")
    assert g2.state == "RUN_SEGMENT"


def test_respuesta_vieja_se_descarta():
    print("\n=== Una respuesta del hito anterior no dispara nada ===")
    fsm = SemanticMissionFSM(_cfg(milestone_cooldown_s=0.0))
    pose = Pose(0, 0, 0)
    fsm.update(pose, 0.0)
    for k in range(3):
        g = fsm.update(pose, 1.0 + k * 0.5,
                       vlm=VlmObservation("piano", True, 0.9, t=1.0 + k * 0.5))
    assert g.state == "TURN_SEARCH"
    g = fsm.update(pose, 3.0, vlm=VlmObservation("piano", True, 0.95, t=3.0))
    print(f"  llega otro 'piano' estando en giro: {g.state}")
    assert g.state == "TURN_SEARCH", "una respuesta del hito viejo no puede avanzar la fase"


def test_timeout_stop_y_advance():
    print("\n=== Fail-safes por distancia ===")
    segs = [dict(SEGMENTS[0]), dict(SEGMENTS[3])]
    segs[0] = {**segs[0], "max_distance_m": 2.0, "on_timeout": "stop"}
    cfg = SemanticMissionConfig.from_dict({"segments": segs})
    fsm = SemanticMissionFSM(cfg)
    g = None
    for k in range(20):
        g = fsm.update(Pose(0.5 * k, 0, 0), k * 0.5)
        if g.mission_done:
            break
    print(f"  on_timeout=stop -> {g.state}: {g.reason}")
    assert g.mission_done and fsm.failed_reason is not None

    segs[0] = {**segs[0], "on_timeout": "advance"}
    cfg2 = SemanticMissionConfig.from_dict({"segments": segs})
    fsm2 = SemanticMissionFSM(cfg2)
    for k in range(20):
        g2 = fsm2.update(Pose(0.5 * k, 0, 0), k * 0.5)
        if g2.state == "TURN_SEARCH":
            break
    print(f"  on_timeout=advance -> {g2.state}: {g2.reason}")
    assert g2.state == "TURN_SEARCH", "advance tiene que ejecutar el giro igual"


def test_sweep_de_rescate():
    print("\n=== Barrido de rescate (on_timeout: sweep) ===")
    segs = [{**SEGMENTS[0], "max_distance_m": 2.0, "on_timeout": "sweep"},
            dict(SEGMENTS[3])]
    cfg = SemanticMissionConfig.from_dict({"segments": segs, "milestone_cooldown_s": 0.0,
                                           "sweep_max_deg": 30.0})
    fsm = SemanticMissionFSM(cfg)
    pose = Pose(0, 0, 0)
    estados, t = [], 0.0
    for k in range(200):
        g = fsm.update(pose, t)
        if not estados or estados[-1] != g.state:
            estados.append(g.state)
            print(f"    t={t:5.1f}s  {g.state:<12} {g.reason}")
        rumbo = math.atan2(-g.x_right_m, g.y_forward_m)
        paso = max(-math.radians(15), min(math.radians(15), rumbo))
        if g.linear_scale > 0.05:
            pose = Pose(pose.x + 0.5 * math.cos(pose.theta + paso),
                        pose.y + 0.5 * math.sin(pose.theta + paso), pose.theta + paso)
        else:
            pose = Pose(pose.x, pose.y, pose.theta + paso)
        t += 0.5
        if g.state == "TURN_SEARCH":
            break
    assert "SWEEP" in estados, "nunca entro al barrido"
    assert estados[-1] == "TURN_SEARCH", "tras agotar el barrido tiene que seguir igual"


def test_centrado_corrige_hacia_el_pasillo():
    print("\n=== El centrado por BEV corrige el carril ===")
    fsm = SemanticMissionFSM(_cfg(corridor_centering_weight=0.6,
                                  corridor_centering_max_m=0.8))
    pose = Pose(0, 0, 0)
    sin = fsm.update(pose, 0.0, corridor=None)
    con = fsm.update(pose, 0.5,
                     corridor=CorridorHint(lateral_offset_m=0.6, clearance_m=3.0, valid=True))
    print(f"  sin hint: x_right={sin.x_right_m:+.2f}   con pasillo a la izq: "
          f"x_right={con.x_right_m:+.2f}")
    assert con.x_right_m < sin.x_right_m - 0.2, "deberia corregirse hacia la izquierda"
    assert abs(con.x_right_m) <= 0.8 + 1e-6, "la correccion tiene que estar topeada"


def _self_test() -> None:
    test_heading_lock()
    test_corridor_hint()
    test_route_hint()
    test_cone_photo_log()
    test_circuito_completo()
    test_un_hito_solo_no_cambia_de_fase()
    test_respuesta_vieja_se_descarta()
    test_timeout_stop_y_advance()
    test_sweep_de_rescate()
    test_centrado_corrige_hacia_el_pasillo()
    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
