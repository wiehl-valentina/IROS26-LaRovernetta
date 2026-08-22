"""Prueba de integracion de IndoorBridge._step() SIN SDK, SIN SAM-TP y SIN
GPU (mismo espiritu que test_recover_offline.py: no reinventa la logica,
llama al metodo REAL de la clase real sobre un objeto armado a mano que le
da exactamente los atributos que necesita).

Se fabrica con IndoorBridge.__new__(IndoorBridge) (salta __init__, que
cargaria SAM-TP y abriria HTTP) y se completan sus atributos con las
implementaciones REALES de Odometry, PersistentMap, PathFollower,
PlannerConfig, ConeMissionFSM — todo lo que no necesita ni robot ni GPU.
Solo se reemplazan las dos fronteras que si lo necesitan:

    self.client       -> stub: frames en blanco, sin telemetria de ruedas
                         (el robot se queda quieto; esta prueba valida el
                         CABLEADO del paso completo, no la dinamica de
                         manejo real)
    self.cone_detector -> stub: en vez de correr OpenCV, devuelve una
                         secuencia de ConeDetection ya calculada para que
                         corresponda a un cono quieto a distancias
                         decrecientes del robot. Los pixeles de esa
                         secuencia se calculan con el camino INVERSO de
                         cone_detector.ground_point_from_pixel, asi que de
                         paso esta prueba cruza esa geometria contra si misma.
    self.perception    -> stub: BEV siempre completamente transitable, para
                         que el planner encuentre camino sin depender de
                         SAM-TP real.

Lo que SI corre de punta a punta, sin stub: front_is_blocked, plan_on_bev,
PathFollower.command, Bridge._apply_commit/_recover/_unstick heredados,
PersistentMap.integrate/extract_bev, Odometry.update, y sobre todo
ConeMissionFSM.update — que es el modulo nuevo mas critico de esta entrega.

Uso:
    python -m genie_rover.Indoor.test_indoor_mission_offline
"""

from __future__ import annotations

import time
import types

import numpy as np

from genie_path_planner.geometry import camera_planar_axes
from genie_path_planner.planner import PlannerConfig

from ..bridge import LoopStats
from .cone_detector import ConeDetection
from .indoor_bridge import IndoorBridge
from .mission import ConeMissionFSM, MissionConfig
from ..navigation import HeadingEstimator, PathFollower
from ..odometry import Odometry, OdometryConfig
from ..perception import BevResult, camera_pose_from_height_pitch
from ..persistent_map import MapConfig, PersistentMap


def _pixel_from_ground(x_forward: float, y_left: float, camera_k: np.ndarray,
                       camera_pose: np.ndarray, ground_z: float = 0.0
                       ) -> tuple[float, float]:
    """Camino inverso de cone_detector.ground_point_from_pixel: dado un punto
    del piso en el marco del robot, que pixel de la imagen le corresponde.
    Solo se usa aca, para fabricar bounding boxes sinteticos con una
    distancia conocida de antemano."""
    t = camera_pose[:3, 3]
    r = camera_pose[:3, :3]
    fwd_xy, left_xy = camera_planar_axes(camera_pose)
    world_xy = t[:2] + x_forward * fwd_xy + y_left * left_xy
    point_world = np.array([world_xy[0], world_xy[1], ground_z])
    p_cam = r.T @ (point_world - t)
    u = camera_k[0, 0] * p_cam[0] / p_cam[2] + camera_k[0, 2]
    v = camera_k[1, 1] * p_cam[1] / p_cam[2] + camera_k[1, 2]
    return float(u), float(v)


def _build_stub() -> IndoorBridge:
    b = IndoorBridge.__new__(IndoorBridge)  # salta __init__ (SAM-TP + HTTP)

    camera_k = np.array([[900.0, 0.0, 960.0],
                         [0.0, 900.0, 540.0],
                         [0.0, 0.0, 1.0]])
    camera_pose = camera_pose_from_height_pitch(height_m=0.15, pitch_down_deg=15.0)

    # --- percepcion: BEV siempre libre, para no depender de SAM-TP real ---
    bev_shape = (134, 134)
    bev_result = BevResult(
        traversability=np.ones(bev_shape, dtype=np.float32),
        observed=np.ones(bev_shape, dtype=np.uint8),
        image_traversability=np.ones((64, 64), dtype=np.float32),
        stats={"bev_observed_cells": float(bev_shape[0] * bev_shape[1])},
    )
    b.perception = types.SimpleNamespace(
        process=lambda rgb: bev_result,
        camera_k=camera_k, camera_pose=camera_pose, ground_z=0.0,
    )
    b.forward_range, b.side_range = 2.0, 1.2
    b.resolution = 0.03

    # --- cliente: frames en blanco, robot quieto (sin rpms/gyros) ----------
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    b._frame_i = 0

    def _front_frame():
        b._frame_i += 1
        return rgb, float(b._frame_i)

    b.client = types.SimpleNamespace(
        front_frame=_front_frame,
        telemetry=lambda: types.SimpleNamespace(raw={"rpms": [], "gyros": []}),
    )
    b.stale_frame_s = 5.0
    b._last_frame_ts = 0.0
    b._last_frame_change = time.time()

    # --- cono: secuencia de distancias decrecientes, via geometria inversa -
    distancias = [None, None, 3.0, 2.4, 1.8, 1.2, 0.9, 0.7, 0.6, 0.55, 0.5, 0.48, 0.48, 0.48]
    detecciones = []
    for d in distancias:
        if d is None:
            detecciones.append(None)
            continue
        u, v = _pixel_from_ground(d, 0.0, camera_k, camera_pose)
        bbox = (u - 20, v - 60, u + 20, v)
        detecciones.append(ConeDetection(bbox_xyxy=bbox, confidence=0.85))
    b._detecciones = iter(detecciones)
    b.cone_detector = types.SimpleNamespace(detect=lambda rgb: next(b._detecciones))
    b.cone_frames_detected = 0
    b.mission_final_state = "SEARCH"
    b.mission_final_distance_m = None

    # --- odometria y mapa: implementaciones reales -------------------------
    b.use_map = True
    b.odometry = Odometry(OdometryConfig(gps_correction=False))
    b.pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    b.plan_forward_m, b.plan_side_m = 3.0, 2.0

    # --- planner y seguidor: implementaciones reales, config chica y rapida
    b.planner_cfg = PlannerConfig(grid_size=80, num_goals=10, num_mid_points_per_goal=5,
                                  path_num_samples=40, footprint_px=6,
                                  threshold_cost=0.6, use_clustering=False,
                                  random_seed=7)
    b.follower = PathFollower(lookahead_m=1.0, align_threshold_deg=90.0,
                              max_linear=0.5, max_angular=0.45, angular_sign=-1.0)
    b.heading_est = HeadingEstimator()

    # --- mision: real -------------------------------------------------------
    b.mission = ConeMissionFSM(MissionConfig(
        detect_confidence_min=0.4, approach_start_m=2.0, verify_at_m=0.8,
        stop_distance_m=0.5, verify_hits_required=3, verify_window=5,
        verify_min_confidence=0.5, verify_max_jump_m=0.6, lost_cone_grace_s=1.0,
        search_mode="wander", approach_max_linear_scale=0.6,
    ))

    # --- estado de seguridad / bucle, igual que Bridge.__init__ ------------
    b.stats = LoopStats()
    b._consecutive_empty = 0
    b.recovery_after_empty = 3
    b.recovery_turn_s = 1.0
    b._consecutive_turns = 0
    b.max_consecutive_turns = 6
    b._turn_sign_history = []
    b.unstick_forward_s = 1.0
    b._commit_side = 0
    b._commit_until = 0.0
    b.commit_hold_s = 2.0
    b.commit_min_deg = 8.0
    b.commit_override_deg = 30.0
    b.debug_dir = None
    b.dry_run = True
    b._stop_requested = False

    return b


def main() -> None:
    print("=== IndoorBridge._step() de punta a punta, sin SDK/SAM-TP/GPU ===\n")
    b = _build_stub()

    estados = []
    for i in range(14):
        if b._stop_requested:
            print(f"  (iteracion {i}: la mision ya pidio detenerse, corto el loop)")
            break
        b._step()
        estados.append(b.mission_final_state)

    print(f"\nprogresion de estados: {estados}")
    print(f"stop_requested: {b._stop_requested}")
    print(f"cone_frames_detected: {b.cone_frames_detected}")
    print(f"mission_final_distance_m: {b.mission_final_distance_m}")

    assert "NAVIGATE" in estados or "APPROACH" in estados, \
        "nunca reacciono a la deteccion del cono"
    assert "VERIFY" in estados, "nunca llego a verificar"
    assert estados[-1] == "STOP", f"deberia terminar en STOP, termino en {estados[-1]}"
    assert b._stop_requested, "STOP deberia haber pedido frenar el bucle (request_stop)"
    assert b.cone_frames_detected >= 3
    assert b.stats.errors == 0, "no debería haber habido excepciones en ningun _step()"

    print("\nTodos los asserts pasaron: percepcion(stub) -> deteccion(stub+geometria real) "
          "-> mapa(real) -> mision(real) -> planner(real) -> seguidor(real) -> freno.")


if __name__ == "__main__":
    main()
