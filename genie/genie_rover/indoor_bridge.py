"""IndoorBridge: variante de Bridge para navegar SIN GPS y buscar un cono.

Hereda de genie_rover.bridge.Bridge y reutiliza TAL CUAL (no reimplementa
nada de esto):

    self.client       RoverClient          (sdk_client.py)
    self.perception    PerceptionPipeline   (perception.py, SAM-TP -> BEV)
    self.odometry      Odometry             (odometry.py, ya funciona sin
                                             GPS: gps_correction se puede
                                             apagar en el config y el resto
                                             de la integracion —
                                             giroscopo+ruedas— no lo necesita)
    self.pmap          PersistentMap        (persistent_map.py, memoria del
                                             mundo, ancla en el marco de la
                                             ODOMETRIA, nunca en GPS)
    self.follower      PathFollower         (navigation.py)
    self.planner_cfg   PlannerConfig
    plan_on_bev(...)                        (genie_path_planner.planner)
    front_is_blocked(...)                   (navigation.py)
    self._recover() / self._unstick() / self._apply_commit()  (bridge.py)
    self.send() / self.request_stop() / self.run() / self._maybe_dump_debug()

Lo unico que NO se reutiliza de Bridge es _step(): el `_step` original arma
la meta local a partir de un checkpoint GPS (goal_from_gps/HeadingEstimator).
Ese es justamente el pedazo que no tiene sentido offline/indoor, asi que
IndoorBridge._step() lo reemplaza por completo (percepcion, mapa, chequeo de
obstaculo y seguimiento de camino siguen siendo las mismas llamadas que
Bridge._step, solo cambia DE DONDE sale la meta: ahora de
genie_rover.mission.ConeMissionFSM en vez de un checkpoint GPS). No se tocó
bridge.py para no arriesgar el comportamiento ya validado en el robot real
para el modo ERC/outdoor.

Uso (mismo patron que bridge.py: dry-run por defecto, --go para moverse):

    # simulacro
    python -m genie_rover.indoor_bridge --config configs/indoor_cone_search.yaml

    # de verdad
    python -m genie_rover.indoor_bridge --config configs/indoor_cone_search.yaml \
        --go --max-seconds 180 --debug-dir debug/indoor_run1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from genie_path_planner.planner import plan_on_bev

from .bridge import Bridge, _check_placeholders
from .cone_detector import ConeDetectorConfig, ConeDetectorPipeline, ground_point_from_bbox
from .mission import ConeMissionFSM, MissionConfig
from .navigation import DriveCommand, front_is_blocked
from .sdk_client import RoverError


class IndoorBridge(Bridge):
    def __init__(self, cfg: dict, dry_run: bool = True, debug_dir: str | None = None):
        super().__init__(cfg, dry_run=dry_run, debug_dir=debug_dir)

        if not self.use_map:
            raise ValueError(
                "IndoorBridge necesita memory.enabled: true — sin PersistentMap "
                "no hay ni exploracion por frontera ni una pose estable para "
                "ubicar al cono en el piso."
            )

        cone_cfg = ConeDetectorConfig.from_dict(cfg.get("cone", {}))
        self.cone_detector = ConeDetectorPipeline(cone_cfg)

        mission_cfg = MissionConfig.from_dict(cfg.get("mission", {}))
        self.mission_cfg = mission_cfg
        self.mission = ConeMissionFSM(mission_cfg)

        # Estadisticas propias, ademas de las de LoopStats (heredadas).
        self.cone_frames_detected = 0
        self.mission_final_state = "SEARCH"
        self.mission_final_distance_m: float | None = None

    # ------------------------------------------------------------------ paso

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
        res = self.perception.process(rgb)

        cone = self.cone_detector.detect(rgb)
        cone_ground = None
        if cone is not None:
            self.cone_frames_detected += 1
            cone_ground = ground_point_from_bbox(
                cone, self.perception.camera_k, self.perception.camera_pose,
                ground_z=self.perception.ground_z,
            )

        pose = self.odometry.update(telem.raw)
        self.pmap.integrate(res.traversability, res.observed, pose,
                            self.forward_range, self.side_range, t=now)
        plan_bev, plan_obs = self.pmap.extract_bev(
            pose, self.plan_forward_m, self.plan_side_m,
            *res.traversability.shape)
        fwd, side = self.plan_forward_m, self.plan_side_m
        st = self.pmap.stats()

        mission_goal = self.mission.update(pose, self.pmap, cone, cone_ground, now)
        self.mission_final_state = mission_goal.state
        self.mission_final_distance_m = (cone_ground.distance_m if cone_ground is not None
                                         else self.mission_final_distance_m)

        cono_desc = (f"cono conf={cone.confidence:.2f}" if cone is not None else "sin cono")
        print(f"[{self.stats.iterations:04d}] estado={mission_goal.state:<9} "
              f"meta=(x_right={mission_goal.x_right_m:+.2f} "
              f"y_forward={mission_goal.y_forward_m:+.2f})  {cono_desc}  "
              f"pose=({pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.theta):+.0f}gr)  "
              f"mapa: {st['celdas_vistas']} celdas  -- {mission_goal.reason}")

        if mission_goal.mission_done:
            self.send(DriveCommand(0.0, 0.0, mission_goal.reason))
            print("[indoor_bridge] mision cumplida, frenando y terminando la corrida")
            self.request_stop()
            return

        # El chequeo de colision usa SIEMPRE la observacion fresca del frame,
        # igual que en Bridge._step: no queremos que el promedio del mapa
        # diluya algo que se cruzo recien.
        if front_is_blocked(res.traversability, self.resolution):
            self.stats.blocked += 1
            self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"))
            return

        plan = plan_on_bev(
            bev_traversability=plan_bev,
            observed_mask=plan_obs,
            goal_x_m=float(mission_goal.x_right_m),
            goal_y_m=float(mission_goal.y_forward_m),
            bev_resolution_m=(2.0 * side) / plan_bev.shape[1],
            config=self.planner_cfg,
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
        self.stats.plans_ok += 1

        cmd = self.follower.command(path)
        cmd = self._apply_commit(cmd, path)
        if mission_goal.linear_scale != 1.0:
            cmd = DriveCommand(cmd.linear * mission_goal.linear_scale, cmd.angular,
                               cmd.reason + f" (x{mission_goal.linear_scale:.2f} por {mission_goal.state})")

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
        self._maybe_dump_debug_indoor(rgb, res, plan, cone)

    # ------------------------------------------------------------------ debug

    def _maybe_dump_debug_indoor(self, rgb, res, plan, cone) -> None:
        if not self.debug_dir:
            return
        try:
            from PIL import Image, ImageDraw
            n = self.stats.iterations
            frame = Image.fromarray(rgb)
            if cone is not None:
                draw = ImageDraw.Draw(frame)
                x0, y0, x1, y1 = cone.bbox_xyxy
                draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=4)
                draw.text((x0, max(0, y0 - 14)), f"{cone.label} {cone.confidence:.2f}",
                         fill=(255, 140, 0))
            frame.save(self.debug_dir / f"{n:05d}_rgb.jpg", quality=80)
            Image.fromarray(plan.visualization).save(self.debug_dir / f"{n:05d}_plan.png")
            np.save(self.debug_dir / f"{n:05d}_bev.npy", res.traversability)
            Image.fromarray(self.pmap.to_image(self.odometry.pose)).save(
                self.debug_dir / f"{n:05d}_mapa.png")
        except Exception as exc:
            print(f"[indoor_bridge] no pude escribir el debug: {exc}")

    def _print_summary(self) -> None:
        super()._print_summary()
        print("  --- mision indoor (cono) ---")
        print(f"  frames con cono detectado: {self.cone_frames_detected}")
        print(f"  estado final:              {self.mission_final_state}")
        if self.mission_final_distance_m is not None:
            print(f"  ultima distancia al cono:  {self.mission_final_distance_m:.2f} m")
        cumplida = self.mission_final_state == "STOP"
        print(f"  mision cumplida:           {'SI' if cumplida else 'NO'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--go", action="store_true",
                    help="enviar comandos de verdad (sin esto es simulacro)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _check_placeholders(cfg)

    bridge = IndoorBridge(cfg, dry_run=not args.go, debug_dir=args.debug_dir)

    if args.go:
        print("\n" + "=" * 62)
        print("  MODO REAL: el rover se va a mover buscando el cono.")
        print("  Ctrl-C frena. Tene el robot a la vista.")
        print("=" * 62)
        for i in (3, 2, 1):
            print(f"  {i} ...")
            time.sleep(1)

    bridge.run(max_seconds=args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
