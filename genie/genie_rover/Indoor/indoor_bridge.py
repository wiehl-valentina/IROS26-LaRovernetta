
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
    self.send() / self.request_stop() / self._maybe_dump_debug()

Lo unico que NO se reutiliza de Bridge es _step(): el `_step` original arma
la meta local a partir de un checkpoint GPS (goal_from_gps/HeadingEstimator).
Ese es justamente el pedazo que no tiene sentido offline/indoor, asi que
IndoorBridge._step() lo reemplaza por completo (percepcion, mapa, chequeo de
obstaculo y seguimiento de camino siguen siendo las mismas llamadas que
Bridge._step, solo cambia DE DONDE sale la meta: ahora de
genie_rover.mission.ConeMissionFSM en vez de un checkpoint GPS).

DOS MISIONES EN LA MISMA CLASE (`mission.mode` en el config):

    "cone_tour"          la de siempre. La meta sale de ConeMissionFSM: el
                         cono es el objetivo y el checkpoint, la ruta se
                         recorre por waypoints/frontier/wander y la mision
                         termina al agotarla.

    "semantic_corridor"  el rover navega por TRAMOS RECTOS con el rumbo
                         bloqueado (SemanticMissionFSM): la meta es un punto
                         deslizante sobre un carril virtual, no un punto fijo,
                         y lo que hace avanzar la mision es que el VLM
                         confirme un hito visual (un piano, tres sillas en
                         hilera, el fin del pasillo). Los waypoints, si hay,
                         pasan a ser una SUGERENCIA de rumbo que nunca es
                         meta ni condicion de fin. Y el cono deja de ser
                         objetivo: si aparece uno, se le saca una foto y el
                         rover sigue derecho sin frenar ni desviarse
                         (_maybe_photo_cone_event).

En los dos modos el control de bajo nivel es exactamente el mismo: SAM-TP ->
BEV -> plan_on_bev -> PathFollower, con front_is_blocked y el recovery
heredado de Bridge. Lo unico que cambia es DE DONDE sale la meta local.

run() SI tiene un override chico (ver mas abajo): necesita cerrar el
hilo/nodo ROS2 de RtabmapPoseBridge al terminar, algo que Bridge.run() no
sabe hacer porque no conoce ese atributo.

Uso (mismo patron que bridge.py: dry-run por defecto, --go para moverse):

    # simulacro
    python -m genie_rover.Indoor.indoor_bridge --config configs/indoor_cone_search.yaml

    # de verdad
    python -m genie_rover.Indoor.indoor_bridge --config configs/indoor_cone_search.yaml \
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

from ..bridge import Bridge, _check_placeholders
from .cone_detector import ConeDetectorConfig, ConeDetectorPipeline, ground_point_from_bbox
from ..console_report import MissionConsoleReporter
from .external_map import load_ros_occupancy_map
from .mission import (
    ConeMissionFSM,
    ConePhotoLog,
    MissionConfig,
    SemanticMissionConfig,
    SemanticMissionFSM,
    SemanticState,
    corridor_hint_from_bev,
)
from .vlm_semantic import SemanticVlm, VlmConfig
from .rtabmap_pose_bridge import RtabmapPoseBridge
from ..navigation import DriveCommand, front_is_blocked
from ..odometry import Pose
from ..sdk_client import RoverError
from .route_status import RouteStatus

class IndoorBridge(Bridge):
    # La mision de cono no usa checkpoints GPS en absoluto -- pisa el default
    # de Bridge (True) para que Bridge.run() no los lea ni los anuncie.
    uses_gps_checkpoints = False

    # Atributo de CLASE (no solo de instancia): asi un IndoorBridge armado
    # con __new__(IndoorBridge) saltando __init__ (como hace
    # test_indoor_mission_offline.py) sigue teniendo self._rtab is None en
    # vez de romper con AttributeError.
    _rtab = None

    # Mismo motivo que _rtab: un IndoorBridge armado con __new__ saltando
    # __init__ tiene igual un publicador APAGADO en vez de romper _step() con
    # AttributeError. Al estar deshabilitado todos sus metodos salen temprano,
    # asi que compartir la instancia entre esos casos es inerte.
    _route_status = RouteStatus(None, enabled=False)

    # Mismo motivo que _rtab/_route_status: defaults de CLASE para que un
    # IndoorBridge armado con __new__ saltando __init__ (como hace
    # test_indoor_mission_offline.py) corra el modo de conos de siempre sin
    # romper con AttributeError en _step().
    mission_mode = "cone_tour"      # "cone_tour" | "semantic_corridor"
    sem_cfg = None
    vlm = None
    cone_log = None
    cone_photo_enabled = True
    cone_photo_min_conf = 0.45

    def __init__(self, cfg: dict, dry_run: bool = True, debug_dir: str | None = None):
        super().__init__(cfg, dry_run=dry_run, debug_dir=debug_dir)

        if not self.use_map:
            raise ValueError(
                "IndoorBridge necesita memory.enabled: true — sin PersistentMap "
                "no hay ni exploracion por frontera ni una pose estable para "
                "ubicar al cono en el piso."
            )

        self._maybe_load_external_map(cfg.get("memory", {}))
        self._maybe_enable_rtabmap_correction(cfg)

        cone_cfg = ConeDetectorConfig.from_dict(cfg.get("cone", {}))
        self.cone_detector = ConeDetectorPipeline(cone_cfg)

        # --- que mision se corre -------------------------------------------
        # "cone_tour"         la de siempre: el cono es la meta y el checkpoint
        #                     (ConeMissionFSM, waypoints/frontier/wander).
        # "semantic_corridor" tramos rectos con rumbo bloqueado; lo que hace
        #                     avanzar la mision es que el VLM confirme un hito
        #                     visual, y el cono deja de ser meta: solo dispara
        #                     una foto sin frenar ni desviar.
        mission_raw = cfg.get("mission", {}) or {}
        self.mission_mode = str(mission_raw.get("mode", "cone_tour"))
        if self.mission_mode not in ("cone_tour", "semantic_corridor"):
            raise ValueError(
                f"mission.mode desconocido: {self.mission_mode!r} "
                "(cone_tour | semantic_corridor)")

        # MissionConfig se arma en los dos modos: en el semantico no gobierna
        # la navegacion, pero sigue dando los parametros del detector de cono
        # y los defaults de la foto. from_dict ignora las claves que no le
        # corresponden (mode/segments/route_hint/console_color/...), asi que
        # los dos modos pueden compartir la misma seccion `mission:`.
        mission_cfg = MissionConfig.from_dict(mission_raw)
        self.mission_cfg = mission_cfg

        if self.mission_mode == "semantic_corridor":
            self.sem_cfg = SemanticMissionConfig.from_dict(mission_raw)
            self.mission = SemanticMissionFSM(self.sem_cfg)

            # El cono como EVENTO, no como meta: esto vive fuera de la FSM a
            # proposito, para que sacar la foto no pueda alterar la
            # trayectoria (ver _maybe_photo_cone_event).
            foto_cfg = mission_raw.get("cone_photo", {}) or {}
            self.cone_photo_enabled = bool(foto_cfg.get("enabled", True))
            self.cone_photo_min_conf = float(
                foto_cfg.get("min_confidence", mission_cfg.detect_confidence_min))
            self.cone_log = ConePhotoLog(
                float(foto_cfg.get("revisit_radius_m", mission_cfg.revisit_radius_m)))
            self.photo_dir = Path(foto_cfg.get("photo_dir", mission_cfg.photo_dir))

            self.vlm = SemanticVlm(VlmConfig.from_dict(cfg.get("vlm", {})))
            tramos = ", ".join(seg.id for seg in self.sem_cfg.segments)
            print(f"[indoor_bridge] mision semantica: {len(self.sem_cfg.segments)} "
                  f"tramo(s) [{tramos}], VLM backend={self.vlm.cfg.backend}")
            if not self.vlm.enabled:
                print("[indoor_bridge] AVISO: vlm.backend='off' -- ningun hito se "
                      "va a confirmar nunca; cada tramo va a terminar por su "
                      "fail-safe de distancia (segments[].on_timeout).")
        else:
            self.mission = ConeMissionFSM(mission_cfg)
            self.cone_photo_enabled = mission_cfg.take_photo
            self.photo_dir = Path(mission_cfg.photo_dir)

        # --- ruta dibujada en el dashboard del SDK (localhost:8000) ---------
        # Publica el estado de la ruta a un JSON dentro de static/ del SDK,
        # que map.js lee cada 1.5 s para dibujarla ENCIMA de su mapa. Se
        # activa SOLO con dashboard.route_status_path en el config Y una
        # mision que tenga algo que dibujar (waypoints, o el modo semantico):
        # sin eso queda apagado, no escribe nada y el dashboard queda
        # exactamente como el original (no hay endpoint nuevo ni cambio en
        # main.py del SDK -- el archivo ES el canal y el interruptor).
        # En "cone_tour" la ruta vive en ConeMissionFSM._route; en el modo
        # semantico, si hay guia del mapeo previo, en
        # SemanticMissionFSM._route_hint. Los dos son privados y pueden ser
        # None, de ahi los getattr.
        dash_cfg = cfg.get("dashboard", {}) or {}
        route = (getattr(self.mission, "_route", None)
                 or getattr(self.mission, "_route_hint", None))
        self._route_status = RouteStatus(
            dash_cfg.get("route_status_path"),
            waypoints=getattr(route, "points", []),
            enabled=(bool(dash_cfg.get("route_status_path"))
                     and (mission_cfg.search_mode == "waypoints"
                          or self.mission_mode == "semantic_corridor")),
            anchor=dash_cfg.get("anchor"),
            reach_radius_m=mission_cfg.waypoint_reach_radius_m,
            frame=("map" if self._rtab is not None else "odom"),
        )
        if self._route_status.enabled:
            print(f"[indoor_bridge] ruta publicada al dashboard en "
                  f"{dash_cfg['route_status_path']} "
                  f"({len(self._route_status.waypoints)} waypoints)")

        # Estadisticas propias, ademas de las de LoopStats (heredadas).
        self.cone_frames_detected = 0
        self.mission_final_state = "SEARCH"
        self.mission_final_distance_m: float | None = None

        if self.cone_photo_enabled:
            self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.cone_photos_saved = 0

        # Consola en tabla (ver console_report.py): reemplaza el par de lineas
        # sueltas por iteracion ("estado=..." + "[ENVIADO]/[DRY-RUN]...") por
        # una fila compacta, con los cambios de estado/checkpoint/eventos en su
        # propia linea aparte para que no se pierdan en el medio de la tabla.
        # mission.console_color: null/ausente = autodetectar (isatty), true/false
        # = forzar. Se lee del dict crudo en vez de sumarlo a MissionConfig para
        # no tocar mission.py por una opcion puramente de presentacion.
        self._reporter = MissionConsoleReporter(
            target_label="cono",
            enable_color=(cfg.get("mission", {}) or {}).get("console_color"))

    # ------------------------------------------------------ mapa importado

    def _maybe_load_external_map(self, mem_cfg: dict) -> None:
        """Reemplaza el PersistentMap vacio (armado por Bridge.__init__) por
        uno importado de ROS map_server / RTAB-Map, si el config lo pide.

        Ver genie_rover/external_map.py para el formato soportado y, sobre
        todo, la limitacion importante: sin esto el mapa arranca vacio
        (comportamiento de siempre); CON esto, el mapa arranca poblado pero
        SOLO queda bien alineado con la realidad si `start_pose_map` refleja
        de verdad donde esta parado el robot en el instante de arrancar.

        Esto es el "caso A" (pose de arranque CONOCIDA). Para el "caso B"
        (no se sabe donde arranca el robot dentro del mapa grabado), usar en
        cambio mapping.rtabmap_correction (ver _maybe_enable_rtabmap_correction)
        — son dos estrategias alternativas, no hace falta ni tiene sentido
        activar las dos juntas.
        """
        ext = (mem_cfg or {}).get("external_map", {}) or {}
        if not ext.get("enabled", False):
            return

        yaml_path = ext.get("yaml_path")
        if not yaml_path:
            raise ValueError("memory.external_map.enabled=true pero falta yaml_path")

        self.pmap = load_ros_occupancy_map(
            yaml_path,
            resolution_m_per_px=ext.get("resolution_m_per_px") or self.resolution,
            margin_m=float(ext.get("margin_m", 1.5)),
            update_weight=float(mem_cfg.get("update_weight", 0.45)),
            decay_per_s=float(mem_cfg.get("decay_per_s", 0.08)),
            recenter_margin_m=float(mem_cfg.get("recenter_margin_m", 1.5)),
            min_confidence=float(mem_cfg.get("min_confidence", 0.15)),
        )

        sp = ext.get("start_pose_map", {}) or {}
        self.odometry.pose = Pose(
            float(sp.get("x_m", 0.0)), float(sp.get("y_m", 0.0)),
            math.radians(float(sp.get("yaw_deg", 0.0))),
        )
        print(f"[indoor_bridge] AVISO: arrancando con mapa importado de {yaml_path}. "
              f"Esto asume que el robot esta PARADO AHORA en "
              f"({self.odometry.pose.x:+.2f}, {self.odometry.pose.y:+.2f}) mirando a "
              f"{math.degrees(self.odometry.pose.theta):+.0f} grados de ESE mapa. "
              "Si no es asi, el mapa va a quedar desalineado con lo que ve la camara.")

    # ------------------------------------------------------ correccion RTAB-Map

    def _maybe_enable_rtabmap_correction(self, cfg: dict) -> None:
        """Arma RtabmapPoseBridge si `mapping.rtabmap_correction.enabled: true`.

        BUG que esto arregla: esta logica existia antes SOLO en
        MapSessionBridge (la sesion de grabar el mapa, `map-session`), nunca
        en IndoorBridge (la mision real de conos, `indoor-bridge`). Poner
        `mapping.rtabmap_correction.enabled: true` en el config no tenia
        ningun efecto en la mision real: seguia navegando 100% a ciegas por
        dead-reckoning (rueda+giro), sin relocalizar nunca contra un mapa
        grabado antes. Se movio la logica aca (la clase base de ambas) para
        que la mision real tambien pueda arrancar en una pose desconocida y
        relocalizarse apenas la camara reconoce algo del mapa grabado (caso
        B de _maybe_load_external_map, arriba).

        Requiere que, EN PARALELO, este corriendo la sesion ROS2 de mapeo en
        modo localizacion (`rtabmap_mapping.launch.py ... localization:=true`,
        argumento `database_path:=` -- OJO, no `db_path:=`). Deshabilitado
        por defecto: si la seccion no esta o `enabled` no es true, no cambia
        nada del comportamiento actual. Nunca lanza: si rclpy/tf2_ros no
        estan instalados o el nodo de mapeo no esta corriendo, cae a
        dead-reckoning solo con un aviso.
        """
        rtab_cfg = (cfg.get("mapping", {}) or {}).get("rtabmap_correction", {}) or {}
        if not rtab_cfg.get("enabled", False):
            return
        try:
            self._rtab = RtabmapPoseBridge(
                map_frame=rtab_cfg.get("map_frame", "map"),
                base_frame=rtab_cfg.get("base_frame", "base_link"),
                lookup_timeout_s=float(rtab_cfg.get("lookup_timeout_s", 0.2)),
            )
            print("[indoor_bridge] correccion de pose por RTAB-Map habilitada "
                  f"({rtab_cfg.get('map_frame', 'map')} -> "
                  f"{rtab_cfg.get('base_frame', 'base_link')})")
        except Exception as exc:
            print("[indoor_bridge] AVISO: no pude habilitar la correccion de "
                  f"RTAB-Map, sigo con dead-reckoning solo: {exc}")
            self._rtab = None

    # ------------------------------------------------------------------ cono

    def _pick_cone(self, rgb: np.ndarray) -> tuple:
        """Elige QUE deteccion de cono usar este frame, entre TODAS las que
        dio el detector (no solo la de mas confianza/area).

        Puede haber mas de un cono a la vista, y el detector puede preferir
        por area/confianza un falso positivo mas grande antes que el cono
        real, mas chico porque esta parcialmente tapado o mas lejos. Ademas,
        un cono a otra altura (estante, escalon) puede no dar interseccion
        valida con el plano del piso para su bbox "obvio" pero si para una
        deteccion secundaria mas cercana al piso. Por eso se calcula el punto
        en el piso (con el respaldo por tamaño aparente de
        ground_point_from_bbox) para CADA candidato, se descartan los que no
        dan una posicion valida, y se elige el mas cercano de los que quedan
        — es la señal mas confiable de "este es el cono al que hay que ir",
        mejor que la confianza cruda del detector.
        """
        detector = self.cone_detector
        if hasattr(detector, "detect_all"):
            candidates = detector.detect_all(rgb)
        else:  # compatibilidad si algun backend viejo no lo implementa
            one = detector.detect(rgb)
            candidates = [one] if one is not None else []

        real_height_m = getattr(getattr(detector, "cfg", None), "real_height_m", None)
        best_det, best_ground = None, None
        for det in candidates:
            ground = ground_point_from_bbox(
                det, self.perception.camera_k, self.perception.camera_pose,
                ground_z=self.perception.ground_z, real_height_m=real_height_m,
            )
            if ground is None:
                continue
            if best_ground is None or ground.distance_m < best_ground.distance_m:
                best_det, best_ground = det, ground
        return best_det, best_ground

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

        # Si hay correccion de RTAB-Map activa, reemplazar la pose ANTES de
        # que odometry.update() integre el delta de este frame -- asi el
        # dead-reckoning arranca cada frame desde la ultima pose relocalizada
        # (map -> base_link) en vez de acumular deriva indefinidamente sobre
        # una pose que nunca se corrige. Si get_corrected_pose() todavia no
        # tiene una TF disponible (por ejemplo, arranque, o la camara no
        # reconocio nada del mapa todavia), sigue con dead-reckoning para
        # este frame sin romper nada.
        if self._rtab is not None:
            corrected = self._rtab.get_corrected_pose()
            if corrected is not None:
                self.odometry.pose = corrected

        telem = self.client.telemetry()
        res = self.perception.process(rgb)

        cone, cone_ground = self._pick_cone(rgb)
        if cone is not None:
            self.cone_frames_detected += 1

        pose = self.odometry.update(telem.raw)
        self.pmap.integrate(res.traversability, res.observed, pose,
                            self.forward_range, self.side_range, t=now)
        plan_bev, plan_obs = self.pmap.extract_bev(
            pose, self.plan_forward_m, self.plan_side_m,
            *res.traversability.shape)
        fwd, side = self.plan_forward_m, self.plan_side_m
        st = self.pmap.stats()

        # Se calcula ACA (antes se calculaba recien al llamar a plan_on_bev):
        # el modo semantico lo necesita para leer el centro del pasillo del
        # mismo BEV, antes de decidir la meta.
        bev_resolution_m = (2.0 * side) / plan_bev.shape[1]

        if self.mission_mode == "semantic_corridor":
            mission_goal = self._semantic_goal(pose, now, rgb, plan_bev, plan_obs,
                                               bev_resolution_m)
        else:
            mission_goal = self.mission.update(pose, self.pmap, cone, cone_ground, now)
        self.mission_final_state = mission_goal.state
        self.mission_final_distance_m = (cone_ground.distance_m if cone_ground is not None
                                         else self.mission_final_distance_m)

        # Cambios de estado/checkpoint van a su propia linea, bien marcada --
        # antes quedaban enterrados como una palabra mas en cada fila y era
        # facil perderselos en una corrida larga. La tabla (self._reporter.row,
        # mas abajo, uno por rama de salida) reemplaza la vieja linea
        # "estado=... meta=... pose=..." de un print suelto por iteracion.
        self._reporter.state_change(mission_goal.state, mission_goal.reason)
        self._reporter.checkpoint(self.mission.checkpoints_done, mission_goal.reason)

        # Mismo estado que la fila de consola, pero para el dashboard. Va ACA,
        # antes de las ramas de salida temprana (foto / mision cumplida /
        # obstaculo / sin camino), asi el overlay se actualiza en todas y no
        # se queda congelado justo cuando pasa algo interesante. Apagado
        # (enabled=False) es un return inmediato; el throttle interno de 0.5 s
        # evita castigar el disco a la frecuencia del loop.
        self._route_status.publish(
            pose,
            current_index=getattr(getattr(self.mission, "_route", None), "idx", None),
            checkpoints_done=self.mission.checkpoints_done,
            state=str(getattr(self.mission.state, "value", self.mission.state)),
            mission_done=mission_goal.mission_done,
            pose_source=("rtabmap" if self._rtab is not None else "odometry"),
        )

        if self.mission_mode == "semantic_corridor":
            # En esta mision lo que importa de un vistazo no es el cono (que
            # ya no es meta) sino en que tramo va y que hito esta esperando.
            consulta = self.mission.current_query()
            cono_desc = (f"{self.mission.segment.id}/"
                         f"{consulta.id if consulta is not None else '-'}")
        else:
            cono_desc = (f"cono {cone.confidence:.2f}" if cone is not None else "sin cono")

        def _row(action: str) -> None:
            self._reporter.row(
                iteration=self.stats.iterations, state=mission_goal.state, pose=pose,
                target_desc=cono_desc, map_cells=st["celdas_vistas"],
                trav=res.traversability, action=action)

        # El cono, en el modo semantico, no frena ni desvia: se saca la foto
        # y el frame sigue su curso normal (por eso no hay return aca).
        if self.mission_mode == "semantic_corridor":
            self._maybe_photo_cone_event(rgb, cone, cone_ground, pose)

        if mission_goal.request_photo:
            self.send(DriveCommand(0.0, 0.0, mission_goal.reason), quiet=True)
            _row(f"foto: {mission_goal.reason}")
            # Posicion MUNDO del cono que se esta dando por completado, para
            # marcarla en el dashboard donde esta el cono y no donde freno el
            # rover. cone_ground viene en el marco del robot (+x adelante,
            # +y izquierda), misma convencion que odometry.py.
            if cone_ground is not None:
                cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
                self._route_status.add_checkpoint(
                    pose.x + cone_ground.x_forward_m * cos_t - cone_ground.y_left_m * sin_t,
                    pose.y + cone_ground.x_forward_m * sin_t + cone_ground.y_left_m * cos_t,
                )
            self._save_cone_photo(rgb, cone)
            self.mission.mark_photo_taken()
            return

        if mission_goal.mission_done:
            self.send(DriveCommand(0.0, 0.0, mission_goal.reason), quiet=True)
            _row(f"listo: {mission_goal.reason}")
            self._reporter.event("MISION CUMPLIDA", "frenando y terminando la corrida", "green")
            self.request_stop()
            return

        # El chequeo de colision usa SIEMPRE la observacion fresca del frame,
        # igual que en Bridge._step: no queremos que el promedio del mapa
        # diluya algo que se cruzo recien.
        if front_is_blocked(res.traversability, self.resolution):
            self.stats.blocked += 1
            self.send(DriveCommand(0.0, 0.0, "OBSTACULO al frente"), quiet=True)
            _row("frenado: obstaculo al frente")
            return

        plan = plan_on_bev(
            bev_traversability=plan_bev,
            observed_mask=plan_obs,
            goal_x_m=float(mission_goal.x_right_m),
            goal_y_m=float(mission_goal.y_forward_m),
            bev_resolution_m=bev_resolution_m,
            config=self.planner_cfg,
            # Mismo banco de caminos candidatos cacheado que usa
            # Bridge._step() (ver Bridge._path_bank): sin esto, plan_on_bev
            # recalculaba el banco EN CADA FRAME (~3s en la RTX 2080 de
            # referencia), mas que el dead-man watchdog del SDK (3s por
            # defecto) -- el rover se frenaria solo entre comando y comando,
            # exactamente el problema que la mejora de mi compañero elimino
            # del lado outdoor. self._bank/self._bank_shape ya existen: los
            # inicializa Bridge.__init__, heredado via super().__init__().
            candidate_path_bank=self._path_bank(plan_bev.shape, bev_resolution_m),
        )

        path = plan.final_path_xy_m
        if path is None or len(path) < 2:
            self.stats.plans_empty += 1
            self._consecutive_empty += 1
            if self._consecutive_empty >= self.recovery_after_empty:
                _row(f"RECUPERACION ({self._consecutive_empty} planes vacios seguidos)")
                # Bridge._recover() no toma argumentos (arma el frame que
                # necesita internamente, via self.client.front_frame() en
                # _preguntar_vlm() si use_vlm_recovery esta activo). El
                # archivo original llamaba self._recover(rgb) -- eso tira
                # TypeError la primera vez que se llega aca, apenas la
                # mision indoor encadena recovery_after_empty planes vacios.
                self._recover()
            else:
                self.send(DriveCommand(0.0, 0.0, "el planner no encontro camino"), quiet=True)
                _row(f"sin camino ({self._consecutive_empty}/{self.recovery_after_empty})")
            return

        self._consecutive_empty = 0
        self.stats.plans_ok += 1

        # committed=True: igual que Bridge._send_path_command() del lado
        # outdoor -- una vez que el camino ya esta elegido, el seguidor debe
        # CURVAR hacia el en vez de pivotear en el lugar por cada error de
        # rumbo grande. Sin esto la mision indoor pivotea mas de lo
        # necesario en corredores angostos, justo donde girar en el lugar es
        # mas propenso a rozar una pared.
        cmd = self.follower.command(path, committed=True)
        cmd = self._apply_commit(cmd, path)
        if mission_goal.linear_scale != 1.0:
            cmd = DriveCommand(cmd.linear * mission_goal.linear_scale, cmd.angular,
                               cmd.reason + f" (x{mission_goal.linear_scale:.2f} por {mission_goal.state})")

        if cmd.linear == 0.0 and cmd.angular != 0.0:
            self._consecutive_turns += 1
            self._turn_sign_history.append(1.0 if cmd.angular > 0 else -1.0)
            if self._consecutive_turns >= self.max_consecutive_turns:
                _row(f"DESATASCO ({self._consecutive_turns} giros seguidos)")
                self._unstick()
                return
        else:
            self._consecutive_turns = 0
            self._turn_sign_history.clear()

        self.send(cmd, quiet=True)
        _row(f"{cmd.linear:+.2f}m/s {cmd.angular:+.2f}rad/s  {cmd.reason}")
        self._maybe_dump_debug_indoor(rgb, res, plan, cone)

    # ------------------------------------------------- mision semantica

    def _semantic_goal(self, pose, now: float, rgb: np.ndarray,
                       plan_bev: np.ndarray, plan_obs: np.ndarray,
                       bev_resolution_m: float):
        """Meta del frame en el modo "semantic_corridor".

        Junta las dos entradas de SemanticMissionFSM y la llama:

          * el VLM (semantico, lento, ~1.4 Hz, puede no haber respuesta),
            que solo decide SI el tramo termino;
          * el BEV (geometrico, todos los frames), que dice donde esta el
            centro del pasillo y cuanto despeje hay adelante.

        Si el VLM no contesto todavia, `obs` es None y el tramo sigue recto:
        es el comportamiento seguro, no un error.
        """
        consulta = self.mission.current_query()
        obs = None
        if consulta is not None and self.vlm is not None:
            # Durante un giro la latencia si importa (el robot esta pivoteando
            # a ciegas hasta que le confirmen la apertura), asi que ahi se
            # pregunta mas seguido.
            girando = self.mission.state == SemanticState.TURN_SEARCH
            obs = self.vlm.observe(rgb, consulta, now, urgent=girando)

        pasillo = corridor_hint_from_bev(
            plan_bev, plan_obs, bev_resolution_m,
            lookahead_m=self.sem_cfg.segment_lookahead_m,
            row0_is_far=self.sem_cfg.bev_row0_is_far,
        )
        return self.mission.update(pose, now, vlm=obs, corridor=pasillo)

    def _maybe_photo_cone_event(self, rgb: np.ndarray, cone, cone_ground,
                                pose) -> None:
        """Foto del cono como EVENTO: no frena, no desvia, no cambia de fase.

        Es la diferencia de fondo con `cone_tour`, donde ver un cono
        significaba abandonar la ruta e ir hacia el. Aca el cono es una cosa
        que se anota al pasar: si hay una deteccion valida, lo bastante
        confiable y que no sea un cono ya fotografiado (filtro por radio en
        el mapa, igual que antes), se guarda el frame y se sigue derecho en
        el mismo ciclo.
        """
        if not self.cone_photo_enabled or self.cone_log is None:
            return
        if cone is None or cone_ground is None:
            return
        if cone.confidence < self.cone_photo_min_conf:
            return

        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        x_world = pose.x + cone_ground.x_forward_m * cos_t - cone_ground.y_left_m * sin_t
        y_world = pose.y + cone_ground.x_forward_m * sin_t + cone_ground.y_left_m * cos_t
        if not self.cone_log.should_photograph(x_world, y_world):
            return

        self.mission_final_distance_m = cone_ground.distance_m
        self._save_cone_photo(rgb, cone)
        n = self.cone_log.record(x_world, y_world)
        self._route_status.add_checkpoint(x_world, y_world, label=f"cono {n}")
        self._reporter.event(
            "FOTO DE CONO",
            f"cono #{n} a {cone_ground.distance_m:.2f} m, sigo derecho sin frenar",
            "green")

    # ------------------------------------------------------------------ ciclo

    def run(self, max_seconds: float | None = None) -> None:
        """Igual que Bridge.run(), pero ademas cierra el hilo/nodo ROS2 de
        RtabmapPoseBridge al terminar (fin normal, --max-seconds, Ctrl-C o
        excepcion) -- Bridge.run() no sabe que self._rtab existe.

        Tambien apaga el overlay del dashboard: close() deja el JSON en
        {"active": false} para que no quede una ruta vieja dibujada en
        localhost:8000 despues de terminar la corrida."""
        try:
            super().run(max_seconds=max_seconds)
        finally:
            if self._rtab is not None:
                self._rtab.shutdown()
            if self.vlm is not None:
                self.vlm.close()
            self._route_status.close()

    # ------------------------------------------------------------------ foto

    def _save_cone_photo(self, rgb: np.ndarray, cone) -> None:
        """Guarda una foto del checkpoint (cono) apenas se confirma y el
        rover esta a stop_distance_m o menos. Llamada desde _step() cuando
        mission_goal.request_photo es True (estado PHOTO de ConeMissionFSM).
        Nunca lanza: si falla, la mision igual sigue su curso (mark_photo_taken
        se llama de todas formas para no quedar trabada esperando para
        siempre por un error de disco/permmisos).
        """
        try:
            from PIL import Image, ImageDraw
            frame = Image.fromarray(rgb)
            if cone is not None:
                draw = ImageDraw.Draw(frame)
                x0, y0, x1, y1 = cone.bbox_xyxy
                draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=4)
                dist_txt = (f"{self.mission_final_distance_m:.2f}m"
                           if self.mission_final_distance_m is not None else "?")
                draw.text((x0, max(0, y0 - 14)),
                         f"{cone.label} {cone.confidence:.2f} @ {dist_txt}",
                         fill=(255, 140, 0))
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.photo_dir / f"cono_{self.cone_photos_saved:03d}_{ts}.jpg"
            frame.save(path, quality=90)
            self.cone_photos_saved += 1
            print(f"[indoor_bridge] foto del cono (checkpoint) guardada en {path}")
        except Exception as exc:
            print(f"[indoor_bridge] no pude guardar la foto del cono: {exc}")

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
        if self.mission_mode == "semantic_corridor":
            self._print_summary_semantic()
            return
        print("  --- mision indoor (cono) ---")
        print(f"  frames con cono detectado: {self.cone_frames_detected}")
        print(f"  checkpoints completados:   {self.mission.checkpoints_done}")
        print(f"  fotos del cono guardadas:  {self.cone_photos_saved}")
        print(f"  estado final:              {self.mission_final_state}")
        if self.mission_final_distance_m is not None:
            print(f"  ultima distancia al cono:  {self.mission_final_distance_m:.2f} m")
        cumplida = self.mission_final_state == "STOP"
        print(f"  mision cumplida:           {'SI' if cumplida else 'NO'}")

    def _print_summary_semantic(self) -> None:
        total = len(self.sem_cfg.segments) if self.sem_cfg is not None else 0
        print("  --- mision indoor (tramos + hitos visuales) ---")
        print(f"  tramos completados:        {self.mission.segments_done}/{total}")
        print(f"  tramo final:               {self.mission.segment.id}")
        print(f"  estado final:              {self.mission_final_state}")
        print(f"  frames con cono detectado: {self.cone_frames_detected}")
        print(f"  fotos de cono guardadas:   {self.cone_photos_saved}")
        if self.vlm is not None:
            print(f"  VLM:                       {self.vlm.stats_line()}")
        if self.mission.failed_reason:
            print(f"  mision cumplida:           NO ({self.mission.failed_reason})")
        else:
            cumplida = self.mission.state == SemanticState.DONE
            print(f"  mision cumplida:           {'SI' if cumplida else 'NO'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--go", action="store_true",
                    help="enviar comandos de verdad (sin esto es simulacro)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--mode", choices=["cone_tour", "semantic_corridor"], default=None,
                    help="pisa mission.mode del config. cone_tour = la mision de "
                         "siempre (el cono es la meta); semantic_corridor = tramos "
                         "rectos con cambios de fase confirmados por el VLM")
    ap.add_argument("--vlm-backend", choices=["http", "gemini", "off"], default=None,
                    help="pisa vlm.backend del config (solo en semantic_corridor). "
                         "'off' corre el recorrido sin VLM: cada tramo termina por "
                         "su fail-safe de distancia")
    ap.add_argument("--search-mode", choices=["wander", "frontier", "waypoints"],
                    default=None,
                    help="pisa mission.search_mode del config, para elegir el modo "
                         "de busqueda del tour de checkpoints sin editar el yaml "
                         "(wander | frontier | waypoints)")
    ap.add_argument("--waypoints-path", default=None,
                    help="pisa mission.waypoints_path del config (solo tiene efecto "
                         "con --search-mode waypoints o si el config ya lo pide), "
                         "ej. configs/waypoints_example.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    mission_cfg = cfg.setdefault("mission", {})
    if args.mode is not None:
        mission_cfg["mode"] = args.mode
    if args.vlm_backend is not None:
        cfg.setdefault("vlm", {})["backend"] = args.vlm_backend
    if args.search_mode is not None:
        mission_cfg["search_mode"] = args.search_mode
    if args.waypoints_path is not None:
        mission_cfg["waypoints_path"] = args.waypoints_path

    modo = mission_cfg.get("mode", "cone_tour")
    if modo == "semantic_corridor":
        if not mission_cfg.get("segments"):
            raise SystemExit(
                "mission.mode 'semantic_corridor' necesita mission.segments (la lista "
                "de tramos con su hito visual) -- ver configs/indoor_semantic_tour.yaml."
            )
    elif (mission_cfg.get("search_mode") == "waypoints"
          and not mission_cfg.get("waypoints_path")):
        raise SystemExit(
            "search_mode 'waypoints' necesita mission.waypoints_path (via config o "
            "--waypoints-path) -- ver configs/waypoints_example.yaml."
        )

    _check_placeholders(cfg)

    bridge = IndoorBridge(cfg, dry_run=not args.go, debug_dir=args.debug_dir)

    if modo == "semantic_corridor":
        que_hace = "recorrer los tramos confirmando hitos con el VLM"
        detalle = (f"  Tramos: {len(mission_cfg.get('segments', []))}  |  "
                   f"VLM: {cfg.get('vlm', {}).get('backend', 'http')}")
    else:
        que_hace = "buscar el cono"
        detalle = f"  Modo de busqueda: {mission_cfg.get('search_mode', 'wander')}"

    if args.go:
        print("\n" + "=" * 62)
        print(f"  MODO REAL: el rover se va a mover a {que_hace}.")
        print(detalle)
        print("  Ctrl-C frena. Tene el robot a la vista.")
        print("=" * 62)
        for i in (3, 2, 1):
            print(f"  {i} ...")
            time.sleep(1)
    else:
        print(f"[indoor_bridge] {detalle.strip()} (dry-run)")

    bridge.run(max_seconds=args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
