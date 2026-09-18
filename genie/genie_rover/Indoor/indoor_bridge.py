"""IndoorBridge: variante de Bridge para navegar SIN GPS, mision SEMANTICA.

Hereda de genie_rover.bridge.Bridge y reutiliza TAL CUAL (no reimplementa
nada de esto):

    self.client        RoverClient          (sdk_client.py)
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
Bridge._step, solo cambia DE DONDE sale la meta).

LA MISION (unica): "semantic_corridor". El rover navega por TRAMOS RECTOS con
el rumbo bloqueado (SemanticMissionFSM): la meta es un punto deslizante sobre
un carril virtual, no un punto fijo, y lo que hace avanzar la mision es que
el VLM confirme un hito visual (un piano, tres sillas en hilera, el fin del
pasillo). Los waypoints, si hay, pasan a ser una SUGERENCIA de rumbo que
nunca es meta ni condicion de fin.

ELIMINADO: toda la deteccion de conos (cone_detector.py, ConeMissionFSM,
ConePhotoLog, la seccion `cone:` del config, el modo "cone_tour" y las fotos
de checkpoint). El rover ya no busca, ni verifica, ni fotografia conos.

El control de bajo nivel es el de siempre: SAM-TP -> BEV -> plan_on_bev ->
PathFollower, con front_is_blocked y el recovery heredado de Bridge.

run() SI tiene un override chico (ver mas abajo): necesita cerrar el
hilo/nodo ROS2 de RtabmapPoseBridge al terminar, algo que Bridge.run() no
sabe hacer porque no conoce ese atributo.

Uso (mismo patron que bridge.py: dry-run por defecto, --go para moverse):

    # simulacro
    python -m genie_rover.Indoor.indoor_bridge --config configs/indoor_semantic_tour.yaml

    # de verdad
    python -m genie_rover.Indoor.indoor_bridge --config configs/indoor_semantic_tour.yaml \
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
from ..console_report import MissionConsoleReporter
from .external_map import load_ros_occupancy_map
from .mission import (
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
    # La mision indoor no usa checkpoints GPS en absoluto -- pisa el default
    # de Bridge (True) para que Bridge.run() no los lea ni los anuncie.
    uses_gps_checkpoints = False

    # Atributos de CLASE (no solo de instancia): asi un IndoorBridge armado
    # con __new__(IndoorBridge) saltando __init__ (como hacen las pruebas
    # offline) sigue teniendo valores inertes en vez de romper _step() con
    # AttributeError. El publicador de clase esta deshabilitado, asi que
    # todos sus metodos salen temprano y compartirlo es inocuo.
    _rtab = None
    _route_status = RouteStatus(None, enabled=False)
    sem_cfg = None
    vlm = None
    _last_vlm_logged_t = 0.0

    def __init__(self, cfg: dict, dry_run: bool = True, debug_dir: str | None = None):
        super().__init__(cfg, dry_run=dry_run, debug_dir=debug_dir)

        if not self.use_map:
            raise ValueError(
                "IndoorBridge necesita memory.enabled: true — sin PersistentMap "
                "no hay ni memoria del pasillo ya visto ni una pose estable "
                "sobre la que apoyar el carril de cada tramo."
            )

        self._maybe_load_external_map(cfg.get("memory", {}))
        self._maybe_enable_rtabmap_correction(cfg)

        mission_raw = cfg.get("mission", {}) or {}
        self._build_mission(mission_raw, cfg)

        # --- ruta dibujada en el dashboard del SDK (localhost:8000) ---------
        # Publica el estado de la ruta a un JSON dentro de static/ del SDK,
        # que map.js lee cada 1.5 s para dibujarla ENCIMA de su mapa. Se
        # activa SOLO con dashboard.route_status_path en el config: sin eso
        # queda apagado, no escribe nada y el dashboard queda exactamente
        # como el original (no hay endpoint nuevo ni cambio en main.py del
        # SDK -- el archivo ES el canal y el interruptor).
        # Lo que se dibuja, si hay guia del mapeo previo, sale de
        # SemanticMissionFSM._route_hint (privado y puede ser None, de ahi
        # el getattr).
        dash_cfg = cfg.get("dashboard", {}) or {}
        route = getattr(self.mission, "_route_hint", None)
        self._route_status = RouteStatus(
            dash_cfg.get("route_status_path"),
            waypoints=getattr(route, "points", []),
            enabled=bool(dash_cfg.get("route_status_path")),
            anchor=dash_cfg.get("anchor"),
            reach_radius_m=float(dash_cfg.get("reach_radius_m", 0.6)),
            frame=("map" if self._rtab is not None else "odom"),
        )
        if self._route_status.enabled:
            print(f"[indoor_bridge] ruta publicada al dashboard en "
                  f"{dash_cfg['route_status_path']} "
                  f"({len(self._route_status.waypoints)} waypoints de guia)")

        # Estadistica propia, ademas de las de LoopStats (heredadas).
        self.mission_final_state = "RUN_SEGMENT"

        # Consola en tabla (ver console_report.py): reemplaza el par de lineas
        # sueltas por iteracion ("estado=..." + "[ENVIADO]/[DRY-RUN]...") por
        # una fila compacta, con los cambios de estado/tramo/eventos en su
        # propia linea aparte para que no se pierdan en el medio de la tabla.
        # mission.console_color: null/ausente = autodetectar (isatty), true/false
        # = forzar.
        self._reporter = MissionConsoleReporter(
            target_label="tramo",
            enable_color=mission_raw.get("console_color"))

    # ------------------------------------------------------------- la mision

    def _build_mission(self, mission_raw: dict, cfg: dict) -> None:
        """Arma self.mission (+ self.sem_cfg + self.vlm).

        Es un metodo aparte, y no codigo suelto dentro de __init__, para que
        map_session.py pueda cambiar la fuente de la meta (exploracion por
        frontera, sin VLM ni tramos) sin que este archivo tenga que volver a
        ramificar por "modo de mision".
        """
        self.sem_cfg = SemanticMissionConfig.from_dict(mission_raw)
        self.mission = SemanticMissionFSM(self.sem_cfg)

        self.vlm = SemanticVlm(VlmConfig.from_dict(cfg.get("vlm", {})))
        tramos = ", ".join(seg.id for seg in self.sem_cfg.segments)
        print(f"[indoor_bridge] mision semantica: {len(self.sem_cfg.segments)} "
              f"tramo(s) [{tramos}], VLM backend={self.vlm.cfg.backend}")
        if not self.vlm.enabled:
            print("[indoor_bridge] AVISO: el VLM esta apagado o sin credencial -- "
                  "ningun hito se va a confirmar nunca; cada tramo va a terminar "
                  "por su fail-safe de distancia (segments[].on_timeout).")

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
        en IndoorBridge (la mision real, `indoor-bridge`). Poner
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

        pose = self.odometry.update(telem.raw)
        self.pmap.integrate(res.traversability, res.observed, pose,
                            self.forward_range, self.side_range, t=now)
        plan_bev, plan_obs = self.pmap.extract_bev(
            pose, self.plan_forward_m, self.plan_side_m,
            *res.traversability.shape)
        fwd, side = self.plan_forward_m, self.plan_side_m
        st = self.pmap.stats()

        # Se calcula ACA (antes se calculaba recien al llamar a plan_on_bev):
        # la mision lo necesita para leer el centro del pasillo del mismo
        # BEV, antes de decidir la meta.
        bev_resolution_m = (2.0 * side) / plan_bev.shape[1]

        mission_goal = self._mission_goal(pose, now, rgb, plan_bev, plan_obs,
                                          bev_resolution_m)
        self.mission_final_state = mission_goal.state

        # Cambios de estado/tramo van a su propia linea, bien marcada --
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
            checkpoints_done=self.mission.checkpoints_done,
            state=str(getattr(self.mission.state, "value", self.mission.state)),
            mission_done=mission_goal.mission_done,
            pose_source=("rtabmap" if self._rtab is not None else "odometry"),
        )

        # Lo que importa de un vistazo es en que tramo va y que hito esta
        # esperando confirmar el VLM.
        consulta = self.mission.current_query()
        hito_desc = (f"{self.mission.segment.id}/"
                     f"{consulta.id if consulta is not None else '-'}")

        def _row(action: str) -> None:
            self._reporter.row(
                iteration=self.stats.iterations, state=mission_goal.state, pose=pose,
                target_desc=hito_desc, map_cells=st["celdas_vistas"],
                trav=res.traversability, action=action)

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
        self._maybe_dump_debug_indoor(rgb, res, plan)

    # ---------------------------------------------------------- meta del frame

    def _mission_goal(self, pose, now: float, rgb: np.ndarray,
                      plan_bev: np.ndarray, plan_obs: np.ndarray,
                      bev_resolution_m: float):
        """Meta de este frame.

        Junta las dos entradas de SemanticMissionFSM y la llama:

          * el VLM (semantico, lento, ~1.4 Hz, puede no haber respuesta),
            que solo decide SI el tramo termino;
          * el BEV (geometrico, todos los frames), que dice donde esta el
            centro del pasillo y cuanto despeje hay adelante.

        Si el VLM no contesto todavia, `obs` es None y el tramo sigue recto:
        es el comportamiento seguro, no un error.

        map_session.py hereda esto tal cual: su FSM de exploracion por
        frontera ignora `vlm` y `corridor` y usa `pmap`, con la misma firma.
        """
        consulta = self.mission.current_query()
        obs = None
        if consulta is not None and self.vlm is not None:
            # Durante un giro la latencia si importa (el robot esta pivoteando
            # a ciegas hasta que le confirmen la apertura), asi que ahi se
            # pregunta mas seguido.
            girando = self.mission.state == SemanticState.TURN_SEARCH
            # `pose` habilita el disparo por movimiento de vlm_semantic: no se
            # vuelve a consultar hasta que el robot avanzo query_every_m o
            # giro query_every_deg desde la consulta anterior. Es la MISMA
            # pose que usa la mision (ya corregida por RTAB-Map si esta
            # activo), no una medida aparte.
            obs = self.vlm.observe(rgb, consulta, now, urgent=girando, pose=pose)

        pasillo = corridor_hint_from_bev(
            plan_bev, plan_obs, bev_resolution_m,
            lookahead_m=self.sem_cfg.segment_lookahead_m,
            row0_is_far=self.sem_cfg.bev_row0_is_far,
        )
        meta = self.mission.update(pose, now, vlm=obs, corridor=pasillo,
                                    pmap=self.pmap)
        self._log_decision(obs)
        return meta

    def _log_decision(self, obs) -> None:
        """Que hizo la FSM con la ultima respuesta del VLM.

        vlm_semantic.py ya imprime TODA respuesta (`[vlm] ...`). Aca se
        imprime solo lo que mueve la aguja: un "si" del VLM y cuantas
        confirmaciones seguidas lleva acumuladas contra las que hacen falta
        para cambiar de fase. Asi, mirando la consola, se ve la cadena
        completa: imagen -> respuesta -> cuenta -> cambio de fase.
        """
        if obs is None or not obs.present:
            return
        if obs.t == self._last_vlm_logged_t:
            return                      # respuesta cacheada, ya la contamos
        self._last_vlm_logged_t = obs.t

        seg = self.mission.segment
        faltan = seg.milestone.confirm_hits
        if self.mission.state == SemanticState.TURN_SEARCH and seg.turn is not None:
            faltan = seg.turn.confirm_hits
        hits = getattr(self.mission, "_hits", 0)
        self._reporter.event(
            "VLM",
            f"'{obs.id}' confirmado (conf {obs.confidence:.2f}, {obs.position}): "
            f"{hits}/{faltan} confirmaciones seguidas"
            + ("  -> cambia de fase" if hits >= faltan else ""),
            "cyan" if hits < faltan else "green")

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

    # ------------------------------------------------------------------ debug

    def _maybe_dump_debug_indoor(self, rgb, res, plan) -> None:
        if not self.debug_dir:
            return
        try:
            from PIL import Image
            n = self.stats.iterations
            Image.fromarray(rgb).save(self.debug_dir / f"{n:05d}_rgb.jpg", quality=80)
            Image.fromarray(plan.visualization).save(self.debug_dir / f"{n:05d}_plan.png")
            np.save(self.debug_dir / f"{n:05d}_bev.npy", res.traversability)
            Image.fromarray(self.pmap.to_image(self.odometry.pose)).save(
                self.debug_dir / f"{n:05d}_mapa.png")
        except Exception as exc:
            print(f"[indoor_bridge] no pude escribir el debug: {exc}")

    def _print_summary(self) -> None:
        super()._print_summary()
        total = len(self.sem_cfg.segments) if self.sem_cfg is not None else 0
        print("  --- mision indoor (tramos + hitos visuales) ---")
        print(f"  tramos completados:        {self.mission.segments_done}/{total}")
        print(f"  tramo final:               {self.mission.segment.id}")
        print(f"  estado final:              {self.mission_final_state}")
        if self.vlm is not None:
            print(f"  VLM:                       {self.vlm.stats_line()}")
        if self.mission.failed_reason:
            print(f"  mision cumplida:           NO ({self.mission.failed_reason})")
        else:
            cumplida = self.mission.state == SemanticState.DONE
            print(f"  mision cumplida:           {'SI' if cumplida else 'NO'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="ej. configs/indoor_semantic_tour.yaml")
    ap.add_argument("--go", action="store_true",
                    help="enviar comandos de verdad (sin esto es simulacro)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--vlm-backend", choices=["gemini", "off"], default=None,
                    help="pisa vlm.backend del config. 'off' corre el recorrido "
                         "sin VLM: cada tramo termina por su fail-safe de "
                         "distancia (segments[].on_timeout)")
    ap.add_argument("--vlm-model", default=None,
                    help="pisa vlm.model del config (ej. gemini-2.5-flash)")
    ap.add_argument("--vlm-log-dir", default=None,
                    help="guarda el jpg mandado y el json contestado de CADA "
                         "consulta al VLM, para revisarlos despues")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    mission_cfg = cfg.setdefault("mission", {})
    if args.vlm_backend is not None:
        cfg.setdefault("vlm", {})["backend"] = args.vlm_backend
    if args.vlm_model is not None:
        cfg.setdefault("vlm", {})["model"] = args.vlm_model
    if args.vlm_log_dir is not None:
        cfg.setdefault("vlm", {})["log_dir"] = args.vlm_log_dir

    if not mission_cfg.get("segments"):
        raise SystemExit(
            "la mision indoor necesita mission.segments (la lista de tramos con "
            "su hito visual) -- ver configs/indoor_semantic_tour.yaml."
        )

    _check_placeholders(cfg)

    bridge = IndoorBridge(cfg, dry_run=not args.go, debug_dir=args.debug_dir)

    detalle = (f"  Tramos: {len(mission_cfg.get('segments', []))}  |  "
               f"VLM: {cfg.get('vlm', {}).get('model', 'gemini-2.0-flash')}")

    if args.go:
        print("\n" + "=" * 62)
        print("  MODO REAL: el rover se va a mover a recorrer los tramos "
              "confirmando hitos con el VLM.")
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
