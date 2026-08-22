"""MapSessionBridge: programa de mapeo reusable con genie_rover + (opcional) RTAB-Map.

Que hace, en una linea: explora por frontera (igual que IndoorBridge con
`mission.search_mode: "frontier"`, sin tocar esa logica) armando el
PersistentMap de siempre con la percepcion SAM-TP en vivo, y lo exporta
periodicamente + al terminar a formato ROS `map_server` (yaml+pgm) --
el mismo formato que ya sabe leer `external_map.load_ros_occupancy_map`
para que lo reuses despues con `indoor_bridge.py` (mision de cono) o
cualquier otro programa tuyo.

Opcionalmente (`mapping.rtabmap_correction.enabled: true` en el config),
reemplaza la pose de dead-reckoning (rueda+giro+GPS) por la pose corregida
que publica RTAB-Map (TF `map -> base_link`, cierre de bucles visual) --
para eso tiene que estar corriendo, EN PARALELO, la sesion ROS2 de
`examples/ros2/mapping/rtabmap_mapping.launch.py`. Si no esta corriendo, o
si rclpy/ROS2 no estan disponibles en este entorno, cae automaticamente a
usar solo rueda+giro (el comportamiento de siempre) e imprime un aviso una
sola vez -- nunca falla en silencio ni se cuelga esperando ROS2.

No modifica `indoor_bridge.py` ni `bridge.py` (misma politica que el resto
del repo: subclasear, no tocar lo ya validado).

Uso (mismo patron que bridge.py/indoor_bridge.py: dry-run por defecto):

    # simulacro, no mueve al robot
    python -m genie_rover.Indoor.Indoor.map_session --config configs/indoor_mapping.yaml

    # de verdad, con correccion de RTAB-Map (rtabmap_mapping.launch.py ya
    # corriendo en otra terminal)
    python -m genie_rover.Indoor.Indoor.map_session --config configs/indoor_mapping.yaml \\
        --go --max-seconds 300 --map-out maps/sesion1

Al terminar (tiempo agotado, Ctrl+C, o error) SIEMPRE intenta un export
final antes de salir, ademas de los exports periodicos durante la corrida.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from .indoor_bridge import IndoorBridge


class MapSessionBridge(IndoorBridge):
    def __init__(self, cfg: dict, dry_run: bool = True, debug_dir: str | None = None):
        super().__init__(cfg, dry_run=dry_run, debug_dir=debug_dir)

        map_cfg = cfg.get("mapping", {}) or {}
        self.map_out_prefix = str(map_cfg.get("map_out_prefix", "maps/session"))
        self.export_every_s = float(map_cfg.get("export_every_s", 30.0))
        self._last_export_t = time.time()
        self._exports_done = 0
        self._did_final_export = False

        self._rtab = None
        rtab_cfg = map_cfg.get("rtabmap_correction", {}) or {}
        if rtab_cfg.get("enabled", False):
            try:
                from .rtabmap_pose_bridge import RtabmapPoseBridge
                self._rtab = RtabmapPoseBridge(
                    map_frame=rtab_cfg.get("map_frame", "map"),
                    base_frame=rtab_cfg.get("base_frame", "base_link"),
                    lookup_timeout_s=float(rtab_cfg.get("lookup_timeout_s", 0.2)),
                )
                print("[map_session] correccion de RTAB-Map ACTIVADA "
                      "(TF map->base_link).")
            except Exception as exc:
                print(f"[map_session] AVISO: no pude activar la correccion de "
                      f"RTAB-Map ({exc}).\n"
                      f"[map_session] Sigo con odometria de rueda+giro sola "
                      f"(el comportamiento de siempre). Si esperabas la "
                      f"correccion, revisa que rtabmap_mapping.launch.py este "
                      f"corriendo y que este proceso tenga rclpy/tf2_ros "
                      f"instalados.")
                self._rtab = None

    # ------------------------------------------------------------- override

    def _step(self):
        if self._rtab is not None:
            corrected = self._rtab.get_corrected_pose()
            if corrected is not None:
                # Reemplaza la pose ENTERA (no solo para integrar el mapa):
                # asi PersistentMap.integrate() y PersistentMap.extract_bev()
                # (el planner) siguen viendo un unico marco consistente.
                self.odometry.pose = corrected

        result = super()._step()
        self._maybe_export(force=False)
        return result

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        finally:
            self._maybe_export(force=True)
            if self._rtab is not None:
                print(f"[map_session] TF lookups: {self._rtab.lookups_ok} ok, "
                      f"{self._rtab.lookups_failed} fallidos")
                self._rtab.shutdown()

    # --------------------------------------------------------------- export

    def _maybe_export(self, force: bool) -> None:
        if force and self._did_final_export:
            return
        now = time.time()
        if not force and (now - self._last_export_t) < self.export_every_s:
            return
        self._last_export_t = now
        if force:
            self._did_final_export = True

        suffix = "final" if force else f"{self._exports_done:03d}"
        prefix = f"{self.map_out_prefix}_{suffix}"
        try:
            yaml_path, img_path = self.pmap.export_ros_map(prefix)
        except AttributeError:
            print("[map_session] ERROR: PersistentMap no tiene export_ros_map(). "
                  "Aplicaste el parche de persistent_map_export_addon.py? "
                  "(ver genie/genie_rover/persistent_map_export_addon.py)")
            return
        except Exception as exc:
            print(f"[map_session] AVISO: fallo el export del mapa: {exc}")
            return

        self._exports_done += 1
        s = self.pmap.stats()
        print(f"[map_session] mapa exportado -> {yaml_path} "
              f"(cobertura={s['cobertura']*100:.1f}%, "
              f"celdas_vistas={s['celdas_vistas']})")


# ------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="ej. configs/indoor_mapping.yaml")
    ap.add_argument("--go", action="store_true",
                    help="mueve al robot de verdad (por defecto: dry-run)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="corta la sesion despues de N segundos")
    ap.add_argument("--debug-dir", default=None,
                    help="carpeta para volcar debug de cada frame (opcional)")
    ap.add_argument("--map-out", default=None,
                    help="prefijo de salida del mapa, pisa mapping.map_out_prefix del config")
    ap.add_argument("--export-every-s", type=float, default=None,
                    help="pisa mapping.export_every_s del config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    mapping_cfg = cfg.setdefault("mapping", {})
    if args.map_out is not None:
        mapping_cfg["map_out_prefix"] = args.map_out
    if args.export_every_s is not None:
        mapping_cfg["export_every_s"] = args.export_every_s

    bridge = MapSessionBridge(cfg, dry_run=not args.go, debug_dir=args.debug_dir)

    # NOTA: el nombre/firma exacto de run() (que argumentos acepta ademas de
    # max_seconds) depende de tu bridge.py real -- si difiere, ajusta esta
    # linea; el resto de MapSessionBridge no depende de como se invoque run().
    kwargs = {}
    if args.max_seconds is not None:
        kwargs["max_seconds"] = args.max_seconds
    bridge.run(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
