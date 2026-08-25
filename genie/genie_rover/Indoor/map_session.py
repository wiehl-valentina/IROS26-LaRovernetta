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

IMPORTANTE (esto cambio): esa correccion de RTAB-Map ahora vive en
`IndoorBridge` (`_maybe_enable_rtabmap_correction()`, llamada desde
`IndoorBridge.__init__`, y aplicada al principio de `IndoorBridge._step()`),
no aca. Antes MapSessionBridge la duplicaba por completo -- armaba su PROPIO
`RtabmapPoseBridge` en `__init__` (pisando el que ya habia armado
`IndoorBridge.__init__`, dejando el primero huerfano: un nodo ROS2 + un hilo
de spin que nadie volvia a cerrar nunca) y la volvia a aplicar en `_step()`
(una segunda consulta TF por frame, redundante con la que ya hacia
`IndoorBridge._step()`). MapSessionBridge ya NO instancia `RtabmapPoseBridge`
ni toca `self.odometry.pose` directamente: solo usa `self._rtab` (heredado)
para imprimir las estadisticas finales de lookups. El resto de la politica
del repo se mantiene: esta clase sigue sin tocar la logica de exploracion/
mision de IndoorBridge, solo agrega el export periodico a formato ROS.

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

        # self._rtab ya lo arma IndoorBridge.__init__ (via
        # _maybe_enable_rtabmap_correction), leyendo la misma seccion
        # mapping.rtabmap_correction del config que este __init__ leia antes
        # a mano -- no lo repetimos aca. Instanciar un segundo
        # RtabmapPoseBridge en este punto pisaba self._rtab con una segunda
        # conexion ROS2 y dejaba la primera (la de IndoorBridge.__init__)
        # sin cerrar nunca: un nodo + un hilo de spin huerfanos corriendo en
        # segundo plano el resto del proceso.

    # ------------------------------------------------------------- override

    def _step(self):
        # La correccion de pose por RTAB-Map (cuando self._rtab esta activo)
        # ya la aplica IndoorBridge._step(), al principio, antes de que
        # odometry.update() integre el delta de este frame -- exactamente el
        # mismo mecanismo que este archivo implementaba antes por su cuenta.
        # Repetirla aca era una segunda consulta TF por frame que solo
        # duplicaba trabajo (y sumaba de mas a lookups_ok/lookups_failed)
        # sin cambiar el resultado.
        result = super()._step()
        self._maybe_export(force=False)
        return result

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        finally:
            self._maybe_export(force=True)
            if self._rtab is not None:
                # super().run() (IndoorBridge.run()) ya cerro self._rtab en
                # su propio finally -- eso corre ANTES de llegar aca, porque
                # queda "adentro" del super().run() de arriba. Esto solo
                # imprime las estadisticas finales (shutdown() no las borra),
                # no lo vuelve a cerrar.
                print(f"[map_session] TF lookups: {self._rtab.lookups_ok} ok, "
                      f"{self._rtab.lookups_failed} fallidos")

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

    bridge.run(max_seconds=args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
