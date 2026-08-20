"""Prueba el regimen cercano (_retroceso_y_recover y sus partes) del bridge
REAL, sin SDK, sin rover, sin SAM-TP/torch -- solo PersistentMap y Odometry,
que son numpy puro.

No inventa una copia de la logica: llama a los metodos de verdad de la clase
Bridge (Bridge._map_free_and_coverage, Bridge._retroceder,
Bridge._girar_hacia, Bridge._recover_informado) sobre un objeto minimo que
imita los atributos que esos metodos tocan, con un PersistentMap sintetico
donde SE SABE de antemano donde esta libre y donde no.

Uso:
    python -m genie_rover.test_near_regime_offline
"""

from __future__ import annotations

import math
import types

import numpy as np

from .bridge import Bridge
from .navigation import DriveCommand, PathFollower
from .odometry import Pose
from .persistent_map import MapConfig, PersistentMap


def _mapa_con_obstaculo_adelante() -> PersistentMap:
    """Mapa 8x8 m centrado en el origen: todo transitable y bien observado,
    salvo una franja justo delante del robot (mundo +x, que es donde mira el
    robot con theta=0). Atras y a los costados queda libre."""
    pmap = PersistentMap(MapConfig(size_m=8.0, resolution_m_per_px=0.03))
    pmap.value[:] = 1.0
    pmap.conf[:] = 1.0
    for x in np.arange(0.05, 0.65, pmap.cfg.resolution_m_per_px):
        for y in np.arange(-0.3, 0.3, pmap.cfg.resolution_m_per_px):
            f, c = pmap.world_to_cell(float(x), float(y))
            if 0 <= f < pmap.n and 0 <= c < pmap.n:
                pmap.value[f, c] = 0.0
    return pmap


def _build_stub(pmap: PersistentMap):
    stub = types.SimpleNamespace()
    stub.resolution = 0.03
    stub.follower = PathFollower(angular_sign=-1.0, turn_speed=0.35,
                                 max_linear=0.35, max_angular=0.45)
    stub.retroceso_max_m = 0.4
    stub.retroceso_paso_m = 0.2
    stub.retroceso_linear = -0.18
    stub.retroceso_min_libre_pct = 55.0
    stub.retroceso_min_cobertura_pct = 30.0
    stub.recovery_headings_deg = [0.0, 90.0, -90.0, 180.0]
    stub.recovery_min_cobertura_pct = 25.0
    # Chico a proposito: si el radio de chequeo es mucho mas grande que el
    # obstaculo sintetico (0.6x0.6 m), el promedio lo diluye y todo sale
    # "libre" de entrada -- no probaria nada. Con un radio del orden del
    # obstaculo, el bloqueo pesa lo suficiente como para que el test discrimine.
    stub.heading_search_radius_m = 0.5
    stub.use_vlm_recovery = False
    stub.stats = types.SimpleNamespace(
        retrocesos=0, recoveries_por_mapa=0, recoveries_por_vlm=0,
        recoveries_ciegas=0, near_regime_activations=0,
    )
    stub._stop_requested = False
    stub.pmap = pmap

    stub.heading_est = types.SimpleNamespace(reset_track=lambda: None)
    stub._consecutive_turns = 0
    stub._turn_sign_history = []
    stub._consecutive_empty = 0
    stub._commit_side = 0
    stub._consecutive_blocked = 3
    stub._plan_path_world = np.zeros((5, 2))
    stub._plan_pose = Pose(0.0, 0.0, 0.0)

    pose_holder = {"pose": Pose(0.0, 0.0, 0.0)}

    class _Odo:
        @property
        def pose(self):
            return pose_holder["pose"]

        def update(self, _raw):
            return pose_holder["pose"]

    stub.odometry = _Odo()

    sent: list[DriveCommand] = []
    stub.send = lambda cmd: sent.append(cmd)
    stub._sent = sent

    class _Client:
        def telemetry(self):
            return types.SimpleNamespace(raw={})

        def front_frame(self):
            return np.zeros((8, 8, 3), dtype=np.uint8), 0.0

    stub.client = _Client()

    class _Perception:
        def process(self, _rgb):
            # BEV totalmente libre: simula que, tras retroceder o girar, la
            # camara ya deja de ver el obstaculo.
            return types.SimpleNamespace(traversability=np.ones((16, 16), dtype=np.float32))

    stub.perception = _Perception()

    # _girar_hacia y _recover_informado llaman a otros metodos de Bridge por
    # self (self._map_free_and_coverage, self._girar_hacia, etc). El stub no
    # es una instancia real de Bridge, asi que hay que atarlos a mano para
    # que esas llamadas cruzadas encuentren algo.
    for name in ("_map_free_and_coverage", "_girar_hacia", "_barrido_ciego",
                "_preguntar_vlm", "_retroceder", "_recover_informado",
                "_retroceso_y_recover"):
        setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))

    return stub


def _self_test() -> None:
    pmap = _mapa_con_obstaculo_adelante()
    stub = _build_stub(pmap)
    pose0 = Pose(0.0, 0.0, 0.0)

    print("=== _map_free_and_coverage: discrimina adelante (bloqueado) de atras/costados (libre) ===")
    libre_adelante, cob_adelante = stub._map_free_and_coverage(pose0, 0.0, 0.5)
    libre_atras, cob_atras = stub._map_free_and_coverage(pose0, 180.0, 0.5)
    libre_derecha, cob_derecha = stub._map_free_and_coverage(pose0, 90.0, 0.5)
    print(f"  adelante (0):    libre={libre_adelante:.0f}% cobertura={cob_adelante:.0f}%")
    print(f"  atras (180):     libre={libre_atras:.0f}% cobertura={cob_atras:.0f}%")
    print(f"  derecha (90):    libre={libre_derecha:.0f}% cobertura={cob_derecha:.0f}%")
    assert libre_adelante < 50.0, "el obstaculo sintetico no se detecto adelante"
    assert libre_atras > 80.0 and libre_derecha > 80.0, "atras/costados deberian salir libres"

    print("\n=== _retroceder: corta cuando el frente (fresco) da libre ===")
    stub._retroceder()
    print(f"  comandos enviados: {len(stub._sent)}")
    assert any(c.linear < 0 for c in stub._sent), "no llego a mandar ningun comando de reversa"
    assert stub._sent[-1] == DriveCommand(0.0, 0.0, "fin del retroceso") or stub._sent[-1].linear == 0.0
    stub._sent.clear()

    print("\n=== _girar_hacia: gira en pasos y para al llegar al objetivo ===")
    stub._girar_hacia(40.0, step_deg=20.0)
    print(f"  comandos enviados: {len(stub._sent)}")
    angulares = [c for c in stub._sent if c.angular != 0.0]
    assert len(angulares) >= 2, "deberia haber girado en mas de un paso"
    assert all(c.angular < 0 for c in angulares), \
        "con angular_sign=-1 un heading positivo (derecha) tiene que dar angular negativo"
    stub._sent.clear()

    print("\n=== _recover_informado: elige un rumbo libre por mapa, no el bloqueado ===")
    stub._recover_informado()
    print(f"  recoveries_por_mapa={stub.stats.recoveries_por_mapa} "
          f"recoveries_ciegas={stub.stats.recoveries_ciegas}")
    assert stub.stats.recoveries_por_mapa == 1, "deberia haber encontrado un rumbo por mapa"
    assert stub.stats.recoveries_ciegas == 0, "no deberia haber caido al barrido ciego"

    print("\n=== _retroceso_y_recover: el orquestador completo no rompe y limpia el estado ===")
    stub.stats.recoveries_por_mapa = 0
    stub._retroceso_y_recover(np.ones((16, 16), dtype=np.float32))
    print(f"  near_regime_activations={stub.stats.near_regime_activations} "
          f"retrocesos={stub.stats.retrocesos} recoveries_por_mapa={stub.stats.recoveries_por_mapa}")
    assert stub.stats.near_regime_activations == 1
    assert stub._consecutive_blocked == 0, "tiene que resetear el contador de bloqueo"
    assert stub._plan_path_world is None, "tiene que invalidar el plan cacheado"

    print("\nTodos los asserts pasaron.")


if __name__ == "__main__":
    _self_test()
