#!/usr/bin/env python3
"""test_route_secondary.py — La ruta grabada (genie_rover.route) es guia
secundaria entre checkpoints oficiales: nunca se reclama en el SDK.

Verifica:
1. Sin ruta cargada, Bridge._meta_con_ruta devuelve la meta oficial intacta.
2. Con ruta cargada y el oficial lejos, la meta local sale de la ruta
   (RouteFollower.objetivo()), no del checkpoint oficial.
3. Con el oficial cerca (< oficial_directo_m), se va directo a el aunque
   haya ruta cargada.
4. En ningun caso _meta_con_ruta llama a client.claim_checkpoint(): decidir
   si algo se "reclama" es responsabilidad exclusiva del bloque de
   checkpoint oficial en _step, ajeno a la ruta.
"""

import types
import unittest

from genie_rover.bridge import Bridge
from genie_rover.route import RoutePoint, RouteFollower


def _stub_con_ruta(ruta, oficial_directo_m=15.0, ruta_abandono_m=15.0):
    stub = types.SimpleNamespace()
    stub.ruta = ruta
    stub.dashboard = None
    stub._dash_state = "arrancando"
    stub._dash_target = None
    stub._route_stats = {"metas_ruta": 0, "metas_oficial": 0, "ruta_ignorada": 0}
    stub._en_directo = False
    stub.oficial_directo_m = oficial_directo_m
    stub.ruta_abandono_m = ruta_abandono_m
    stub.goal_range_m = 3.5
    stub._last_goal = None
    for name in ["_meta_con_ruta", "_directo_al_oficial"]:
        setattr(stub, name, types.MethodType(getattr(Bridge, name), stub))
    return stub


class TestRutaSecundaria(unittest.TestCase):
    def setUp(self):
        # Ruta corta de 3 puntos, unos metros al norte del origen.
        self.origen = (9.791500, -84.105900)
        puntos = [
            RoutePoint(self.origen[0], self.origen[1], "inicio"),
            RoutePoint(self.origen[0] + 5.0 / 111194.9, self.origen[1], "medio"),
            RoutePoint(self.origen[0] + 10.0 / 111194.9, self.origen[1], "fin"),
        ]
        self.ruta = RouteFollower(puntos)

        self.target = types.SimpleNamespace(sequence=1, latitude=9.795000, longitude=-84.105900)
        self.oficial_goal = types.SimpleNamespace(distance_m=400.0, relative_bearing_deg=0.0)
        self.guard_status = types.SimpleNamespace(
            effective_lat=self.origen[0], effective_lon=self.origen[1],
            can_claim_checkpoints=True,
        )

    def test_sin_ruta_pasa_la_meta_oficial_intacta(self):
        stub = _stub_con_ruta(ruta=None)
        goal_oficial = object()
        goal, desc = stub._meta_con_ruta(self.guard_status, 0.0, self.target,
                                         self.oficial_goal, goal_oficial, "cp#1 a 400 m")
        self.assertIs(goal, goal_oficial)
        self.assertEqual(desc, "cp#1 a 400 m")

    def test_con_ruta_y_oficial_lejos_sigue_la_ruta(self):
        stub = _stub_con_ruta(ruta=self.ruta, oficial_directo_m=15.0)
        goal_oficial = types.SimpleNamespace(distance_m=400.0, relative_bearing_deg=0.0)
        goal, desc = stub._meta_con_ruta(self.guard_status, 0.0, self.target,
                                         goal_oficial, goal_oficial, "cp#1 a 400 m")
        self.assertIsNot(goal, goal_oficial)
        self.assertEqual(stub._dash_target["kind"], "ruta")
        self.assertEqual(stub._route_stats["metas_ruta"], 1)
        self.assertEqual(stub._route_stats["metas_oficial"], 0)

    def test_con_oficial_cerca_va_directo_e_ignora_la_ruta(self):
        stub = _stub_con_ruta(ruta=self.ruta, oficial_directo_m=15.0)
        goal_oficial = types.SimpleNamespace(distance_m=5.0, relative_bearing_deg=0.0)
        goal, desc = stub._meta_con_ruta(self.guard_status, 0.0, self.target,
                                         goal_oficial, goal_oficial, "cp#1 a 5 m")
        self.assertIs(goal, goal_oficial)
        self.assertEqual(stub._dash_target["kind"], "oficial")
        self.assertEqual(stub._route_stats["metas_ruta"], 0)
        self.assertEqual(stub._route_stats["metas_oficial"], 1)

    def test_nunca_reclama_checkpoint_por_la_ruta(self):
        """_meta_con_ruta no tiene ninguna via para llamar a claim_checkpoint:
        solo lee guard_status y el follower de ruta, nunca un client del SDK."""
        stub = _stub_con_ruta(ruta=self.ruta, oficial_directo_m=15.0)
        stub.client = types.SimpleNamespace(
            claim_checkpoint=lambda: (_ for _ in ()).throw(
                AssertionError("la ruta secundaria no debe reclamar checkpoints")))
        goal_oficial = types.SimpleNamespace(distance_m=400.0, relative_bearing_deg=0.0)
        # Si _meta_con_ruta tuviera alguna via para reclamar, el
        # claim_checkpoint stub de arriba haria fallar el test.
        for _ in range(20):
            stub._meta_con_ruta(self.guard_status, 0.0, self.target,
                                goal_oficial, goal_oficial, "cp#1 a 400 m")


if __name__ == "__main__":
    unittest.main()
