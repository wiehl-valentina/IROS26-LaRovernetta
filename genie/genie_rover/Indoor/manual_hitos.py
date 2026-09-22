"""Hitos confirmados A MANO con Enter, para ensayar el recorrido sin el lugar.

Reemplaza al detector de fotos (hito_detector.py) con el mismo contrato:
`observe()` devuelve una VlmObservation o None, nunca bloquea y nunca lanza.
La diferencia es QUIEN decide que el hito esta: aca sos vos apretando Enter.

Sirve para probar en tu casa todo lo que NO es reconocer imagenes: el orden
de los tramos, que cada giro sea para el lado correcto, que TURN_ALIGN centre
en el pasillo, los fail-safes de distancia, el overlay del dashboard.

Como se usa:

    python -m genie_rover.Indoor.indoor_bridge \\
        --config configs/indoor_hitos.yaml --vlm-backend manual [--go]

La consola avisa que hito se esta buscando. Cuando "lo verias" (ej. llegaste
al lugar de tu casa que hace de piano), apreta Enter:

  * en RUN_SEGMENT confirma el hito del tramo -> el rover gira (si el tramo
    tiene `on_milestone`) o re-bloquea el rumbo y sigue recto;
  * en TURN_SEARCH confirma la apertura -> corta el giro y pasa a alinear.

Un Enter vale para UN hito: queda confirmando hasta que la mision lo acepta
(hacen falta `confirm_hits` seguidas y respetar `milestone_cooldown_s`) y
despues se apaga solo. Si lo apretas cuando no se busca nada (alineacion
fina), se descarta en vez de adelantar el hito siguiente.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .mission import VlmObservation, VlmQuery


@dataclass
class ManualHitosConfig:
    backend: str = "manual"
    # Si la mision no acepta el Enter en este tiempo (ej. el cooldown es mas
    # largo), se descarta para no confirmar un hito que ya no es el que
    # tenias en mente.
    press_timeout_s: float = 8.0


class ManualHitos:
    """Detector de hitos manejado por teclado."""

    def __init__(self, cfg: ManualHitosConfig | None = None):
        self.cfg = cfg or ManualHitosConfig()
        self.calls = 0
        self.presses = 0
        self.confirmados = 0

        self._lock = threading.Lock()
        self._pendiente = False
        self._pendiente_t = 0.0
        self._objetivo: str | None = None     # query activa al apretar Enter
        self._ultima_query: VlmQuery | None = None
        self._ultimo_anuncio: str | None = None
        self._cerrado = False

        self._hilo = threading.Thread(target=self._leer_teclado, daemon=True,
                                      name="manual_hitos")
        self._hilo.start()
        print("[manual] hitos A MANO: apreta Enter cuando el rover 'vea' el "
              "hito que se esta buscando. Ctrl-C frena.")

    # ------------------------------------------------------------- interfaz

    @property
    def enabled(self) -> bool:
        return True

    def observe(self, rgb: np.ndarray, query: VlmQuery, now: float,
                urgent: bool = False, pose=None) -> VlmObservation | None:
        self.calls += 1
        with self._lock:
            self._ultima_query = query
            if query.id != self._ultimo_anuncio:
                if self._pendiente and self._objetivo == self._ultimo_anuncio:
                    # La mision acepto el Enter y ya cambio de hito.
                    self.confirmados += 1
                    self._pendiente = False
                self._ultimo_anuncio = query.id
                print(f"\n[manual] >>> buscando '{query.id}': {query.prompt}"
                      f"\n[manual]     Enter = lo vi")

            if not self._pendiente:
                return None
            if self._objetivo is None:
                self._objetivo = query.id
            if query.id != self._objetivo:
                print(f"[manual] Enter descartado: era para '{self._objetivo}' "
                      f"y ahora se busca '{query.id}'")
                self._pendiente = False
                return None
            if (time.monotonic() - self._pendiente_t) > self.cfg.press_timeout_s:
                print(f"[manual] Enter descartado: la mision no confirmo "
                      f"'{query.id}' en {self.cfg.press_timeout_s:.0f} s")
                self._pendiente = False
                return None

        # Una observacion NUEVA por frame (t distinto): la FSM cuenta cada una
        # como una confirmacion seguida hasta llegar a confirm_hits.
        return VlmObservation(id=query.id, present=True, confidence=1.0,
                              position="centro", reason="Enter", t=now)

    def stats_line(self) -> str:
        # `confirmados` no cuenta el ultimo hito: al terminar la mision no hay
        # query nueva que avise que fue aceptado.
        return f"manual: {self.presses} Enter apretados"

    def close(self) -> None:
        self._cerrado = True

    # ------------------------------------------------------------- teclado

    def _leer_teclado(self) -> None:
        while not self._cerrado:
            linea = sys.stdin.readline()
            if linea == "":                   # stdin cerrado (no hay terminal)
                print("[manual] AVISO: no hay teclado (stdin cerrado); los "
                      "hitos solo van a terminar por fail-safe de distancia.")
                return
            with self._lock:
                self.presses += 1
                q = self._ultima_query
                if self._pendiente:
                    print("[manual] ya hay un Enter esperando confirmacion")
                    continue
                self._pendiente = True
                self._pendiente_t = time.monotonic()
                # Se fija contra que hito vale en el proximo observe(): si en
                # este momento no se busca nada (TURN_ALIGN), la query que
                # llegue va a ser la del tramo siguiente y se descarta.
                self._objetivo = q.id if q is not None else None
                print(f"[manual] Enter -> confirmando "
                      f"'{self._objetivo or '(el proximo hito)'}'")
