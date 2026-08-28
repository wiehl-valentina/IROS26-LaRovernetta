"""Salida de consola prolija y comun para las misiones de bridge.py.

Usada por AMBAS misiones:

  - Bridge (mision GPS, "bridge padre" / outdoor): checkpoints por
    lat/lon, `python -m genie_rover.bridge`.
  - IndoorBridge / MapSessionBridge (mision de cono indoor, sin GPS):
    `python -m genie_rover.Indoor.indoor_bridge`.

Antes cada iteracion imprimia DOS lineas sueltas (el "estado=..." /
"rumbo=..." de turno segun el bridge, y el "[ENVIADO]/[DRY-RUN]..." de
Bridge.send()), identicas en formato salvo por los numeros. En una corrida
larga (100+ iteraciones, tipico en CPU) eso tapa la terminal entera sin que
se pueda seguir la mision de un vistazo, y un cambio de estado real quedaba
enterrado como una palabra mas en medio del ruido.

Este modulo arma UNA fila de "tabla" compacta por iteracion (alineada, con
encabezado de columnas cada tantas filas), y saca todo lo que NO es rutina
-- un cambio de estado, un checkpoint completado, un evento (obstaculo, sin
camino, mision cumplida, recuperacion) -- a su PROPIA linea bien marcada,
para que no se pierda en el medio de la tabla ni se confunda con una fila
mas.

La "barra de adelante" de cada fila resume la franja central del BEV (lo
que hay justo delante del robot) en bloques solidos de color, de lejos
(izquierda) a cerca (derecha), para ver el terreno de un vistazo sin tener
que abrir ningun --debug-dir. Solo 3 estados: verde `█` transitable, rojo
`█` no transitable, gris `░` nunca observado.

En vez de mostrar la pose cruda todo el tiempo (poco util de un vistazo),
la columna de "meta" muestra que tan lejos esta el objetivo actual --
checkpoint GPS (distancia + rumbo relativo) o cono indoor (confianza +
distancia aproximada) -- que es lo que en la practica importa para seguir
la mision. La pose completa (x, y, rotacion) sigue disponible: se imprime
en la fila igual, pero angosta, y con su propio ancho recortable.

Los colores ANSI se autodetectan (se apagan solos si stdout no es una
terminal -- por ejemplo corriendo con `... > log.txt`, para no ensuciar el
archivo con codigos de escape) y se pueden forzar con la variable de
entorno NO_COLOR=1 (los apaga) o FORCE_COLOR=1 (los prende). Tambien se
puede pisar por config con `mission.console_color: true/false`.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_RESET = "\033[0m"
_BOLD = "\033[1m"
_ANSI = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "gray": "\033[90m", "white": "\033[97m",
}

# Un color por estado, para reconocer la fase de un vistazo sin tener que
# leer la palabra. Cubre tanto los estados de ConeMissionFSM (mision
# indoor) como los estados sencillos que reporta Bridge (mision GPS).
_STATE_COLOR = {
    # ConeMissionFSM (indoor)
    "SEARCH": "cyan", "NAVIGATE": "blue", "APPROACH": "yellow",
    "VERIFY": "magenta", "PHOTO": "green", "STOP": "green",
    # Bridge (GPS, mision "padre")
    "SIN_META": "gray", "NAVEGANDO": "blue", "CERCA_CHECKPOINT": "yellow",
    "OBSTACULO": "red", "SIN_CAMINO": "yellow", "RECUPERANDO": "magenta",
    "DESATASCO": "magenta", "FRENADO": "red",
}


def _auto_color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


class MissionConsoleReporter:
    """Une lo que antes eran 2+ prints sueltos por iteracion en una tabla.

    Uso tipico -- mision GPS (ver bridge.py, Bridge._step):
        self._reporter = MissionConsoleReporter(target_label="checkpoint")
        ...
        self._reporter.state_change("NAVEGANDO", goal_desc)
        self._reporter.checkpoint(target.sequence, msg)
        ...
        self._reporter.row(iteration=self.stats.iterations, state="NAVEGANDO",
                           pose=self.odometry.pose if self.odometry else None,
                           target_desc=goal_desc, map_cells=map_cells,
                           trav=res.traversability, action=cmd.reason)

    Uso tipico -- mision indoor (ver indoor_bridge.py):
        self._reporter = MissionConsoleReporter(target_label="cono")
        ...
        self._reporter.state_change(mission_goal.state, mission_goal.reason)
        self._reporter.checkpoint(self.mission.checkpoints_done, razon)
        ...
        self._reporter.row(iteration=..., state=..., pose=..., target_desc=...,
                           map_cells=..., trav=res.traversability, action=...)
    """

    HEADER_EVERY = 20  # repetir el encabezado cada N filas, para no perder
    # la referencia de columnas en una corrida larga sin taparla de entrada.

    BAR_WIDTH = 12  # bloques de color de la "franja de adelante"

    def __init__(self, target_label: str = "meta", enable_color: bool | None = None,
                 bar_width: int | None = None):
        self.target_label = target_label
        self.color = _auto_color_enabled() if enable_color is None else bool(enable_color)
        self.bar_width = int(bar_width) if bar_width else self.BAR_WIDTH

        self._t0 = time.time()
        self._prev_state: str | None = None
        self._prev_checkpoints: int | None = None
        self._rows_since_header = 0

    # ------------------------------------------------------------- helpers

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _c(self, text: str, color: str, bold: bool = False) -> str:
        if not self.color:
            return text
        code = _ANSI.get(color, "")
        prefix = (_BOLD if bold else "") + code
        return f"{prefix}{text}{_RESET}" if prefix else text

    # ------------------------------------------------------- eventos aparte

    def state_change(self, new_state: str, reason: str) -> None:
        """Linea aparte, en negrita y color, SOLO cuando el estado
        realmente cambia -- lo que antes quedaba enterrado como una palabra
        mas en medio de cada fila."""
        if self._prev_state is not None and new_state != self._prev_state:
            line = (f"[t+{self._elapsed():6.1f}s] === "
                    f"{self._prev_state} -> {new_state} === {reason}")
            print(self._c(line, _STATE_COLOR.get(new_state, "white"), bold=True))
            self._rows_since_header = 0  # reimprime el encabezado despues del corte visual
        self._prev_state = new_state

    def checkpoint(self, done, reason: str) -> None:
        """`done` es cualquier valor que aumente cuando se completa un
        checkpoint (contador o numero de secuencia) -- solo importa que
        cambie respecto de la llamada anterior."""
        if done != self._prev_checkpoints:
            print(self._c(f"[t+{self._elapsed():6.1f}s] *** {self.target_label} "
                          f"{done} completado *** {reason}", "green", bold=True))
            self._prev_checkpoints = done
            self._rows_since_header = 0

    def event(self, tag: str, message: str, color: str = "yellow") -> None:
        """Para lo excepcional (obstaculo, mision cumplida, foto, error) --
        siempre en su propia linea, nunca mezclado en una fila de la tabla."""
        print(self._c(f"[t+{self._elapsed():6.1f}s] !! {tag}: {message}", color, bold=True))
        self._rows_since_header = 0

    # --------------------------------------------------------------- barra

    def traversability_bar(self, trav: np.ndarray | None) -> str:
        """Resume la franja central del BEV (el corredor justo delante del
        robot) en bloques solidos de color -- misma idea que
        `genie_path_planner.projection.traversability_vis`, pero con solo 3
        estados (sin "intermedio"): verde `█` transitable, rojo `█` no
        transitable, gris `░` nunca observado. De lejos (izquierda) a cerca
        (derecha)."""
        if trav is None or trav.size == 0:
            return self._c("░" * self.bar_width, "gray")

        h, w = trav.shape[:2]
        # Franja central angosta (mismo ancho de corredor que usa el chequeo
        # de colision), de la fila mas lejana a la mas cercana.
        c0, c1 = max(0, w // 2 - w // 6), min(w, w // 2 + w // 6)
        strip = trav[:, c0:c1] if c1 > c0 else trav
        edges = np.linspace(0, h, self.bar_width + 1).astype(int)

        chars = []
        for i in range(self.bar_width):
            r0, r1 = edges[i], max(edges[i] + 1, edges[i + 1])
            band = strip[r0:r1]
            known = band[band >= 0]
            if known.size == 0:
                chars.append(self._c("░", "gray"))
                continue
            v = float(known.mean())
            color = "green" if v >= 0.5 else "red"
            chars.append(self._c("█", color))
        return "".join(chars)

    # --------------------------------------------------------------- tabla

    def _header(self) -> None:
        cols = (f"{'iter':>5} {'t':>7}  {'estado':<12} {self.target_label:<20} "
                f"{'mapa':>8}  {'adelante':<{self.bar_width}}  accion")
        print(self._c(cols, "gray"))
        print(self._c("-" * min(len(cols), 100), "gray"))

    def row(self, iteration: int, state: str, pose, target_desc: str,
           map_cells: int | None, trav: np.ndarray | None, action: str) -> None:
        if self._rows_since_header % self.HEADER_EVERY == 0:
            self._header()
        self._rows_since_header += 1

        bar = self.traversability_bar(trav)
        state_txt = self._c(f"{state:<12}", _STATE_COLOR.get(state, "white"))
        target_txt = target_desc if len(target_desc) <= 20 else target_desc[:17] + "..."
        map_txt = "-" if map_cells is None else f"{map_cells:>6}c"
        action_trim = action if len(action) <= 42 else action[:39] + "..."

        pose_suffix = ""
        if pose is not None:
            pose_suffix = (f"  pose=({pose.x:+.1f},{pose.y:+.1f},"
                           f"{math.degrees(pose.theta):+.0f}gr)")

        print(f"{iteration:>5} {self._elapsed():6.1f}s  {state_txt} {target_txt:<20} "
              f"{map_txt:>7}  {bar}  {action_trim}{pose_suffix}")
