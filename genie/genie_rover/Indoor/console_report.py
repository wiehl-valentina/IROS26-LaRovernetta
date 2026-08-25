"""Salida de consola prolija para la mision indoor.

Antes, cada iteracion imprimia DOS lineas sueltas (el "estado=..." de
indoor_bridge.py y el "[ENVIADO]/[DRY-RUN]..." de Bridge.send()), identicas
en formato salvo por los numeros, y en una corrida larga (100+ iteraciones,
tipico en CPU) eso tapa la terminal entera sin que se pueda seguir la mision
de un vistazo.

Este modulo arma UNA fila de "tabla" compacta por iteracion (alineada,
con encabezado de columnas cada tantas filas), y saca todo lo que NO es
rutina -- un cambio de estado de la FSM, un checkpoint completado, un
evento (obstaculo, sin camino, mision cumplida) -- a su PROPIA linea bien
marcada, para que no se pierda en el medio de la tabla ni se confunda con
una fila mas.

La "barra de adelante" de cada fila reusa la misma idea de color que
`genie_path_planner.projection.traversability_vis` (verde = transitable,
rojo = no transitable/obstaculo, gris = nunca observado): un resumen de la
franja central del BEV (lo que hay justo delante del robot), de lejos
(izquierda) a cerca (derecha), para ver el terreno de un vistazo sin tener
que abrir ningun --debug-dir.

Los colores ANSI se autodetectan (se apagan solos si stdout no es una
terminal -- por ejemplo corriendo con `... > log.txt`, para no ensuciar el
archivo con codigos de escape) y se pueden forzar con la variable de
entorno NO_COLOR=1 (los apaga) o FORCE_COLOR=1 (los prende).
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

# Un color por estado de ConeMissionFSM, para reconocer la fase de un
# vistazo sin tener que leer la palabra.
_STATE_COLOR = {
    "SEARCH": "cyan", "NAVIGATE": "blue", "APPROACH": "yellow",
    "VERIFY": "magenta", "PHOTO": "green", "STOP": "green",
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

    Uso tipico (ver indoor_bridge.py):
        self._reporter = MissionConsoleReporter()
        ...
        self._reporter.state_change(mission_goal.state, mission_goal.reason)
        self._reporter.checkpoint(self.mission.checkpoints_done, razon)
        ...
        self._reporter.row(iteration=..., state=..., pose=..., cone_desc=...,
                           map_cells=..., trav=res.traversability, action=...)
    """

    HEADER_EVERY = 20  # repetir el encabezado cada N filas, para no perder
    # la referencia de columnas en una corrida larga sin taparla de entrada.

    def __init__(self, enable_color: bool | None = None, bar_width: int = 10):
        self.color = _auto_color_enabled() if enable_color is None else bool(enable_color)
        self.bar_width = max(4, int(bar_width))
        self._t0 = time.time()
        self._rows_since_header = 0
        self._prev_state: str | None = None
        self._prev_checkpoints = 0

    # ---------------------------------------------------------------- color

    def _c(self, text: str, color: str, bold: bool = False) -> str:
        if not self.color:
            return text
        return f"{_BOLD if bold else ''}{_ANSI.get(color, '')}{text}{_RESET}"

    def _elapsed(self) -> float:
        return time.time() - self._t0

    # ------------------------------------------------------- barra de piso

    def traversability_bar(self, trav: np.ndarray) -> str:
        """Resume la franja central del BEV (el corredor justo delante del
        robot) en self.bar_width bloques de color -- misma paleta que
        traversability_vis: verde alto/transitable, rojo bajo/obstaculo,
        gris nunca observado (-1). Primer bloque = mas lejos, ultimo =
        junto al robot.
        """
        if trav is None or trav.size == 0:
            return self._c("?" * self.bar_width, "gray")
        h, w = trav.shape[:2]
        c0, c1 = int(w * 0.4), int(w * 0.6)
        if c1 <= c0:
            c0, c1 = max(0, w // 2 - 1), min(w, w // 2 + 1)
        strip = trav[:, c0:c1]
        edges = np.linspace(0, h, self.bar_width + 1).astype(int)
        out = []
        for i in range(self.bar_width):
            r0, r1 = edges[i], max(edges[i] + 1, edges[i + 1])
            seg = strip[r0:r1]
            known = seg[seg >= 0.0]
            if known.size == 0:
                out.append(self._c("░", "gray"))
                continue
            val = float(known.mean())
            color = "green" if val >= 0.66 else ("yellow" if val >= 0.35 else "red")
            out.append(self._c("█", color))
        return "".join(out)

    # ------------------------------------------------------------- eventos

    def state_change(self, new_state: str, reason: str) -> None:
        """Imprime una linea aparte SOLO cuando la FSM realmente cambia de
        estado (SEARCH -> NAVIGATE, etc.) -- lo que antes quedaba enterrado
        como una palabra mas en medio de cada fila."""
        if self._prev_state is not None and new_state != self._prev_state:
            line = (f"[t+{self._elapsed():6.1f}s] === "
                    f"{self._prev_state} -> {new_state} === {reason}")
            print(self._c(line, _STATE_COLOR.get(new_state, "white"), bold=True))
            self._rows_since_header = 0  # reimprime el encabezado despues del corte visual
        self._prev_state = new_state

    def checkpoint(self, done: int, reason: str) -> None:
        if done != self._prev_checkpoints:
            print(self._c(f"[t+{self._elapsed():6.1f}s] *** checkpoint #{done} "
                          f"completado *** {reason}", "green", bold=True))
            self._prev_checkpoints = done
            self._rows_since_header = 0

    def event(self, tag: str, message: str, color: str = "yellow") -> None:
        """Para lo excepcional (obstaculo, mision cumplida, foto, error) --
        siempre en su propia linea, nunca mezclado en una fila de la tabla."""
        print(self._c(f"[t+{self._elapsed():6.1f}s] !! {tag}: {message}", color, bold=True))
        self._rows_since_header = 0

    # --------------------------------------------------------------- tabla

    def _header(self) -> None:
        cols = (f"{'iter':>5} {'t':>7}  {'estado':<9} {'pose (x,y,rot)':<19} "
                f"{'cono':<13} {'mapa':>8}  {'adelante':<{self.bar_width}}  accion")
        print(self._c(cols, "gray"))
        print(self._c("-" * min(len(cols), 100), "gray"))

    def row(self, iteration: int, state: str, pose, cone_desc: str,
           map_cells: int, trav: np.ndarray, action: str) -> None:
        if self._rows_since_header % self.HEADER_EVERY == 0:
            self._header()
        self._rows_since_header += 1

        pose_txt = f"{pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.theta):+.0f}gr"
        bar = self.traversability_bar(trav)
        state_txt = self._c(f"{state:<9}", _STATE_COLOR.get(state, "white"))
        action_trim = action if len(action) <= 42 else action[:39] + "..."
        print(f"{iteration:>5} {self._elapsed():6.1f}s  {state_txt} {pose_txt:<19} "
              f"{cone_desc:<13} {map_cells:>6}c  {bar}  {action_trim}")
