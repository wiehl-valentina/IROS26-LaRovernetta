"""Eleccion de rumbo de escape y giro a lazo cerrado.

Destino: `genie/genie_rover/escape.py` (archivo nuevo). Reemplaza la logica
que hoy vive inline en `Bridge._recover_informado` y `Bridge._girar_hacia`.

POR QUE EXISTE
--------------
Dos bugs medidos sobre el log del 08/09 (216 iteraciones, 5.34 m recorridos).

**1. La eleccion de rumbo elige la pared.** `_recover_informado` hace:

    if cobertura_pct >= recovery_min_cobertura_pct and libre_pct > mejor_libre:

es decir, descarta cualquier sector con poca cobertura y despues compara
`libre_pct` crudos. Adelante SIEMPRE tiene cobertura alta -- es lo unico que
la camara mira -- asi que adelante siempre entra al concurso y los costados
casi nunca. Resultado: de 50 decisiones de escape, 43 no eligieron el sector
mas libre y 37 eligieron +0 grados, que es justo el rumbo que acababa de
bloquearse. Bloque real de la iteracion 0064:

    rumbo   +0 grados: libre=10% cobertura=74%   <- elegido
    rumbo  +90 grados: libre=24% cobertura=15%   <- descartado por cobertura
    rumbo  -90 grados: libre=13% cobertura=24%   <- descartado por cobertura
    rumbo +180 grados: libre= 0% cobertura= 0%

Aca la cobertura deja de ser un filtro y pasa a ser un PESO: cada sector se
puntua con una media entre lo que se observo y un prior de "no se". Con eso
un sector desconocido vale mas que uno que ya demostro estar bloqueado, que
es la conducta que falta. Y el rumbo bloqueado queda descalificado de una.

**2. Los giros en el lugar no giran.** `_girar_hacia` es lazo abierto:
manda `angular=recovery_turn_speed` durante `step_deg / recovery_deg_per_s`
segundos y ASUME que el robot giro `step_deg`. Con los defaults (45 grados,
45 grados/s) eso es 1 s por paso. Medido sobre el log: 13 intentos de giro,
mediana ~2 grados de giro REAL. El rover no rota a 0.45 en pasto. Como el
lazo abierto no se entera, se declara el giro completo y se vuelve a mirar la
misma pared.

`TurnController` cierra el lazo contra el rumbo real y sube la magnitud
angular cuando detecta que no pasa nada (banda muerta / rozamiento estatico).

Sin dependencias fuera de numpy/stdlib. `python3 escape.py` corre los tests.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


def wrap_deg(a: float) -> float:
    """Normaliza a (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- eleccion


@dataclass
class EscapeConfig:
    # Cuanto vale "no se" en fraccion libre. Un sector sin datos se trata como
    # si estuviera `prior_free` libre. Tiene que ser MAYOR que la fraccion
    # libre tipica de un sector ya demostrado bloqueado (0.05-0.10 en el log)
    # y menor que la de uno claramente abierto (>0.5).
    prior_free: float = 0.15
    # Peso del prior, en las mismas unidades que la cobertura. 0.30 significa
    # que un sector con 30% de cobertura queda mitad evidencia, mitad prior.
    prior_weight: float = 0.30
    # El rumbo actualmente bloqueado queda descalificado salvo que muestre al
    # menos esta fraccion libre (o sea: salvo que algo haya cambiado).
    unblock_free: float = 0.35
    # Penalizacion por repetir un rumbo ya intentado sin haber avanzado.
    repeat_penalty: float = 0.12
    repeat_memory: int = 4
    # Empujoncito hacia la meta. Solo desempata; no puede dominar.
    goal_bonus: float = 0.06
    # Si el mejor puntaje no llega a esto, no hay escape confiable: el que
    # llama deberia retroceder en vez de girar a ciegas.
    min_score: float = 0.10


@dataclass
class Sector:
    free: float      # 0..1
    coverage: float  # 0..1

    @classmethod
    def from_pct(cls, free_pct: float, coverage_pct: float) -> "Sector":
        return cls(free=free_pct / 100.0, coverage=coverage_pct / 100.0)


def shrunk_free(sector: Sector, cfg: EscapeConfig) -> float:
    """Fraccion libre corregida por cuanta evidencia hay detras.

    Media ponderada: la observacion pesa `coverage`, el prior pesa
    `prior_weight`. Cobertura 0.74 con 10% libre -> 0.11. Cobertura 0.15 con
    24% libre -> 0.18. Lo desconocido le gana a lo que ya sabemos que esta
    bloqueado, y lo bien observado y abierto le gana a lo desconocido.
    """
    w = max(sector.coverage, 0.0)
    return (sector.free * w + cfg.prior_free * cfg.prior_weight) / (w + cfg.prior_weight)


class EscapeChooser:
    """Elige a que rumbo relativo girar. Una instancia por bridge."""

    def __init__(self, cfg: Optional[EscapeConfig] = None) -> None:
        self.cfg = cfg or EscapeConfig()
        self._recent: deque = deque(maxlen=self.cfg.repeat_memory)

    def note_progress(self) -> None:
        """Llamar cuando el robot volvio a avanzar (>0.15 m): la memoria de
        intentos fallidos deja de aplicar."""
        self._recent.clear()

    def choose(
        self,
        sectors: Dict[float, Sector],
        *,
        blocked_heading: float = 0.0,
        goal_rel_deg: Optional[float] = None,
    ) -> Tuple[Optional[float], Dict[float, float]]:
        """Devuelve (rumbo_relativo, puntajes). None = no hay escape confiable."""
        cfg = self.cfg
        scores: Dict[float, float] = {}

        for heading, sec in sectors.items():
            if abs(wrap_deg(heading - blocked_heading)) < 1e-6 and sec.free < cfg.unblock_free:
                scores[heading] = -1.0
                continue

            score = shrunk_free(sec, cfg)
            score -= sum(1 for h in self._recent if h == heading) * cfg.repeat_penalty
            if goal_rel_deg is not None:
                cos = math.cos(math.radians(wrap_deg(goal_rel_deg - heading)))
                score += cfg.goal_bonus * max(cos, 0.0)
            scores[heading] = score

        if not scores:
            return None, scores
        best = max(scores, key=lambda h: scores[h])
        if scores[best] < cfg.min_score:
            return None, scores
        self._recent.append(best)
        return best, scores


def format_scores(sectors: Dict[float, Sector], scores: Dict[float, float]) -> Iterable[str]:
    """Lineas de log, para que se vea POR QUE se eligio lo que se eligio."""
    for h in sorted(sectors, key=lambda k: -scores.get(k, -9)):
        s = sectors[h]
        nota = " (descalificado: es el rumbo bloqueado)" if scores.get(h, 0) < 0 else ""
        yield (f"rumbo {h:+.0f} grados: libre={s.free * 100:.0f}% "
               f"cobertura={s.coverage * 100:.0f}% -> puntaje {scores.get(h, 0):.2f}{nota}")


# ------------------------------------------------------------------- giro


@dataclass
class TurnConfig:
    tolerance_deg: float = 15.0
    angular_start: float = 0.50
    angular_max: float = 1.00
    angular_step: float = 0.15
    # Si en `stall_ticks` ticks el rumbo no cambio mas que `stall_deg`, hay
    # banda muerta: subimos la magnitud. Esto es lo que el lazo abierto no
    # podia hacer porque nunca miraba el rumbo.
    stall_ticks: int = 2
    stall_deg: float = 3.0
    # Corte duro. A ~0.35 s por tick esto son ~14 s de giro maximo.
    max_ticks: int = 40
    # Si tras `wrong_way_ticks` ticks el error EMPEORO mas que
    # `wrong_way_deg`, el robot esta girando para el lado contrario: casi
    # siempre navigation.angular_sign al reves. Abortar rapido y decirlo es
    # mucho mejor que insistir 14 s a maximo angular en la direccion mala.
    wrong_way_ticks: int = 6
    wrong_way_deg: float = 10.0
    # Si hay mas que esto libre adelante, giramos en ARCO (con algo de
    # linear). Un poco de traccion longitudinal rompe el rozamiento estatico
    # mucho mejor que el giro puro, que es justo lo que fallaba en pasto.
    arc_min_clearance_m: float = 0.55
    arc_linear: float = 0.12
    tick_s: float = 0.35


@dataclass
class TurnCommand:
    linear: float
    angular: float
    done: bool
    reason: str


class TurnController:
    """Giro relativo con realimentacion de rumbo.

        turn = TurnController(-90.0, rumbo_actual, angular_sign=+1.0)
        while True:
            cmd = turn.step(rumbo_actual, clearance_m=clearance)
            send(cmd.linear, cmd.angular)
            if cmd.done:
                break

    `heading` puede ser la brujula o `math.degrees(pose.theta)`: lo unico que
    importa es que sea la MISMA fuente en todas las llamadas.
    """

    def __init__(self, delta_deg: float, heading0: float,
                 angular_sign: float = 1.0, max_angular: float = 1.0,
                 cfg: Optional[TurnConfig] = None) -> None:
        self.cfg = cfg or TurnConfig()
        self.target = (heading0 + delta_deg) % 360.0
        self.angular_sign = float(angular_sign)
        self.max_angular = float(max_angular)
        self.angular = min(self.cfg.angular_start, self.max_angular)
        self._last = heading0
        self._stalled = 0
        self._ticks = 0
        self.progress_deg = 0.0
        self._err0 = abs(wrap_deg(self.target - heading0))

    def step(self, heading: float, clearance_m: float = 0.0) -> TurnCommand:
        cfg = self.cfg
        self._ticks += 1
        err = wrap_deg(self.target - heading)

        if abs(err) <= cfg.tolerance_deg:
            return TurnCommand(0.0, 0.0, True, f"giro completo (err {err:+.0f} gr)")
        if self._ticks > cfg.max_ticks:
            return TurnCommand(0.0, 0.0, True,
                               f"giro abortado tras {self._ticks} ticks (err {err:+.0f} gr)")
        if (self._ticks >= cfg.wrong_way_ticks
                and abs(err) - self._err0 > cfg.wrong_way_deg):
            return TurnCommand(0.0, 0.0, True,
                               "ABORTO: el rover gira para el lado contrario "
                               f"(err {self._err0:.0f} -> {abs(err):.0f} gr). "
                               "Revisa navigation.angular_sign")

        movido = abs(wrap_deg(heading - self._last))
        self.progress_deg += movido
        if movido < cfg.stall_deg:
            self._stalled += 1
            if self._stalled >= cfg.stall_ticks:
                self.angular = min(self.angular + cfg.angular_step,
                                   cfg.angular_max, self.max_angular)
                self._stalled = 0
        else:
            self._stalled = 0
        self._last = heading

        linear = cfg.arc_linear if clearance_m >= cfg.arc_min_clearance_m else 0.0
        angular = self.angular_sign * math.copysign(self.angular, err)
        return TurnCommand(linear, angular, False,
                           f"girando {err:+.0f} gr (angular {angular:+.2f})")


# -------------------------------------------------------------- self test


def _t_elige_lo_desconocido_antes_que_la_pared() -> None:
    sectores = {0.0: Sector.from_pct(10, 74), 90.0: Sector.from_pct(24, 15),
                -90.0: Sector.from_pct(13, 24), 180.0: Sector.from_pct(0, 0)}
    pick, sc = EscapeChooser().choose(sectores, blocked_heading=0.0, goal_rel_deg=-41.0)
    assert pick != 0.0, f"volvio a elegir la pared: {sc}"
    assert sc[0.0] < 0.0, "el rumbo bloqueado tiene que quedar descalificado"
    assert pick in (90.0, -90.0), pick
    print(f"  0064 del log            -> elige {pick:+.0f} (antes elegia +0)")


def _t_no_se_queda_pegado() -> None:
    sectores = {0.0: Sector.from_pct(8, 75), 90.0: Sector.from_pct(25, 15),
                -90.0: Sector.from_pct(24, 23), 180.0: Sector.from_pct(0, 0)}
    ch = EscapeChooser()
    picks = [ch.choose(sectores, blocked_heading=0.0)[0] for _ in range(4)]
    assert len(set(picks)) > 1, f"bucle: {picks}"
    # Que el ultimo sea None es la escalada correcta: ya se probaron los dos
    # costados sin avanzar, asi que toca retroceder en vez de seguir girando.
    etiquetas = ["retroceder" if p is None else f"{p:+.0f}" for p in picks]
    print(f"  4 intentos seguidos     -> {etiquetas}")


def _t_pide_retroceso_si_no_hay_nada() -> None:
    sectores = {0.0: Sector.from_pct(3, 51), 90.0: Sector.from_pct(0, 15),
                -90.0: Sector.from_pct(9, 16), 180.0: Sector.from_pct(0, 0)}
    pick, _ = EscapeChooser(EscapeConfig(min_score=0.20)).choose(sectores, blocked_heading=0.0)
    assert pick is None
    print("  todo bloqueado          -> None (retroceder, no girar a ciegas)")


def _t_un_sector_abierto_y_visto_gana() -> None:
    # La cobertura sigue valiendo: si un sector esta bien observado Y libre,
    # tiene que ganarle a uno desconocido.
    sectores = {0.0: Sector.from_pct(5, 80), 90.0: Sector.from_pct(90, 70),
                -90.0: Sector.from_pct(0, 0)}
    pick, _ = EscapeChooser().choose(sectores, blocked_heading=0.0)
    assert pick == 90.0, pick
    print("  costado abierto y visto -> gana al desconocido")


def _t_giro_escala_ante_banda_muerta() -> None:
    turn = TurnController(-90.0, 77.0)
    heading, mags = 77.0, []
    for _ in range(12):
        cmd = turn.step(heading, clearance_m=0.27)
        if cmd.done:
            break
        mags.append(abs(cmd.angular))
        heading -= 1.0  # el rover apenas se mueve, como en el log real
    assert mags[-1] > mags[0], mags
    assert mags[-1] <= TurnConfig().angular_max + 1e-9
    print(f"  rover que no gira       -> angular {mags[0]:.2f} -> {mags[-1]:.2f}")


def _t_giro_no_escala_si_gira_bien() -> None:
    turn = TurnController(-90.0, 90.0)
    heading, mags = 90.0, []
    for _ in range(20):
        cmd = turn.step(heading)
        if cmd.done:
            break
        mags.append(abs(cmd.angular))
        heading -= 15.0
    assert max(mags) == mags[0], f"escalo sin necesidad: {mags}"
    print(f"  rover que gira bien     -> angular constante {mags[0]:.2f}, "
          f"{len(mags)} ticks")


def _t_giro_termina_y_frena() -> None:
    turn = TurnController(-90.0, 90.0)
    assert not turn.step(90.0).done
    cmd = turn.step(2.0)
    assert cmd.done and cmd.linear == 0.0 and cmd.angular == 0.0, cmd


def _t_giro_en_arco_si_hay_lugar() -> None:
    assert TurnController(90.0, 0.0).step(0.0, clearance_m=0.80).linear > 0.0
    assert TurnController(90.0, 0.0).step(0.0, clearance_m=0.30).linear == 0.0
    print("  arco solo si hay lugar  -> ok")


def _t_respeta_angular_sign_y_max() -> None:
    cmd = TurnController(90.0, 0.0, angular_sign=-1.0, max_angular=0.6).step(0.0)
    assert cmd.angular == -0.5, cmd
    t = TurnController(90.0, 0.0, max_angular=0.6)
    for _ in range(10):
        c = t.step(0.0)
    assert abs(c.angular) <= 0.6 + 1e-9, c


def _t_giro_no_es_infinito() -> None:
    turn = TurnController(180.0, 0.0)
    n = 0
    while True:
        cmd = turn.step(0.0)  # el robot nunca se mueve
        n += 1
        if cmd.done:
            break
        assert n < 100
    assert "abortado" in cmd.reason
    print(f"  robot trabado           -> aborta a los {n} ticks, no cuelga")


def _t_detecta_sentido_invertido() -> None:
    # El rover gira para el lado equivocado (angular_sign mal puesto).
    turn = TurnController(-90.0, 0.0)
    heading, n = 0.0, 0
    while True:
        cmd = turn.step(heading)
        n += 1
        if cmd.done:
            break
        heading += 5.0  # se aleja del objetivo
        assert n < 20, "insistio demasiado en la direccion mala"
    assert "lado contrario" in cmd.reason, cmd.reason
    print(f"  sentido invertido       -> aborta a los {n} ticks y lo dice")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("_t_")]
    for t in tests:
        t()
    print(f"\nescape: {len(tests)} tests pasaron.")
