"""Gobernador dinámico de velocidad por latencia (P95) para La Rovernetta.

Modula la velocidad lineal admisible resolviendo la física cuadrática de frenado:
    (1 / (2 * a_brake)) * v^2 + (t_plan_p95 + t_cmd_latency) * v - (d_horizon / margin) = 0

Diferencia explícitamente:
- Latencia de procesamiento (percepción + planificación): t_plan_p95 medida en ventana móvil.
- Latencia de comando a movimiento físico: t_cmd_latency (ASUMIDO: 2.0s, punto medio del rango 1.5 - 2.5s medido una vez en giro 360° motorizado).

Incluye:
- Percentil 95 sobre ventana móvil (30 muestras) para inmunidad contra picos aislados de 1 frame.
- Corte total (Stop & Wait) si v_safe cae por debajo de la velocidad mínima operativa del rover.
- Suavizado asimétrico anti-oscilación: reducción rápida ante aumento de latencia y recuperación gradual.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class GovernorConfig:
    enabled: bool = True
    window_size: int = 30           # Cantidad de muestras en la ventana móvil P95
    a_brake: float = 1.5            # [m/s^2] Desaceleración de frenado (ASUMIDO, consistente con el proyecto)
    cmd_latency_s: float = 2.0      # [s] Retardo comando->movimiento físico (ASUMIDO: punto medio del rango 1.5-2.5s medido en giro 360° motorizado, no remedido por corrida)
    d_horizon_m: float = 1.25       # [m] Horizonte visible frontal (front_far_m de percepción)
    margin: float = 1.2             # Factor de seguridad multiplicativo sobre distancia de parada (ASUMIDO)
    min_speed_mps: float = 0.10     # [m/s] Umbral mínimo operativo; por debajo corta a 0 (Stop & Wait) (ASUMIDO)
    max_linear_speed_mps: float = 0.557  # [m/s] Velocidad a throttle 1.0 (112 RPM, radio 0.0475m)
    alpha_up: float = 0.25          # Tasa de recuperación gradual de velocidad (anti-oscilación)


class VelocityGovernor:
    """Gobernador dinámico de velocidad basado en latencia del pipeline y física de frenado."""

    def __init__(self, cfg: GovernorConfig | None = None):
        self.cfg = cfg or GovernorConfig()
        self._latencies: collections.deque[float] = collections.deque(maxlen=int(self.cfg.window_size))
        self._filtered_v_safe: float | None = None
        self._filtered_throttle_limit: float | None = None
        self._last_t_plan_p95: float = 0.0

    @property
    def latencies(self) -> list[float]:
        return list(self._latencies)

    @property
    def filtered_v_safe(self) -> float | None:
        return self._filtered_v_safe

    @property
    def filtered_throttle_limit(self) -> float | None:
        return self._filtered_throttle_limit

    def record_step_duration(self, duration_s: float) -> None:
        """Registra la latencia de procesamiento del frame (percepción + planificación)."""
        if duration_s > 0.0:
            self._latencies.append(float(duration_s))

    def compute_t_plan_p95(self) -> float:
        """Calcula el percentil 95 de la latencia de procesamiento en la ventana móvil."""
        if not self._latencies:
            return 0.0
        p95 = float(np.percentile(list(self._latencies), 95))
        self._last_t_plan_p95 = p95
        return p95

    def compute_v_safe_raw(self, t_plan_p95: float) -> float:
        """Deriva v_safe (m/s) resolviendo la física de frenado cuadrática:
            (1 / (2 * a)) * v^2 + t_total * v - (d / m) = 0
            v_safe = a * (sqrt(t_total^2 + 2 * d / (a * m)) - t_total)
        """
        t_total = float(t_plan_p95) + float(self.cfg.cmd_latency_s)
        a = float(self.cfg.a_brake)
        d = float(self.cfg.d_horizon_m)
        m = float(self.cfg.margin)

        disc_term = (t_total ** 2) + (2.0 * d) / (a * m)
        v_safe = a * (math.sqrt(disc_term) - t_total)
        return max(0.0, float(v_safe))

    def update(self, step_duration_s: float) -> tuple[float, float, float]:
        """Actualiza el gobernador con la duración del ciclo actual.

        Retorna:
            (t_plan_p95_s, v_safe_mps, throttle_limit)
        """
        if not self.cfg.enabled:
            return 0.0, float("inf"), 1.0

        self.record_step_duration(step_duration_s)
        t_plan_p95 = self.compute_t_plan_p95()
        v_safe_raw = self.compute_v_safe_raw(t_plan_p95)

        # Regla de corte mínimo: si v_safe cae por debajo del umbral mínimo operativo
        # donde los motores DC vencen la fricción estática, se frena a cero (Stop & Wait)
        if v_safe_raw < float(self.cfg.min_speed_mps):
            target_throttle = 0.0
            target_v_safe = 0.0
        else:
            v_max = max(1e-6, float(self.cfg.max_linear_speed_mps))
            target_throttle = float(np.clip(v_safe_raw / v_max, 0.0, 1.0))
            target_v_safe = v_safe_raw

        # Suavizado asimétrico anti-oscilación:
        # - Ataque rápido (frena inmediatamente si target < actual por seguridad)
        # - Relajación lenta (recupera velocidad gradualmente si target > actual)
        if self._filtered_throttle_limit is None:
            self._filtered_throttle_limit = target_throttle
            self._filtered_v_safe = target_v_safe
        else:
            if target_throttle < self._filtered_throttle_limit:
                # Freno inmediato
                self._filtered_throttle_limit = target_throttle
                self._filtered_v_safe = target_v_safe
            else:
                # Recuperación gradual
                alpha = float(np.clip(self.cfg.alpha_up, 0.01, 1.0))
                self._filtered_throttle_limit += alpha * (target_throttle - self._filtered_throttle_limit)
                self._filtered_v_safe += alpha * (target_v_safe - self._filtered_v_safe)

        return t_plan_p95, self._filtered_v_safe, self._filtered_throttle_limit

    def apply_throttle_limit(self, linear_cmd: float) -> float:
        """Aplica el límite de acelerador gobernado sobre el comando lineal propuesto.
        
        Solo modula comandos de avance positivo (linear_cmd > 0).
        """
        if not self.cfg.enabled or self._filtered_throttle_limit is None:
            return linear_cmd

        if linear_cmd <= 0.0:
            return linear_cmd

        # Si el límite es 0.0, corte total (Stop & Wait)
        if self._filtered_throttle_limit <= 0.0:
            return 0.0

        return float(min(linear_cmd, self._filtered_throttle_limit))
