"""NumPy sampling MPC/MPPI optimizer. No SDK, torch or robot side effects.

States: [forward metres, left metres, counterclockwise radians]. Controls:
[m/s, rad/s]. Weighted path-integral updates with bounded controls, followed
by a hard collision check of the updated sequence (and a best-sample fallback).
"""
from dataclasses import dataclass
import math
from numbers import Integral
import numpy as np


@dataclass
class MPPIConfig:
    samples: int = 256
    horizon: int = 16
    iterations: int = 2
    dt: float = 0.15
    temperature: float = 0.5
    max_v: float = 0.22
    max_reverse_v: float = 0.10
    max_w: float = 0.4
    noise_v: float = 0.12
    noise_w: float = 0.45
    radius_m: float = 0.20
    collision_step_m: float = 0.025
    min_traversability: float = 0.4
    terrain_weight: float = 2.0
    goal_weight: float = 4.0
    smooth_weight: float = 0.15
    heading_weight: float = 0.5
    seed: int = 42

    def __post_init__(self):
        for name in ('samples', 'horizon', 'iterations'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'mppi.{name} must be a positive integer')
        if self.samples < 16:
            raise ValueError('mppi.samples must be >= 16')
        for name in ('dt', 'temperature', 'max_v', 'max_reverse_v', 'max_w',
                     'noise_v', 'noise_w', 'radius_m', 'collision_step_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'mppi.{name} must be finite and positive')
        for name in ('terrain_weight', 'goal_weight', 'smooth_weight', 'heading_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'mppi.{name} must be finite and nonnegative')
        if not 0 <= self.min_traversability <= 1:
            raise ValueError('mppi.min_traversability must be in [0,1]')


@dataclass
class MPPIResult:
    controls: np.ndarray
    states: np.ndarray
    cost: float
    valid: bool

    @property
    def final_path_xy_m(self):
        # Existing planner convention: right, forward.
        return np.column_stack((-self.states[:, 1], self.states[:, 0]))


class MPPI:
    def __init__(self, config: MPPIConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.nominal = np.zeros((config.horizon, 2))
        # Sample the whole circular footprint, not only its boundary.
        n = int(np.ceil(config.radius_m / config.collision_step_m))
        a = np.linspace(-config.radius_m, config.radius_m, 2*n+1)
        x, y = np.meshgrid(a, a)
        mask = x*x + y*y <= config.radius_m**2 + 1e-12
        self.footprint = np.column_stack((x[mask], y[mask]))

    def reset(self):
        self.nominal.fill(0)

    def rollout(self, controls):
        """Integrate at substeps to avoid jumping over thin obstacles."""
        c = self.cfg
        steps = max(1, int(np.ceil(max(c.max_v, c.max_reverse_v)*c.dt /
                                    c.collision_step_m)))
        state = np.zeros((len(controls), 3))
        result = [state.copy()]
        for t in range(c.horizon):
            v, w = controls[:, t, 0], controls[:, t, 1]
            for _ in range(steps):
                dt = c.dt / steps
                mid = state[:, 2] + w*dt/2
                state[:, 0] += v*np.cos(mid)*dt
                state[:, 1] += v*np.sin(mid)*dt
                state[:, 2] += w*dt
                result.append(state.copy())
        return np.stack(result, axis=1)

    def evaluate(self, controls, sample_map, goal):
        c = self.cfg
        states = self.rollout(controls)
        valid = np.ones(len(controls), dtype=bool)
        terrain = np.zeros(len(controls))
        # Loop over time, vectorize samples and footprint. Bounded memory.
        for state in states.transpose(1, 0, 2):
            points = state[:, None, :2] + self.footprint[None]
            values, known = sample_map(points[..., 0], points[..., 1])
            # Camera cannot observe under the robot. Unknown cells of the
            # CURRENT occupied disk may be exempted; known obstacles never are.
            occupied = np.sum(points**2, axis=-1) <= c.radius_m**2 + 1e-12
            safe = (known & (values >= c.min_traversability)) | (~known & occupied)
            valid &= safe.all(axis=1)
            terrain += np.mean(np.where(known, 1-values, 0.5), axis=1)
        end = states[:, -1]
        goal = np.asarray(goal, dtype=float)
        angle = np.arctan2(goal[1], goal[0])
        smooth = (np.mean(np.diff(controls, axis=1)**2, axis=(1, 2))
                  if c.horizon > 1 else np.zeros(len(controls)))
        cost = (c.terrain_weight*terrain/states.shape[1] +
                c.goal_weight*np.linalg.norm(end[:, :2]-goal, axis=1) +
                c.heading_weight*(1-np.cos(end[:, 2]-angle)) +
                c.smooth_weight*smooth)
        cost[~valid] = np.inf
        return states, cost, valid

    def plan(self, sample_map, goal, recovery=False, front_blocked=False):
        if np.shape(goal) != (2,) or not np.isfinite(goal).all():
            raise ValueError('MPPI goal must be finite [forward_m, left_m]')
        c = self.cfg
        low = -c.max_reverse_v if recovery else 0.0
        high = 0.0 if front_blocked else c.max_v
        best = None
        for _ in range(c.iterations):
            controls = self.nominal[None] + self.rng.normal(
                size=(c.samples, c.horizon, 2))*[c.noise_v, c.noise_w]
            # Structured seeds help find separated left/right/reverse modes.
            controls[0] = self.nominal
            k = 1
            for v in (high, low, 0.0):
                for w in (-c.max_w, -c.max_w/2, 0.0, c.max_w/2, c.max_w):
                    controls[k, :, :] = (v, w)
                    k += 1
            controls[..., 0] = np.clip(controls[..., 0], low, high)
            controls[..., 1] = np.clip(controls[..., 1], -c.max_w, c.max_w)
            states, costs, valid = self.evaluate(controls, sample_map, goal)
            if not valid.any():
                self.reset()
                return MPPIResult(np.zeros((c.horizon, 2)), np.zeros((1, 3)), np.inf, False)
            idx = int(np.argmin(costs))
            weights = np.zeros(c.samples)
            weights[valid] = np.exp(-(costs[valid]-costs[idx])/c.temperature)
            weights /= weights.sum()
            proposed = np.sum(weights[:, None, None]*controls, axis=0)
            ps, pc, pv = self.evaluate(proposed[None], sample_map, goal)
            # Averaging two safe branches can cross an obstacle. Revalidate.
            if pv[0] and pc[0] <= costs[idx]:
                best = MPPIResult(proposed.copy(), ps[0], float(pc[0]), True)
            else:
                best = MPPIResult(controls[idx].copy(), states[idx], float(costs[idx]), True)
            self.nominal = best.controls.copy()
        self.nominal[:-1] = best.controls[1:]
        self.nominal[-1] = 0
        return best
