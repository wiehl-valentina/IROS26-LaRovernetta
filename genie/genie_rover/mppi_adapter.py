"""Adapters for the existing rover: maps, metric controls and recovery state."""
from dataclasses import dataclass
import math
from numbers import Integral
import numpy as np
from .mppi import MPPI, MPPIConfig
from .navigation import DriveCommand


class LocalMap:
    """Sample forward/left coordinates. Fresh observed cells override memory.

    Unknown is never assumed free outside the currently occupied footprint.
    PersistentMap is read only; it may cover the sides and rear as well.
    """
    def __init__(self, fresh, resolution, persistent=None, pose=None):
        self.fresh = fresh
        self.resolution = resolution
        self.persistent = persistent
        self.pose = pose

    def __call__(self, forward, left):
        values = np.zeros_like(forward, dtype=float)
        known = np.zeros_like(forward, dtype=bool)
        if self.persistent is not None and self.pose is not None:
            p, m = self.pose, self.persistent
            c, s = math.cos(p.theta), math.sin(p.theta)
            x = p.x + c*forward - s*left
            y = p.y + s*forward + c*left
            r = np.rint(m.n/2-(x-m.origin_x)/m.cfg.resolution_m_per_px).astype(int)
            col = np.rint(m.n/2-(y-m.origin_y)/m.cfg.resolution_m_per_px).astype(int)
            inside = (r >= 0) & (r < m.n) & (col >= 0) & (col < m.n)
            rr, cc = np.clip(r, 0, m.n-1), np.clip(col, 0, m.n-1)
            known = inside & (m.conf[rr, cc] >= m.cfg.min_confidence)
            values = np.where(known, m.value[rr, cc], 0.0)
        h, w = self.fresh.traversability.shape
        r = h-1-np.floor(forward/self.resolution).astype(int)
        col = w//2-np.floor(left/self.resolution).astype(int)
        inside = (forward >= 0) & (r >= 0) & (r < h) & (col >= 0) & (col < w)
        rr, cc = np.clip(r, 0, h-1), np.clip(col, 0, w-1)
        v = self.fresh.traversability[rr, cc]
        observed = inside & self.fresh.observed[rr, cc].astype(bool) & np.isfinite(v) & (v >= 0)
        values = np.where(observed, v, values)
        known = (known | observed) & np.isfinite(values)
        return values, known


@dataclass
class ActuationConfig:
    # Measured physical speed at SDK magnitude 1.0; not navigation.max_linear!
    linear_mps_per_unit: float = 1.0
    angular_rps_per_unit: float = 1.0
    calibrated: bool = False
    max_observation_age_s: float = 2.0

    def __post_init__(self):
        for name in ('linear_mps_per_unit', 'angular_rps_per_unit', 'max_observation_age_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'mppi_actuation.{name} must be finite and positive')

    def command(self, control, angular_sign, reason):
        # Optimizer omega positive = left; follower angular_sign maps RIGHT.
        return DriveCommand(float(control[0]/self.linear_mps_per_unit),
                            float(-angular_sign*control[1]/self.angular_rps_per_unit), reason)


@dataclass
class RecoveryConfig:
    max_steps: int = 20
    attempt_steps: int = 6
    progress_m: float = 0.15
    progress_rad: float = 0.5
    target_distance_m: float = 0.5

    def __post_init__(self):
        for name in ('max_steps', 'attempt_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'mppi_recovery.{name} must be a positive integer')
        if not all(math.isfinite(v) and v > 0 for v in (self.progress_m, self.progress_rad, self.target_distance_m)):
            raise ValueError('mppi_recovery distances must be positive')


class MPPIRecovery:
    """Bounded episodes; re-observe after each pulse, retain tried headings.

    Translation progress and a clear front end an episode. Without odometry,
    only a newly clear front ends it. No blind forward or reverse fallback.
    """
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.cfg = config
        self.active = False
        self.steps = 0
        self.origin = None
        self.attempts = np.zeros(5)
        self.goal_world = None
        self.goal_local = None
        self.attempt_remaining = 0

    def start(self, pose):
        if self.active:
            return
        self.active = True
        self.steps = 0
        self.origin = None if pose is None else np.array([pose.x, pose.y, pose.theta])
        self.attempts.fill(0)
        self.goal_world = None
        self.goal_local = None
        self.attempt_remaining = 0
        self.optimizer.reset()

    def finished(self, pose, blocked):
        if blocked or not self.steps:
            return False
        return (self.origin is None or
                (pose is not None and (
                    np.linalg.norm(np.array([pose.x, pose.y])-self.origin[:2]) >= self.cfg.progress_m or
                    abs(math.atan2(math.sin(pose.theta-self.origin[2]),
                                   math.cos(pose.theta-self.origin[2]))) >= self.cfg.progress_rad)))

    def plan(self, local_map, blocked, pose=None):
        if self.attempt_remaining <= 0:
            angles = np.array([0, np.pi/2, -np.pi/2, 3*np.pi/4, -3*np.pi/4])
            goals = self.cfg.target_distance_m*np.column_stack((np.cos(angles), np.sin(angles)))
            v, known = local_map(goals[:, 0], goals[:, 1])
            # Unknown may be inspected by rotating; never authorizes translation.
            scores = np.where(known, v, 0.25)-0.2*self.attempts
            if blocked:
                scores[0] = -np.inf
            idx = int(np.argmax(scores))
            self.attempts[idx] += 1
            self.goal_local = goals[idx]
            self.goal_world = None
            if pose is not None:
                c, s = math.cos(pose.theta), math.sin(pose.theta)
                self.goal_world = np.array([pose.x, pose.y]) + np.array([[c, -s], [s, c]]) @ self.goal_local
            self.attempt_remaining = self.cfg.attempt_steps
            self.optimizer.reset()
        goal = self.goal_local
        if self.goal_world is not None and pose is not None:
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            goal = np.array([[c, s], [-s, c]]) @ (self.goal_world-np.array([pose.x, pose.y]))
        self.steps += 1
        self.attempt_remaining -= 1
        result = self.optimizer.plan(local_map, goal, recovery=True, front_blocked=blocked)
        if not result.valid:
            self.attempt_remaining = 0
        return result


def build_backends(cfg, dry_run):
    """Independent selectors; don't parse/import optional configs when unused."""
    planning = cfg.get('navigation', {}).get('trajectory_algorithm', 'polynomial')
    recovery = cfg.get('safety', {}).get('recovery_algorithm', 'legacy')
    if planning not in ('polynomial', 'mppi', 'nomad') or recovery not in ('legacy', 'mppi'):
        raise ValueError('trajectory_algorithm: polynomial|mppi|nomad; recovery_algorithm: legacy|mppi')
    if planning != 'mppi' and recovery != 'mppi':
        return None, None, None
    act = ActuationConfig(**cfg.get('mppi_actuation', {}))
    if not dry_run and act.calibrated is not True:
        raise ValueError('MPPI real requiere mppi_actuation.calibrated: true y escalas medidas')
    config = MPPIConfig(**cfg.get('mppi', {}))
    nav = cfg['navigation']
    if nav['angular_sign'] not in (-1, 1):
        raise ValueError('MPPI requiere angular_sign = -1 o +1')
    if (max(config.max_v, config.max_reverse_v)/act.linear_mps_per_unit > min(1, nav['max_linear']) or
            config.max_w/act.angular_rps_per_unit > min(1, nav['max_angular'])):
        raise ValueError('Limites metricos MPPI exceden limites SDK/navigation; ajustar escalas o mppi')
    planner = MPPI(config) if planning == 'mppi' else None
    recoverer = MPPIRecovery(MPPI(config), RecoveryConfig(**cfg.get('mppi_recovery', {}))) if recovery == 'mppi' else None
    return planner, recoverer, act
