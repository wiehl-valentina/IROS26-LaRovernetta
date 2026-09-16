"""Offline tests: no robot, GPU, checkpoints or network."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np

from genie_rover.mppi import MPPI, MPPIConfig
from genie_rover.mppi_adapter import LocalMap, ActuationConfig, build_backends, MPPIRecovery, RecoveryConfig
from genie_rover.odometry import Pose
from genie_rover.persistent_map import PersistentMap, MapConfig
from genie_rover.bridge import Bridge


def free(x, y):
    return np.ones_like(x), np.ones_like(x, dtype=bool)


class OptimizerTests(unittest.TestCase):
    def optimizer(self, **kwargs):
        return MPPI(MPPIConfig(samples=32, horizon=8, iterations=1, radius_m=0.1, **kwargs))

    def test_forward_and_bounds(self):
        m = self.optimizer()
        r = m.plan(free, [2, 0])
        self.assertTrue(r.valid)
        self.assertGreater(r.states[-1, 0], 0.1)
        self.assertTrue((r.controls[:, 0] >= 0).all())
        self.assertLessEqual(abs(r.controls[:, 1]).max(), m.cfg.max_w)
        np.testing.assert_allclose(r.final_path_xy_m[:, 0], -r.states[:, 1])
        np.testing.assert_allclose(m.nominal[:-1], r.controls[1:])

    def test_wall_no_crossing(self):
        def wall(x, y):
            return np.where(x > 0.25, 0., 1.), np.ones_like(x, bool)
        m = self.optimizer()
        r = m.plan(wall, [2, 0])
        self.assertTrue(r.valid)
        self.assertLessEqual(r.states[:, 0].max()+m.cfg.radius_m, 0.25+1e-9)
        self.assertTrue(m.evaluate(r.controls[None], wall, [2, 0])[2][0])

    def test_obstacle_under_footprint_rejects_all(self):
        m = self.optimizer()
        r = m.plan(lambda x, y: (np.zeros_like(x), np.ones_like(x, bool)), [1, 0])
        self.assertFalse(r.valid)
        self.assertFalse(m.nominal.any())

    def test_reverse_only_recovery_and_known_rear(self):
        m = self.optimizer()
        r = m.plan(free, [-1, 0], recovery=True, front_blocked=True)
        self.assertTrue(r.valid)
        self.assertLess(r.states[-1, 0], -0.05)
        self.assertTrue((r.controls[:, 0] <= 0).all())
        def unknown_rear(x, y):
            return np.ones_like(x), x >= 0
        m.reset()
        r = m.plan(unknown_rear, [-1, 0], recovery=True, front_blocked=True)
        self.assertTrue(r.valid)
        self.assertAlmostEqual(r.states[:, 0].min(), 0, places=8)

    def test_swept_collision_not_just_endpoints(self):
        m = self.optimizer(dt=1., max_v=1.)
        def thin_wall(x, y):
            return np.where((x > .25) & (x < .30), 0., 1.), np.ones_like(x, bool)
        controls = np.zeros((1, m.cfg.horizon, 2))
        controls[..., 0] = 1
        self.assertFalse(m.evaluate(controls, thin_wall, [5, 0])[2][0])

    def test_validation(self):
        for kwargs in ({'dt': 0}, {'temperature': float('nan')}, {'samples': 2}, {'horizon': 2.5}):
            with self.assertRaises(ValueError):
                MPPIConfig(**kwargs)


class AdapterTests(unittest.TestCase):
    def test_fresh_overrides_memory_and_rear_transform(self):
        memory = PersistentMap(MapConfig(size_m=4, resolution_m_per_px=.1))
        memory.value.fill(1)
        memory.conf.fill(1)
        fresh = SimpleNamespace(traversability=np.zeros((10, 20)), observed=np.ones((10, 20)))
        sampler = LocalMap(fresh, .1, memory, Pose(theta=np.pi/2))
        v, k = sampler(np.array([.3, -.3, 10.]), np.zeros(3))
        np.testing.assert_equal(k, [True, True, False])
        np.testing.assert_allclose(v[:2], [0, 1])

    def test_command_units_sign(self):
        a = ActuationConfig(linear_mps_per_unit=2, angular_rps_per_unit=4)
        cmd = a.command([.2, .4], -1, 'test')
        self.assertAlmostEqual(cmd.linear, .1)
        self.assertAlmostEqual(cmd.angular, .1)
        self.assertAlmostEqual(a.command([.2, .4], 1, 'test').angular, -.1)

    def test_memory_coordinates_rotate_with_rover(self):
        memory = PersistentMap(MapConfig(size_m=4, resolution_m_per_px=.1))
        fresh = SimpleNamespace(traversability=np.full((10, 20), -1.), observed=np.zeros((10, 20)))
        # World x=.3,y=-.2; rover at .3,.3 facing +world y: obstacle behind.
        row, col = memory.world_to_cell(.3, -.2)
        memory.conf[row, col] = 1
        memory.value[row, col] = 0
        sampler = LocalMap(fresh, .1, memory, Pose(.3, .3, np.pi/2))
        v, k = sampler(np.array([-.5, .5]), np.zeros(2))
        np.testing.assert_equal(k, [True, False])
        self.assertEqual(v[0], 0)

    def test_recovery_holds_world_goal(self):
        r = MPPIRecovery(MPPI(MPPIConfig(samples=16, horizon=4, iterations=1)), RecoveryConfig())
        r.start(Pose())
        r.plan(free, True, Pose())
        goal = r.goal_world.copy()
        r.plan(free, True, Pose(theta=.1))
        np.testing.assert_allclose(r.goal_world, goal)
        self.assertEqual(r.attempts.sum(), 1)
        self.assertTrue(r.finished(Pose(theta=.6), False))

    def test_independent_selectors_and_real_gate(self):
        cfg = {'navigation': {'max_linear': .7, 'max_angular': .45, 'angular_sign': -1}, 'safety': {}}
        cfg['mppi'] = {'invalid_unused_key': 1}
        self.assertEqual(build_backends(cfg, True), (None, None, None))
        cfg['mppi'] = {}
        for planner, recovery in [('mppi', 'legacy'), ('polynomial', 'mppi'), ('mppi', 'mppi')]:
            cfg['navigation']['trajectory_algorithm'] = planner
            cfg['safety']['recovery_algorithm'] = recovery
            p, r, a = build_backends(cfg, True)
            self.assertEqual(p is not None, planner == 'mppi')
            self.assertEqual(r is not None, recovery == 'mppi')
            with self.assertRaises(ValueError):
                build_backends(cfg, False)

    def test_recovery_progress_and_attempts(self):
        r = MPPIRecovery(MPPI(MPPIConfig(samples=16, horizon=4, iterations=1)), RecoveryConfig())
        r.start(Pose())
        r.plan(free, True)
        self.assertEqual(r.steps, 1)
        self.assertFalse(r.finished(Pose(), False))
        self.assertTrue(r.finished(Pose(x=.2), False))
        self.assertFalse(r.finished(Pose(x=.2), True))


class BridgeTests(unittest.TestCase):
    def make_bridge(self, planning='polynomial', recovery='legacy', blocked=False):
        cfg = {
            'rover': {'base_url': 'http://unused'},
            'navigation': {'max_linear': .7, 'max_angular': .45, 'angular_sign': -1,
                           'trajectory_algorithm': planning},
            'projection': {'resolution_m_per_px': .03, 'forward_range_m': 2., 'side_range_m': 2.},
            'memory': {'enabled': False},
            'safety': {'recovery_algorithm': recovery, 'obstacle_persist_frames': 1},
            'mppi': {'samples': 16, 'horizon': 4, 'iterations': 1},
        }
        fresh = SimpleNamespace(traversability=np.ones((67, 134)), observed=np.ones((67, 134)),
                                stats={'bev_observed_cells': 100})
        if blocked:
            fresh.traversability[40:55, 57:77] = 0
        with patch('genie_rover.bridge.RoverClient') as client, patch('genie_rover.bridge.PerceptionPipeline') as perception:
            b = Bridge(cfg, dry_run=True)
            client.return_value.front_frame.return_value = (np.zeros((10, 10, 3)), 1.)
            client.return_value.telemetry.return_value = SimpleNamespace(
                latitude=0, longitude=0, orientation=0, timestamp=1, raw={})
            perception.return_value.process.return_value = fresh
        b.send = MagicMock()
        return b

    def test_step_legacy_keeps_polynomial(self):
        b = self.make_bridge()
        with patch('genie_rover.bridge.plan_on_bev') as planner:
            planner.return_value.final_path_xy_m = np.array([[0., 0.], [0., 1.]])
            b._step()
            planner.assert_called_once()
        self.assertEqual(b.stats.plans_ok, 1)
        self.assertGreater(b.send.call_args.args[0].linear, 0)

    def test_step_mppi_never_calls_polynomial(self):
        b = self.make_bridge(planning='mppi')
        with patch('genie_rover.bridge.plan_on_bev') as planner:
            b._step()
            planner.assert_not_called()
        self.assertEqual(b.stats.plans_ok, 1)
        self.assertEqual(b.send.call_args.args[0].linear, 0)

    def test_both_mppi_backends_exit_recovery(self):
        b = self.make_bridge(planning='mppi', recovery='mppi', blocked=True)
        b._step()
        self.assertTrue(b.mppi_recovery.active)
        # No odometry: a clear fresh observation is the documented criterion.
        b.perception.process.return_value.traversability.fill(1)
        b._step()
        self.assertFalse(b.mppi_recovery.active)
        self.assertEqual(b.stats.plans_ok, 1)

    def test_polynomial_with_mppi_recovery_triggers_on_block(self):
        b = self.make_bridge(recovery='mppi', blocked=True)
        with patch('genie_rover.bridge.plan_on_bev') as planner:
            b._step()
            planner.assert_not_called()
        self.assertTrue(b.mppi_recovery.active)
        self.assertEqual(b.mppi_recovery.steps, 1)
        self.assertTrue(all(call.args[0].linear <= 0 for call in b.send.call_args_list))

    def test_recovery_budget_stops(self):
        b = self.make_bridge(recovery='mppi', blocked=True)
        b._step()
        b.mppi_recovery.steps = b.mppi_recovery.cfg.max_steps
        b._step()
        self.assertTrue(b._stop_requested)
        self.assertEqual(b.send.call_args.args[0].linear, 0)

    def test_optional_config_invalid_does_not_load_perception(self):
        cfg = {'navigation': {'trajectory_algorithm': 'typo'}}
        with patch('genie_rover.bridge.PerceptionPipeline') as perception:
            with self.assertRaises(ValueError):
                Bridge(cfg)
            perception.assert_not_called()

    def test_motion_exception_always_attempts_stop(self):
        import time
        b = self.make_bridge(planning='mppi')
        r = b.mppi.plan(free, [1, 0])
        b.send.side_effect = [RuntimeError('transport failure'), None]
        with self.assertRaises(RuntimeError):
            b._send_mppi(r, b.mppi, time.monotonic(), 'test')
        self.assertEqual(b.send.call_count, 2)
        self.assertEqual(b.send.call_args.args[0].linear, 0)

    def test_legacy_recovery_alternates_both_sign_conventions(self):
        for sign in (-1, 1):
            b = self.make_bridge()
            b.follower.angular_sign = sign
            with patch('genie_rover.bridge.time.sleep') as sleep:
                b._recover()
                b._recover()
                b._recover()
                sleep.assert_not_called()
            commands = [c.args[0] for c in b.send.call_args_list]
            speed = b.follower.turn_speed
            self.assertEqual([c.angular for c in commands],
                             [sign*speed, 0, -sign*speed, 0, sign*speed, 0])
            self.assertTrue(all(c.linear == 0 for c in commands))

    def test_legacy_exception_attempts_stop(self):
        b = self.make_bridge()
        b.send.side_effect = [RuntimeError('transport failed'), None]
        with self.assertRaises(RuntimeError):
            b._recover()
        self.assertEqual(b.send.call_args.args[0].angular, 0)

    def test_legacy_stop_requested_does_not_turn(self):
        b = self.make_bridge()
        b._stop_requested = True
        b._recover()
        self.assertEqual(b.send.call_count, 1)
        self.assertEqual(b.send.call_args.args[0].angular, 0)

    def test_legacy_interrupt_during_turn_stops(self):
        b = self.make_bridge()
        b.dry_run = False
        def interrupt(_seconds):
            b.request_stop()
        with patch('genie_rover.bridge.time.sleep', side_effect=interrupt):
            b._recover()
        self.assertEqual(b.send.call_args.args[0].angular, 0)
        self.assertTrue(b._stop_requested)

    def test_pulse_dry_run_stale_and_stop(self):
        import time
        m = MPPI(MPPIConfig(samples=16, horizon=4, iterations=1))
        r = m.plan(free, [1, 0])
        calls = []
        b = SimpleNamespace(_stop_requested=False, mppi_actuation=ActuationConfig(),
                            follower=SimpleNamespace(angular_sign=-1), send=calls.append,
                            dry_run=True, debug_dir=None)
        with patch('genie_rover.bridge.time.sleep') as sleep:
            Bridge._send_mppi(b, r, m, time.monotonic(), 'test')
            sleep.assert_not_called()
        self.assertGreater(calls[0].linear, 0)
        self.assertEqual(calls[-1].linear, 0)
        calls.clear()
        Bridge._send_mppi(b, r, m, time.monotonic()-10, 'test')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].linear, 0)


if __name__ == '__main__':
    unittest.main()
