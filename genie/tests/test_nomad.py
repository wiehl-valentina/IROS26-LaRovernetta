"""NoMaD adapter tests with synthetic proposals; no weights, GPU or network."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

from genie_rover.nomad import NoMaDConfig, NoMaDPlanner, NoMaDRunner, decode_actions, candidate_bank
from genie_path_planner.planner import PlannerConfig, plan_on_bev
from genie_rover.bridge import Bridge
from genie_rover.mppi_adapter import build_backends


class FakeRunner:
    context_length = 2

    def sample(self, images):
        return np.array([[[0, 0], [-.3, .5], [-.6, 1]],
                         [[0, 0], [.3, .5], [.6, 1]]], dtype=float)


class NoMaDTests(unittest.TestCase):
    def ready_planner(self, runner=None):
        planner = NoMaDPlanner(NoMaDConfig(), runner=runner or FakeRunner())
        img = np.zeros((8, 8, 3), np.uint8)
        planner.observe(img, 10, 0)
        planner.observe(img, 11, 1)
        return planner

    def test_decode_deltas_scale_sign_and_origin(self):
        cfg = NoMaDConfig(waypoint_scale_m=.2)
        # normalized deltas corresponding to forward=1, left=0.5
        d = np.array([1, .5])
        normalized = 2*(d-np.array(cfg.action_min))/(np.array(cfg.action_max)-cfg.action_min)-1
        result = decode_actions(np.tile(normalized, (2, 3, 1)), cfg)
        self.assertEqual(result.shape, (2, 4, 2))
        np.testing.assert_allclose(result[0], [[0, 0], [-.1, .2], [-.2, .4], [-.3, .6]], atol=1e-6)

    def test_context_duplicate_interval_and_gap(self):
        p = NoMaDPlanner(NoMaDConfig(), runner=FakeRunner())
        image = np.zeros((8, 8, 3), np.uint8)
        p.observe(image, 1, 0)
        self.assertFalse(p.ready)
        p.observe(image, 1, 1)
        self.assertFalse(p.ready)
        p.observe(image, 2, .1)
        self.assertEqual(len(p.frames), 1)
        p.observe(image, 3, 1)
        self.assertTrue(p.ready)
        p.observe(image, 4, 10)
        self.assertFalse(p.ready)
        self.assertEqual(len(p.frames), 1)
        p.observe(image, 0, 11)  # SDK stream restarted
        self.assertEqual(len(p.frames), 1)

    def test_filters_reverse_stationary_outside_and_nan(self):
        paths = np.array([[[0, 0], [0, .5], [.1, 1]],
                          [[0, 0], [0, -.1], [0, .2]],
                          [[0, 0], [0, 0], [0, 0]],
                          [[0, 0], [0, .5], [5, 1]],
                          [[0, 0], [np.nan, .5], [0, 1]]])
        bank = candidate_bank(paths, (41, 81), .05, 2, 100, 51)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank[0].shape, (51, 2))

    def test_gps_ranks_and_preserves_rectangular_metric_coordinates(self):
        p = self.ready_planner()
        # Deliberately different lateral/forward scales.
        bev = np.ones((41, 81), dtype=np.float32)
        cfg = PlannerConfig(grid_size=100, use_clustering=False, footprint_px=3, smooth_kernel=1)
        with patch('genie_path_planner.planner.sample_paths_polynomial') as poly:
            plan = p.plan(bev, np.ones_like(bev), .6, 2, .05, 3, cfg)
            poly.assert_not_called()
        np.testing.assert_allclose(plan.final_path_xy_m[0], [0, 0], atol=1e-5)
        np.testing.assert_allclose(plan.final_path_xy_m[-1], [.6, 1], atol=1e-5)
        self.assertEqual(plan.metadata['generator'], 'nomad')
        self.assertEqual(len(plan.candidate_paths), 2)

    def test_obstacle_rejects_gps_preferred_branch(self):
        p = self.ready_planner()
        bev = np.ones((81, 81), dtype=np.float32)
        # Right branch near its end, beyond the first 60% of its path.
        bev[39:48, 49:56] = 0
        cfg = PlannerConfig(grid_size=81, use_clustering=False, footprint_px=3, smooth_kernel=1)
        plan = p.plan(bev, np.ones_like(bev), .6, 2, .05, 2, cfg)
        self.assertLess(plan.final_path_xy_m[-1, 0], 0)
        self.assertEqual(len(plan.filtered_paths), 1)

    def test_no_candidates_is_empty_plan_not_polynomial_fallback(self):
        runner = FakeRunner()
        runner.sample = lambda images: np.array([[[0, 0], [0, -.5], [0, -1]]])
        p = self.ready_planner(runner)
        with patch('genie_path_planner.planner.sample_paths_polynomial') as poly:
            plan = p.plan(np.ones((41, 81)), np.ones((41, 81)), 0, 1, .05, 2, PlannerConfig())
            poly.assert_not_called()
        self.assertEqual(len(plan.final_path_xy_m), 0)

    def test_missing_checkpoint_and_real_scale_gate(self):
        with self.assertRaises(FileNotFoundError):
            NoMaDRunner(NoMaDConfig(model_config_path='/tmp/nonexistent-nomad-config-test.yaml'))
        with self.assertRaises(ValueError):
            NoMaDPlanner(NoMaDConfig(), dry_run=False, runner=FakeRunner())

    def test_bad_config(self):
        for kwargs in ({'num_samples': 1.5}, {'waypoint_scale_m': 0}, {'goal_weight': float('nan')},
                       {'action_min': [1]}, {'max_context_gap_s': .01}):
            with self.assertRaises(ValueError):
                NoMaDConfig(**kwargs)


class NoMaDBridgeTests(unittest.TestCase):
    def make_bridge(self, algorithm='nomad', recovery='legacy'):
        cfg = {'navigation': {'trajectory_algorithm': algorithm, 'max_linear': .7,
                              'max_angular': .45, 'angular_sign': -1},
               'safety': {'recovery_algorithm': recovery}, 'memory': {'enabled': False},
               'rover': {'base_url': 'http://unused'},
               'projection': {'resolution_m_per_px': .03, 'forward_range_m': 2., 'side_range_m': 2.},
               'planner': {'use_clustering': False}, 'nomad': {'ignored_if_disabled': True}}
        if algorithm == 'nomad':
            cfg['nomad'] = {}
        with patch('genie_rover.bridge.RoverClient'), patch('genie_rover.bridge.PerceptionPipeline'), \
                patch('genie_rover.nomad.NoMaDRunner', return_value=FakeRunner()) as load:
            b = Bridge(cfg)
            self.assertEqual(load.call_count, int(algorithm == 'nomad'))
        b.client.front_frame.return_value = (np.zeros((8, 8, 3), np.uint8), 1.)
        b.client.telemetry.return_value = SimpleNamespace(latitude=0, longitude=0, orientation=0,
                                                        timestamp=1, raw={})
        b.perception.process.return_value = SimpleNamespace(traversability=np.ones((67, 134)),
                observed=np.ones((67, 134)), stats={'bev_observed_cells': 100})
        b.send = MagicMock()
        return b

    def test_disabled_ignores_config_and_ml_imports(self):
        with patch.dict(sys.modules, {'torch': None, 'diffusers': None, 'vint_train': None, 'diffusion_policy': None}):
            for name in ('polynomial', 'mppi'):
                b = self.make_bridge(name)
                self.assertIsNone(b.nomad)

    def test_warmup_does_not_trigger_recovery(self):
        b = self.make_bridge()
        with patch.object(b, '_recover') as recover:
            b._step()
            recover.assert_not_called()
        self.assertEqual(b.stats.plans_empty, 0)
        self.assertEqual(b.send.call_args.args[0].linear, 0)

    def test_nomad_candidates_enter_existing_planner_and_follower(self):
        b = self.make_bridge()
        b.nomad.observe(np.zeros((8, 8, 3), np.uint8), 0., 0.)
        b.nomad.config.max_context_gap_s = 1e12
        with patch('genie_path_planner.planner.sample_paths_polynomial') as poly:
            b._step()
            poly.assert_not_called()
        self.assertEqual(b.stats.plans_ok, 1)
        command = b.send.call_args.args[0]
        self.assertGreater(abs(command.linear)+abs(command.angular), 0)

    def test_nomad_with_mppi_recovery(self):
        b = self.make_bridge(recovery='mppi')
        self.assertIsNotNone(b.nomad)
        self.assertIsNotNone(b.mppi_recovery)
        self.assertIsNone(b.mppi)

    def test_empty_nomad_plan_uses_selected_recovery(self):
        b = self.make_bridge()
        b.nomad = MagicMock()
        b.nomad.ready = True
        b.nomad.config.max_observation_age_s = 10
        b.nomad.plan.return_value.final_path_xy_m = np.empty((0, 2))
        with patch.object(b, '_recover') as recover, \
                patch('genie_path_planner.planner.sample_paths_polynomial') as poly:
            for _ in range(b.recovery_after_empty):
                b._step()
            recover.assert_called_once()
            poly.assert_not_called()
        self.assertEqual(b.stats.plans_empty, b.recovery_after_empty)

    def test_stale_inference_never_sends_motion(self):
        b = self.make_bridge()
        b.nomad = MagicMock()
        b.nomad.ready = True
        b.nomad.config.max_observation_age_s = .1
        with patch('genie_rover.bridge.time.monotonic', side_effect=[0., 1.]):
            b._step()
        self.assertTrue(all(call.args[0].linear == 0 for call in b.send.call_args_list))


if __name__ == '__main__':
    unittest.main()
