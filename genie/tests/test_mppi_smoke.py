"""Mission wiring smoke test; SDK is always mocked, never contacts a rover.

Set MPPI_REAL_SAM=1 to also load actual SAM weights and infer a black frame.
Run from genie/ with python -m unittest discover -s tests -p test_mppi_smoke.py -v.
"""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from genie_rover import bridge as module


class MissionSmokeTests(unittest.TestCase):
    def test_mission_planning_recovery_and_stop(self):
        cfg_path = Path(__file__).resolve().parents[1] / 'configs/frodobot_rover.yaml'
        instances = []
        original = module.Bridge

        def factory(cfg, **kwargs):
            cfg['safety']['obstacle_persist_frames'] = 1
            cfg['safety']['loop_period_s'] = 0
            b = original(cfg, **kwargs)
            instances.append(b)
            step = b._step

            def bounded_step():
                step()
                if b.stats.iterations >= 1:
                    b.request_stop()

            b._step = bounded_step
            return b

        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bev = SimpleNamespace(traversability=np.ones((67, 134)),
                              observed=np.ones((67, 134)), stats={'bev_observed_cells': 8978})
        from genie_rover.mppi import MPPI
        original_plan = MPPI.plan
        calls = []

        def spy_plan(optimizer, *args, **kwargs):
            calls.append(kwargs.get('recovery', False))
            return original_plan(optimizer, *args, **kwargs)

        with patch.object(module, 'RoverClient') as sdk, \
             patch.object(module, 'PerceptionPipeline') as perception, \
             patch.object(module, 'Bridge', side_effect=factory), \
             patch.object(module, 'front_is_blocked', side_effect=[False, True]), \
             patch.object(module, 'plan_on_bev') as polynomial, \
             patch.object(module.signal, 'signal'), \
             patch.object(MPPI, 'plan', spy_plan), \
             patch('sys.argv', ['bridge', '--config', str(cfg_path), '--start-mission']):
            sdk.return_value.checkpoints.return_value = ([], 0)
            sdk.return_value.front_frame.side_effect = [(image, 1.), (image, 2.)]
            sdk.return_value.telemetry.return_value = SimpleNamespace(
                latitude=0, longitude=0, orientation=0, timestamp=1, raw={})
            perception.return_value.process.return_value = bev
            self.assertEqual(module.main(), 0)
            sdk.return_value.start_mission.assert_called_once()
            self.assertEqual(sdk.return_value.front_frame.call_count, 2)
            self.assertEqual(sdk.return_value.telemetry.call_count, 2)
            self.assertEqual(perception.return_value.process.call_count, 2)
            polynomial.assert_not_called()
            sdk.return_value.control.assert_not_called()
        self.assertEqual(calls, [False, True])
        self.assertEqual(instances[0].stats.errors, 0)
        self.assertTrue(instances[0].mppi_recovery.active)
        self.assertIsNotNone(instances[0].pmap)

    @unittest.skipUnless(os.environ.get('MPPI_REAL_SAM') == '1', 'optional real SAM inference')
    def test_real_sam_black_frame(self):
        from genie_rover.perception import PerceptionPipeline
        cfg = yaml.safe_load(Path('configs/frodobot_rover.yaml').read_text())
        pipeline = PerceptionPipeline(cfg)
        result = pipeline.process(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertEqual(result.traversability.shape, result.observed.shape)
        self.assertTrue(np.isfinite(result.traversability).all())
        self.assertGreater(result.traversability.size, 0)


if __name__ == '__main__':
    unittest.main()
