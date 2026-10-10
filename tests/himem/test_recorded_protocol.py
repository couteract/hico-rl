"""
B01 Test: Recorded protocol (migrated from Window A)

Tests decision recorder and LIBERO environment protocol:
1. Partial replanning with early termination
2. No predicted action tail beyond actual execution
3. Explicit init_state_id after reset, seed-independent
4. Timeout terminal observation without hidden reset
5. Success preempts timeout

Original: A_libero_baseline/tests/workstream_a/test_recorded_protocol.py
Migration: Removed path injection, tests Window B implementation directly
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

# Import from Window B's local implementation
from lerobot.envs.libero import LiberoEnv


class MockDecisionRecorder:
    """Minimal recorder for testing protocol without full recorded_rollout.py"""
    def __init__(self):
        self.decisions = []
        self._active = None

    def begin(self, start_frame, horizon):
        if self._active is not None:
            # Early replan: mark previous as incomplete
            self._active['end_reason'] = 'early_replan'
            self._active['chunk_end_frame'] = start_frame
            self._active['executed_action_count'] = start_frame - self._active['chunk_start_frame']
            self.decisions.append(self._active)

        self._active = {
            'chunk_start_frame': start_frame,
            'predicted_horizon': horizon,
            'environment_reset': len(self.decisions) == 0,
        }

    def finish(self, end_frame, reason):
        if self._active is None:
            raise RuntimeError("No active decision to finish")

        self._active['chunk_end_frame'] = end_frame
        self._active['end_reason'] = reason
        self._active['executed_action_count'] = end_frame - self._active['chunk_start_frame']
        self.decisions.append(self._active)
        self._active = None


class RecordedProtocolTest(unittest.TestCase):
    def test_partial_replan_and_terminal(self):
        """Early replanning marks previous decision incomplete"""
        r = MockDecisionRecorder()
        r.begin(0, 5)  # Plan 5 actions from frame 0
        r.begin(3, 5)  # Replan at frame 3 (only executed 3 actions)
        r.finish(4, 'success')

        self.assertEqual([d['executed_action_count'] for d in r.decisions], [3, 1])
        self.assertEqual(r.decisions[0]['end_reason'], 'early_replan')
        self.assertEqual([d['environment_reset'] for d in r.decisions], [True, False])

    def test_no_predicted_tail(self):
        """Decision boundary ends at actual execution, not predicted horizon"""
        r = MockDecisionRecorder()
        r.begin(0, 50)  # Predict 50-step horizon
        r.finish(2, 'timeout')  # But only executed 2 steps

        # Boundary is [0, 2), not [0, 50)
        self.assertEqual(r.decisions[0]['chunk_end_frame'], 2)

        # Cannot finish again after already finished
        with self.assertRaises(RuntimeError):
            r.finish(3, 'timeout')

    def create_mock_env(self):
        """Create mock LiberoEnv for protocol testing"""
        e = LiberoEnv.__new__(LiberoEnv)
        e._env = Mock()
        e._env.robots = []
        e._env.reset.return_value = 'reset_obs'
        e._env.set_init_state.side_effect = lambda state: state
        e._format_raw_obs = lambda x: x
        e._init_states = np.arange(50)
        e.init_states = True
        e.init_state_id = 0
        e._reset_stride = 1
        e.num_steps_wait = 0
        e.control_mode = 'relative'
        e._max_episode_steps = 2
        e.task = 'test_task'
        e.task_id = 0
        return e

    def test_explicit_state_after_reset_seed_independent(self):
        """init_state_id in options overrides seed-based sampling"""
        e = self.create_mock_env()

        # Explicit init_state_id=17 should be used
        obs, info = e.reset(seed=123, options={'init_state_id': 17})
        self.assertEqual(obs, 17)
        self.assertEqual(info['init_state_id'], 17)

        # reset() must be called before set_init_state()
        calls = [x[0] for x in e._env.mock_calls]
        self.assertLess(calls.index('reset'), calls.index('set_init_state'))

        # Different seed but same init_state_id → same state
        obs, _ = e.reset(seed=999, options={'init_state_id': 17})
        self.assertEqual(obs, 17)

        # Invalid init_state_id should raise
        with self.assertRaises(ValueError):
            e.reset(seed=1, options={'init_state_id': 50})

    def test_timeout_final_observation_no_hidden_reset(self):
        """Timeout returns final observation without hidden reset"""
        e = self.create_mock_env()
        e.reset(seed=1, options={'init_state_id': 4})
        e._env.reset.reset_mock()

        e._env.check_success.return_value = False
        e._env.step.return_value = ('terminal_obs', 0., False, {})

        # First step: not done
        _, _, done, trunc, _ = e.step(np.zeros(7))
        self.assertFalse(done or trunc)

        # Second step: timeout (max_episode_steps=2)
        obs, _, done, trunc, info = e.step(np.zeros(7))
        self.assertFalse(done)
        self.assertTrue(trunc)
        self.assertEqual(obs, 'terminal_obs')
        self.assertFalse(info['final_info']['is_success'])

        # No hidden reset should have been called
        e._env.reset.assert_not_called()

    def test_success_preempts_timeout(self):
        """Success termination takes priority over timeout"""
        e = self.create_mock_env()
        e.reset(seed=1)
        e._elapsed_steps = 1  # Would timeout on next step

        e._env.check_success.return_value = True
        e._env.step.return_value = ('success_obs', 1., True, {})

        _, _, done, trunc, info = e.step(np.zeros(7))
        self.assertTrue(done)
        self.assertFalse(trunc)
        self.assertTrue(info['final_info']['is_success'])


if __name__ == '__main__':
    unittest.main()
