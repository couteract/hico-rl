"""
B01R3 Protocol Validation Tests

Tests all B01R2 manager counterexamples that were incorrectly ACCEPTED:
1. Rank-2 model_input → REJECTED
2. Integer model_input → REJECTED
3. Negative decision_start_frame → REJECTED
4. Forbidden reward field → REJECTED
5. Discontinuous previous interval → REJECTED
6. Empty audit metadata → REJECTED (via forward_from_tokens)
7. Config state_dim wiring → FIXED and tested

Additional B01R3 requirements:
- Real DecisionRecorder usage (no MinimalRecorder)
- Exact rank/dtype/device/finite validation
- Interval continuity checks
- Checkpoint expected-contract tests
"""

import pytest
import torch
import os

# B01R5P2: Removed import-time CUDA_VISIBLE_DEVICES mutation
# Test environment should control this, not individual test files

from lerobot.memory.himem import (
    HiMemConfig,
    TokenSchema,
    CausalToken,
    EventFeatureEncoder,
    create_himem_encoder,
    RollingWindowStateManager,
    save_checkpoint,
    load_checkpoint,
)

# Import real recorder
from tools.recorded_rollout import DecisionRecorder


@pytest.fixture
def config():
    """Small config for CPU testing."""
    return HiMemConfig(
        window_size=8,
        memory_dim=64,
        backend="gru_reference",
        gru_hidden_dim=64,
        gru_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )


@pytest.fixture
def schema(config):
    """Token schema."""
    return config.token_schema


class TestManagerCounterexamples:
    """
    Test all 7 counterexamples from B01R2_MANAGER_AUDIT.json.
    Each must now be REJECTED.
    """

    def test_rank2_model_input_rejected(self, schema):
        """Counterexample 1: rank-2 model_input must be REJECTED."""
        # Create a rank-2 tensor instead of rank-1
        model_input = torch.randn(1, schema.token_dim)  # [1, token_dim] not [token_dim]

        token = CausalToken(
            model_input=model_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_ep",
                "decision_index": 0,
                "decision_start_frame": 0,
                "prev_executed_start": None,
                "prev_executed_end": None,
                "prev_executed_count": None,
                "end_reason": None,  # Fixed: decision 0 must have null end_reason
                "schema_version": "b01r_causal",
                "schema_hash": schema.compute_hash(),
            }
        )

        with pytest.raises(ValueError, match="must be rank 1"):
            token.validate(schema)

    def test_integer_model_input_rejected(self, schema):
        """Counterexample 2: integer model_input must be REJECTED."""
        # Create integer tensor instead of float
        model_input = torch.randint(0, 10, (schema.token_dim,))

        token = CausalToken(
            model_input=model_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_ep",
                "decision_index": 0,
                "decision_start_frame": 0,
                "prev_executed_start": None,
                "prev_executed_end": None,
                "prev_executed_count": None,
                "end_reason": None,  # Fixed: decision 0 must have null end_reason
                "schema_version": "b01r_causal",
                "schema_hash": schema.compute_hash(),
            }
        )

        with pytest.raises(ValueError, match="floating point dtype"):
            token.validate(schema)

    def test_negative_decision_start_frame_rejected(self, schema):
        """Counterexample 3: negative decision_start_frame must be REJECTED."""
        model_input = torch.randn(schema.token_dim)

        token = CausalToken(
            model_input=model_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_ep",
                "decision_index": 0,
                "decision_start_frame": -10,  # Negative!
                "prev_executed_start": None,
                "prev_executed_end": None,
                "prev_executed_count": None,
                "end_reason": "timeout",
                "schema_version": "b01r_causal",
                "schema_hash": schema.compute_hash(),
            }
        )

        with pytest.raises(ValueError, match="decision_start_frame must be non-negative"):
            token.validate(schema)

    def test_forbidden_reward_field_rejected(self, schema):
        """Counterexample 4: forbidden 'reward' field must be REJECTED."""
        model_input = torch.randn(schema.token_dim)

        token = CausalToken(
            model_input=model_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_ep",
                "decision_index": 0,
                "decision_start_frame": 0,
                "prev_executed_start": None,
                "prev_executed_end": None,
                "prev_executed_count": None,
                "end_reason": None,  # Fixed: decision 0 must have null end_reason
                "schema_version": "b01r_causal",
                "schema_hash": schema.compute_hash(),
                "reward": 1.5,  # Forbidden!
            }
        )

        with pytest.raises(ValueError, match="Forbidden field 'reward'"):
            token.validate(schema)

    def test_forbidden_success_failure_done_rejected(self, schema):
        """Test other forbidden fields: success, failure, done."""
        for forbidden_field in ["success", "failure", "done", "advantage", "value"]:
            model_input = torch.randn(schema.token_dim)

            token = CausalToken(
                model_input=model_input,
                audit_metadata={
                    "env_id": "test_env",
                    "episode_id": "test_ep",
                    "decision_index": 0,
                    "decision_start_frame": 0,
                    "prev_executed_start": None,
                    "prev_executed_end": None,
                    "prev_executed_count": None,
                    "end_reason": "timeout",
                    "schema_version": "b01r_causal",
                    "schema_hash": schema.compute_hash(),
                    forbidden_field: True,
                }
            )

            with pytest.raises(ValueError, match=f"Forbidden field '{forbidden_field}'"):
                token.validate(schema)

    def test_discontinuous_previous_interval_rejected(self, config, schema):
        """
        Counterexample 5: state manager must reject discontinuous previous interval.

        Decision 1 at frame 50 claims prev interval [100, 101), but decision 0 started at 0.
        """
        state_mgr = RollingWindowStateManager(config)

        # Decision 0 at frame 0
        token0_input = torch.zeros(schema.token_dim)
        # Set non-zero values only for visual_feature, state, and event_observable
        token0_input[:schema.visual_feature_dim] = torch.randn(schema.visual_feature_dim)
        offset = schema.visual_feature_dim
        token0_input[offset:offset + schema.state_dim] = torch.randn(schema.state_dim)
        # event_observable is at the end
        offset = (schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim +
                  1 + schema.visual_delta_dim + schema.state_delta_dim)
        token0_input[offset:] = torch.randn(schema.event_observable_dim)

        token0 = CausalToken(
            model_input=token0_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_ep",
                "decision_index": 0,
                "decision_start_frame": 0,
                "prev_executed_start": None,
                "prev_executed_end": None,
                "prev_executed_count": None,
                "end_reason": None,  # Fixed: decision 0 must have null end_reason
                "schema_version": "b01r_causal",
                "schema_hash": schema.compute_hash(),
            }
        )
        state_mgr.add_token(token0)

        # Decision 1 at frame 50, but claims prev interval [100, 101)
        # This is discontinuous: prev_start (100) != decision_0_start (0)
        token1_input = torch.randn(schema.token_dim)
        # Set has_previous to exactly 1.0 (required for decision 1)
        offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim
        token1_input[offset] = 1.0

        token1_bad = CausalToken(
            model_input=token1_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_ep",
                "decision_index": 1,
                "decision_start_frame": 50,
                "prev_executed_start": 100,  # Wrong!
                "prev_executed_end": 101,
                "prev_executed_count": 1,
                "end_reason": "queue_depleted",
                "schema_version": "b01r_causal",
                "schema_hash": schema.compute_hash(),
            }
        )

        with pytest.raises(ValueError, match="Interval discontinuity.*prev_start"):
            state_mgr.add_token(token1_bad)

    def test_empty_audit_metadata_rejected(self, config, schema):
        """
        Counterexample 6: forward_from_tokens with empty/missing audit metadata
        must be REJECTED (via validation in forward_from_tokens).
        """
        encoder = create_himem_encoder(config=config)

        # Create token with missing required audit field
        bad_token = CausalToken(
            model_input=torch.randn(schema.token_dim),
            audit_metadata={
                "env_id": "test_env",
                # Missing decision_index, decision_start_frame, etc.
            }
        )

        with pytest.raises(ValueError, match="Missing required audit field"):
            encoder.forward_from_tokens([bad_token])

    def test_config_state_dim_wiring_fixed(self, config):
        """
        Counterexample 7: simplified_ssm_state_dim=17 should create state_dim=17,
        not state_dim=num_layers.

        This was fixed in encoder.py line 56.
        """
        # Create config with experimental backend
        exp_config = HiMemConfig(
            window_size=8,
            memory_dim=64,
            backend="simplified_ssm_experimental",
            simplified_ssm_state_dim=17,
            simplified_ssm_num_layers=3,
            token_schema=TokenSchema(
                visual_feature_dim=32,
                visual_delta_dim=32,
                event_observable_dim=16,
            ),
        )

        encoder = create_himem_encoder(config=exp_config)

        # Check backend got correct state_dim
        assert encoder.backend.state_dim == 17, \
            f"Expected state_dim=17, got {encoder.backend.state_dim}"


class TestRealRecorder:
    """Test integration with real DecisionRecorder from tools.recorded_rollout."""

    def test_queue_depleted_reason(self, config, schema):
        """Test DecisionRecorder queue_depleted end reason."""
        recorder = DecisionRecorder()

        # Decision 0: plan 10 actions, execute all 10 → queue_depleted
        recorder.begin(frame=0, planned_count=10)
        recorder.finish(end=10, reason='queue_depleted')

        assert len(recorder.decisions) == 1
        dec = recorder.decisions[0]
        assert dec['decision_index'] == 0
        assert dec['chunk_start_frame'] == 0
        assert dec['chunk_end_frame'] == 10
        assert dec['executed_action_count'] == 10
        assert dec['end_reason'] == 'queue_depleted'

    def test_early_replan_reason(self, config, schema):
        """Test DecisionRecorder early_replan end reason."""
        recorder = DecisionRecorder()

        # Decision 0: plan 10, execute only 7 → early_replan
        recorder.begin(frame=0, planned_count=10)
        recorder.finish(end=7, reason='early_replan')

        dec = recorder.decisions[0]
        assert dec['executed_action_count'] == 7
        assert dec['end_reason'] == 'early_replan'

    def test_partial_execution_intervals(self, config, schema):
        """Test that recorded intervals are [start, end) and cover executed actions."""
        recorder = DecisionRecorder()

        # Decision 0: [0, 10)
        recorder.begin(frame=0, planned_count=10)
        recorder.finish(end=10, reason='queue_depleted')

        # Decision 1: [10, 17) - only 7 executed
        recorder.begin(frame=10, planned_count=10)
        recorder.finish(end=17, reason='terminal')

        assert len(recorder.decisions) == 2

        dec0 = recorder.decisions[0]
        assert dec0['chunk_start_frame'] == 0
        assert dec0['chunk_end_frame'] == 10
        assert dec0['executed_action_count'] == 10

        dec1 = recorder.decisions[1]
        assert dec1['chunk_start_frame'] == 10
        assert dec1['chunk_end_frame'] == 17
        assert dec1['executed_action_count'] == 7
        assert dec1['end_reason'] == 'terminal'

    def test_zero_execution_rejected(self):
        """DecisionRecorder must reject zero executed actions."""
        recorder = DecisionRecorder()

        recorder.begin(frame=0, planned_count=10)

        with pytest.raises(ValueError, match="Invalid actual executed count"):
            recorder.finish(end=0, reason='early_replan')

    def test_continuous_coverage(self):
        """Test that recorded intervals continuously cover [0, final_end)."""
        recorder = DecisionRecorder()

        # Build sequence covering [0, 52)
        recorder.begin(frame=0, planned_count=10)
        recorder.finish(end=10, reason='queue_depleted')

        recorder.begin(frame=10, planned_count=15)
        recorder.finish(end=25, reason='queue_depleted')

        recorder.begin(frame=25, planned_count=20)
        recorder.finish(end=45, reason='early_replan')

        recorder.begin(frame=45, planned_count=10)
        recorder.finish(end=52, reason='terminal')

        # Verify continuous coverage
        for i in range(len(recorder.decisions) - 1):
            curr = recorder.decisions[i]
            next_dec = recorder.decisions[i + 1]
            assert curr['chunk_end_frame'] == next_dec['chunk_start_frame'], \
                f"Gap between decision {i} and {i+1}"


class TestCheckpointExpectedContract:
    """Test checkpoint loading with expected config/schema/backend validation."""

    def test_window_size_mismatch_rejected(self, config, tmp_path):
        """Checkpoint with W=8 cannot be loaded with expected W=7."""
        encoder = create_himem_encoder(config=config)
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(encoder, ckpt_path)

        # Try to load with different window size expected
        expected_config = HiMemConfig(
            window_size=7,  # Different!
            memory_dim=config.memory_dim,
            backend=config.backend,
            gru_hidden_dim=config.gru_hidden_dim,
            token_schema=config.token_schema,
        )

        with pytest.raises(ValueError, match="Config field 'window_size' mismatch"):
            load_checkpoint(ckpt_path, expected_config=expected_config)

    def test_backend_mismatch_rejected(self, config, tmp_path):
        """GRU checkpoint cannot be loaded expecting Mamba2."""
        encoder = create_himem_encoder(config=config)  # GRU
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(encoder, ckpt_path)

        with pytest.raises(ValueError, match="Backend mismatch"):
            load_checkpoint(ckpt_path, expected_backend="mamba2")

    def test_schema_hash_mismatch_rejected(self, config, tmp_path):
        """Checkpoint with different schema cannot be loaded."""
        encoder = create_himem_encoder(config=config)
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(encoder, ckpt_path)

        # Different schema
        different_schema = TokenSchema(
            visual_feature_dim=64,  # Different!
            visual_delta_dim=64,
            event_observable_dim=16,
        )
        different_config = HiMemConfig(
            window_size=config.window_size,
            memory_dim=config.memory_dim,
            backend=config.backend,
            gru_hidden_dim=config.gru_hidden_dim,
            token_schema=different_schema,
        )

        with pytest.raises(ValueError, match="schema mismatch"):
            load_checkpoint(ckpt_path, expected_config=different_config)

    def test_memory_dim_mismatch_rejected(self, config, tmp_path):
        """Checkpoint with memory_dim=64 cannot be loaded expecting 128."""
        encoder = create_himem_encoder(config=config)
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(encoder, ckpt_path)

        expected_config = HiMemConfig(
            window_size=config.window_size,
            memory_dim=128,  # Different!
            backend=config.backend,
            gru_hidden_dim=128,  # Must also match for GRU
            token_schema=config.token_schema,
        )

        with pytest.raises(ValueError, match="Config field 'memory_dim' mismatch"):
            load_checkpoint(ckpt_path, expected_config=expected_config)

    def test_matching_config_loads_successfully(self, config, tmp_path):
        """Checkpoint loads successfully with matching expected config."""
        encoder = create_himem_encoder(config=config)
        encoder.eval()

        # Save
        ckpt_path = tmp_path / "test.pt"
        save_checkpoint(encoder, ckpt_path)

        # Load with exact same config
        loaded_encoder = load_checkpoint(ckpt_path, expected_config=config)
        loaded_encoder.eval()

        # Verify outputs match
        test_input = torch.randn(5, config.token_schema.token_dim)
        with torch.no_grad():
            out1 = encoder.forward(test_input, return_all=False)
            out2 = loaded_encoder.forward(test_input, return_all=False)

        assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-7)


class TestDtypeDeviceValidation:
    """Test that all tensors must have same dtype and device."""

    def test_mixed_dtype_rejected(self, schema):
        """visual_feature float32, state float64 → REJECTED."""
        feature_enc = EventFeatureEncoder(schema)

        visual_feature = torch.randn(schema.visual_feature_dim, dtype=torch.float32)
        state = torch.randn(schema.state_dim, dtype=torch.float64)  # Different!
        event_observable = torch.randn(schema.event_observable_dim, dtype=torch.float32)

        with pytest.raises(ValueError, match="state dtype.*visual_feature dtype"):
            feature_enc.encode_token(
                visual_feature=visual_feature,
                state=state,
                prev_action_summary=None,
                prev_visual_feature=None,
                prev_state=None,
                event_observable=event_observable,
                episode_id="test_ep",
                env_id="test_env",
                decision_index=0,
                decision_start_frame=0,
                prev_executed_start=None,
                prev_executed_end=None,
                prev_executed_count=None,
                end_reason="timeout",
                schema_hash=schema.compute_hash(),
            )

    def test_mixed_device_rejected(self, schema):
        """Tensors on different devices → REJECTED."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        feature_enc = EventFeatureEncoder(schema)

        visual_feature = torch.randn(schema.visual_feature_dim, device='cpu')
        state = torch.randn(schema.state_dim, device='cuda')  # Different!
        event_observable = torch.randn(schema.event_observable_dim, device='cpu')

        with pytest.raises(ValueError, match="state device.*visual_feature device"):
            feature_enc.encode_token(
                visual_feature=visual_feature,
                state=state,
                prev_action_summary=None,
                prev_visual_feature=None,
                prev_state=None,
                event_observable=event_observable,
                episode_id="test_ep",
                env_id="test_env",
                decision_index=0,
                decision_start_frame=0,
                prev_executed_start=None,
                prev_executed_end=None,
                prev_executed_count=None,
                end_reason="timeout",
                schema_hash=schema.compute_hash(),
            )

    def test_nan_inf_rejected(self, schema):
        """Tensors with NaN/Inf → REJECTED."""
        feature_enc = EventFeatureEncoder(schema)

        visual_feature = torch.randn(schema.visual_feature_dim)
        visual_feature[0] = float('nan')  # Inject NaN
        state = torch.randn(schema.state_dim)
        event_observable = torch.randn(schema.event_observable_dim)

        with pytest.raises(ValueError, match="visual_feature contains NaN or Inf"):
            feature_enc.encode_token(
                visual_feature=visual_feature,
                state=state,
                prev_action_summary=None,
                prev_visual_feature=None,
                prev_state=None,
                event_observable=event_observable,
                episode_id="test_ep",
                env_id="test_env",
                decision_index=0,
                decision_start_frame=0,
                prev_executed_start=None,
                prev_executed_end=None,
                prev_executed_count=None,
                end_reason="timeout",
                schema_hash=schema.compute_hash(),
            )
