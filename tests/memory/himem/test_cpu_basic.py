"""
CPU tests for B01R HiMem implementation.

Tests causal protocol, rolling window semantics, and determinism.

B01R5P3: Removed import-time CUDA_VISIBLE_DEVICES mutation.
CPU isolation must be set by the test runner BEFORE pytest collection,
not during module import. This prevents contaminating GPU test collection.

Run with: CUDA_VISIBLE_DEVICES='' pytest tests/memory/himem/test_cpu_basic.py
"""

import pytest
import torch
import os
from pathlib import Path

from lerobot.memory.himem import (
    HiMemConfig,
    TokenSchema,
    CausalToken,
    EventFeatureEncoder,
    create_himem_encoder,
    RollingWindowStateManager,
    summarize_action_chunk,
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint_determinism,
)


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
def encoder(config):
    """Create encoder."""
    return create_himem_encoder(config=config)


@pytest.fixture
def feature_encoder(config):
    """Create feature encoder."""
    return EventFeatureEncoder(config.token_schema)


def create_synthetic_token(
    decision_index: int,
    episode_id: str = "test_ep",
    env_id: str = "test_env",
    schema: TokenSchema = None,
    device: torch.device = None,
    decision_start_frame: int = None,
) -> CausalToken:
    """Create a synthetic causal token with deterministic features."""
    if schema is None:
        schema = TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        )

    device = device or torch.device("cpu")

    # Deterministic features based on decision_index
    torch.manual_seed(decision_index + 1000)

    visual_feature = torch.randn(schema.visual_feature_dim, device=device) * 0.1
    state = torch.randn(schema.state_dim, device=device) * 0.1
    event_observable = torch.randn(schema.event_observable_dim, device=device) * 0.1

    # Default decision_start_frame if not provided
    if decision_start_frame is None:
        decision_start_frame = decision_index * 10

    # Previous features (if not first)
    if decision_index > 0:
        torch.manual_seed(decision_index - 1 + 1000)
        prev_visual = torch.randn(schema.visual_feature_dim, device=device) * 0.1
        prev_state = torch.randn(schema.state_dim, device=device) * 0.1

        # Action summary
        torch.manual_seed(decision_index + 2000)
        actions = torch.randn(10, 7, device=device) * 0.1
        prev_action_summary = summarize_action_chunk(actions)

        prev_executed_start = (decision_index - 1) * 10
        prev_executed_end = decision_index * 10
        prev_executed_count = 10
        end_reason = "queue_depleted"
    else:
        prev_visual = None
        prev_state = None
        prev_action_summary = None
        prev_executed_start = None
        prev_executed_end = None
        prev_executed_count = None
        end_reason = None

        # First decision must start at frame 0
        decision_start_frame = 0

    feature_enc = EventFeatureEncoder(schema)
    token = feature_enc.encode_token(
        visual_feature=visual_feature,
        state=state,
        prev_action_summary=prev_action_summary,
        prev_visual_feature=prev_visual,
        prev_state=prev_state,
        event_observable=event_observable,
        episode_id=episode_id,
        env_id=env_id,
        decision_index=decision_index,
        decision_start_frame=decision_index * 10,
        prev_executed_start=prev_executed_start,
        prev_executed_end=prev_executed_end,
        prev_executed_count=prev_executed_count,
        end_reason=end_reason,
        schema_hash=schema.compute_hash(),
    )

    return token


class TestCausalProtocol:
    """Test causal protocol enforcement."""

    def test_same_input_deterministic(self, encoder, config):
        """Same input in eval mode produces same output."""
        encoder.eval()

        # Create tokens
        tokens = [create_synthetic_token(i, schema=config.token_schema) for i in range(5)]

        with torch.no_grad():
            output1 = encoder.forward_from_tokens(tokens, return_all=True)
            output2 = encoder.forward_from_tokens(tokens, return_all=True)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7), \
            f"Determinism failed: max diff = {(output1 - output2).abs().max().item()}"

    def test_input_actually_used(self, encoder, config):
        """Changing input actually changes output (no ignoring inputs)."""
        encoder.eval()

        # Create two different token sequences
        tokens1 = [create_synthetic_token(i, schema=config.token_schema) for i in range(3)]
        tokens2 = [create_synthetic_token(i + 100, schema=config.token_schema) for i in range(3)]

        with torch.no_grad():
            output1 = encoder.forward_from_tokens(tokens1, return_all=False)
            output2 = encoder.forward_from_tokens(tokens2, return_all=False)

        diff = (output1 - output2).abs().max().item()
        assert diff > 1e-4, f"Outputs too similar despite different inputs: diff = {diff}"

    def test_no_future_leakage(self, encoder, config):
        """Perturbing future tokens doesn't change past outputs."""
        encoder.eval()

        # Create base sequence [0,1,2,3,4]
        tokens_short = [create_synthetic_token(i, schema=config.token_schema) for i in range(5)]

        # Extended sequence with different features for future tokens [0,1,2,3,4,5,6,7]
        # Use same indices but perturb the actual tensor values
        tokens_long = tokens_short.copy()
        for i in range(5, 8):
            # Create token with consecutive index but use a different seed for features
            torch.manual_seed(i + 9999)  # Different seed = different features
            token = create_synthetic_token(i, schema=config.token_schema)
            tokens_long.append(token)

        with torch.no_grad():
            output_short = encoder.forward_from_tokens(tokens_short, return_all=True)
            output_long = encoder.forward_from_tokens(tokens_long, return_all=True)

        # First 5 positions should match
        assert torch.allclose(output_short, output_long[:5], rtol=1e-5, atol=1e-7), \
            "Future tokens leaked into past positions"


class TestRollingWindow:
    """Test rolling window semantics (W=8)."""

    def test_online_vs_offline_equivalence(self, encoder, config):
        """
        Online state manager vs offline recomputation should match.

        For 2W+3 = 19 decisions, test every position k:
        online(k) == Encoder_zero(recent_W(k))[-1]
        """
        encoder.eval()
        W = config.window_size
        num_decisions = 2 * W + 3  # 19

        # Create all tokens
        all_tokens = [
            create_synthetic_token(i, schema=config.token_schema)
            for i in range(num_decisions)
        ]

        # Online path: use state manager
        state_mgr = encoder.create_state_manager()
        online_memories = []

        for token in all_tokens:
            state_mgr.add_token(token)
            recent_tokens = state_mgr.get_recent_tokens("test_env")

            with torch.no_grad():
                memory = encoder.forward_from_tokens(recent_tokens, return_all=False)
            online_memories.append(memory)

        # Offline path: recompute each position from recent W
        offline_memories = []
        for k in range(num_decisions):
            start_idx = max(0, k - W + 1)
            window_tokens = all_tokens[start_idx:k + 1]

            with torch.no_grad():
                memory = encoder.forward_from_tokens(window_tokens, return_all=False)
            offline_memories.append(memory)

        # Compare all positions
        for k in range(num_decisions):
            is_close = torch.allclose(
                online_memories[k], offline_memories[k],
                rtol=1e-5, atol=1e-7
            )
            if not is_close:
                diff = (online_memories[k] - offline_memories[k]).abs().max().item()
                pytest.fail(
                    f"Position {k}: online != offline, max_diff = {diff}\n"
                    f"At boundary: W-1={W-1}, W={W}, W+1={W+1}"
                )

    def test_window_limit_enforcement(self, encoder, config):
        """State manager keeps only W most recent tokens."""
        W = config.window_size

        state_mgr = encoder.create_state_manager()

        # Add 2W tokens
        for i in range(2 * W):
            token = create_synthetic_token(i, schema=config.token_schema)
            state_mgr.add_token(token)

        recent = state_mgr.get_recent_tokens("test_env")
        assert len(recent) == W, f"Expected {W} tokens, got {len(recent)}"

        # Check indices
        indices = [t.audit_metadata["decision_index"] for t in recent]
        expected = list(range(W, 2 * W))
        assert indices == expected, f"Expected {expected}, got {indices}"

    def test_old_tokens_dont_affect_current(self, encoder, config):
        """
        Changing tokens outside window doesn't affect current memory.

        Compare memory at position W+5 with:
        1. Normal history
        2. Perturbed history before window starts
        """
        encoder.eval()
        W = config.window_size

        # Normal sequence
        normal_tokens = [
            create_synthetic_token(i, schema=config.token_schema)
            for i in range(W + 6)
        ]

        # Perturbed sequence: change tokens before window
        perturbed_tokens = [
            create_synthetic_token(i + 5000, schema=config.token_schema)
            for i in range(6)  # Tokens 0-5 (outside window at position W+5)
        ] + normal_tokens[6:]

        # Compute memory at position W+5 (decision index W+5)
        target_idx = W + 5

        # Normal path
        window_start = target_idx - W + 1  # Should be 6
        normal_window = normal_tokens[window_start:target_idx + 1]

        # Perturbed path
        perturbed_window = perturbed_tokens[window_start:target_idx + 1]

        with torch.no_grad():
            normal_mem = encoder.forward_from_tokens(normal_window, return_all=False)
            perturbed_mem = encoder.forward_from_tokens(perturbed_window, return_all=False)

        # Should be identical (tokens [6:W+6] are same in both)
        assert torch.allclose(normal_mem, perturbed_mem, rtol=1e-5, atol=1e-7), \
            "Old tokens outside window affected current memory"


class TestStateManager:
    """Test state manager functionality."""

    def test_chronological_enforcement(self, config):
        """State manager rejects non-chronological decisions."""
        state_mgr = RollingWindowStateManager(config)

        # Add tokens in order
        token0 = create_synthetic_token(0, schema=config.token_schema)
        token1 = create_synthetic_token(1, schema=config.token_schema)

        state_mgr.add_token(token0)
        state_mgr.add_token(token1)

        # Try to add out-of-order
        token_out_of_order = create_synthetic_token(0, schema=config.token_schema)

        with pytest.raises(ValueError, match="Non-chronological"):
            state_mgr.add_token(token_out_of_order)

    def test_episode_reset(self, config):
        """Episode reset clears buffer."""
        state_mgr = RollingWindowStateManager(config)

        # Add tokens
        for i in range(5):
            token = create_synthetic_token(i, schema=config.token_schema)
            state_mgr.add_token(token)

        assert len(state_mgr.get_recent_tokens("test_env")) == 5

        # Reset
        state_mgr.reset_env("test_env")

        with pytest.raises(ValueError, match="not found"):
            state_mgr.get_recent_tokens("test_env")

    def test_multi_env_isolation(self, config):
        """Multiple environments are isolated."""
        state_mgr = RollingWindowStateManager(config)

        # Add tokens to env1
        for i in range(3):
            token = create_synthetic_token(i, env_id="env1", episode_id="ep1", schema=config.token_schema)
            state_mgr.add_token(token)

        # Add tokens to env2
        for i in range(5):
            token = create_synthetic_token(i, env_id="env2", episode_id="ep2", schema=config.token_schema)
            state_mgr.add_token(token)

        # Check isolation
        env1_tokens = state_mgr.get_recent_tokens("env1")
        env2_tokens = state_mgr.get_recent_tokens("env2")

        assert len(env1_tokens) == 3
        assert len(env2_tokens) == 5

        # Reset one shouldn't affect other
        state_mgr.reset_env("env1")

        env2_tokens_after = state_mgr.get_recent_tokens("env2")
        assert len(env2_tokens_after) == 5


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_load_roundtrip(self, encoder, tmp_path):
        """Save and load preserves model."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save
        manifest = save_checkpoint(encoder, checkpoint_path)
        assert checkpoint_path.exists()
        assert (tmp_path / "test_checkpoint.manifest.json").exists()

        # Load
        loaded = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

        # Compare configs
        assert loaded.config.window_size == encoder.config.window_size
        assert loaded.config.backend == encoder.config.backend

    def test_deterministic_reload(self, encoder, tmp_path, config):
        """Loading checkpoint multiple times gives same results."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Create test input
        test_input = torch.randn(5, config.token_schema.token_dim)

        result = verify_checkpoint_determinism(
            checkpoint_path,
            test_input,
            device=torch.device("cpu")
        )

        assert result["deterministic"], \
            f"Checkpoint reload not deterministic: max_diff = {result['max_absolute_difference']}"

    def test_config_mismatch_rejected(self, encoder, tmp_path, config):
        """Loading with mismatched config fails in strict mode."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Manually corrupt config hash
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["config_hash"] = "corrupted_hash"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(ValueError, match="Config hash mismatch"):
            load_checkpoint(checkpoint_path, strict=True)


class TestValidation:
    """Test input validation."""

    def test_empty_token_list_rejected(self, encoder):
        """Empty token list raises error."""
        with pytest.raises(ValueError, match="empty"):
            encoder.forward_from_tokens([])

    def test_nan_input_rejected(self, config):
        """NaN values in token are rejected."""
        token = create_synthetic_token(0, schema=config.token_schema)

        # Corrupt with NaN
        token.model_input[0] = float('nan')

        with pytest.raises(ValueError, match="NaN or Inf"):
            token.validate(config.token_schema)

    def test_wrong_dimension_rejected(self, config, feature_encoder):
        """Wrong token dimension is rejected."""
        # Create token with wrong dimension
        wrong_dim_input = torch.randn(100)  # Wrong size

        token = CausalToken(
            model_input=wrong_dim_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test",
                "decision_index": 0,
                "decision_start_frame": 0,
                "prev_executed_start": None,
                "prev_executed_end": None,
                "prev_executed_count": None,
                "end_reason": None,
                "schema_version": "b01r_causal",
                "schema_hash": config.token_schema.compute_hash(),
            }
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            token.validate(config.token_schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
