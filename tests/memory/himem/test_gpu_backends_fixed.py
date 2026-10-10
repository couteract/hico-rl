"""
GPU smoke tests for B01R HiMem backends.

Tests all three backends (gru_reference, mamba2, simplified_ssm_experimental)
for basic forward/backward pass and numerical stability on GPU.

B01R5P2: Uses shared lock via workstream_gpu.sh wrapper.
"""

import pytest
import torch
import os
from pathlib import Path

# B01R5P2: Module-level GPU opt-in check - must be EXACTLY "1"
pytestmark = pytest.mark.skipif(
    os.environ.get("HIMEM_RUN_GPU_TESTS") != "1" or os.environ.get("HIMEM_GPU_LOCK_HELD") != "1",
    reason="GPU tests require HIMEM_RUN_GPU_TESTS=1 and HIMEM_GPU_LOCK_HELD=1"
)

from lerobot.memory.himem import (
    HiMemConfig,
    TokenSchema,
    EventFeatureEncoder,
    create_himem_encoder,
)


@pytest.fixture
def device():
    """CUDA device - lock already held by wrapper."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def small_config():
    """Small config for fast GPU testing."""
    return HiMemConfig(
        window_size=4,
        memory_dim=64,
        backend="gru_reference",
        gru_hidden_dim=64,
        gru_num_layers=1,
        mamba2_d_model=64,
        mamba2_d_state=32,
        mamba2_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )


def create_simple_tokens(num_tokens: int, config: HiMemConfig, device: torch.device):
    """Create simple synthetic tokens."""
    feature_enc = EventFeatureEncoder(config.token_schema)
    tokens = []

    for i in range(num_tokens):
        # Use deterministic seed
        torch.manual_seed(1000 + i)

        visual = torch.randn(config.token_schema.visual_feature_dim, device=device) * 0.1
        state = torch.randn(config.token_schema.state_dim, device=device) * 0.1
        event_obs = torch.randn(config.token_schema.event_observable_dim, device=device) * 0.1

        if i > 0:
            # Has previous
            torch.manual_seed(1000 + i - 1)
            prev_visual = torch.randn(config.token_schema.visual_feature_dim, device=device) * 0.1
            prev_state = torch.randn(config.token_schema.state_dim, device=device) * 0.1

            # Simple action summary
            prev_action = torch.randn(config.token_schema.action_summary_dim, device=device) * 0.1

            prev_exec_start = (i - 1) * 10
            prev_exec_end = i * 10
            prev_exec_count = 10
            end_reason = "timeout"
        else:
            prev_visual = None
            prev_state = None
            prev_action = None
            prev_exec_start = None
            prev_exec_end = None
            prev_exec_count = None
            end_reason = None

        token = feature_enc.encode_token(
            visual_feature=visual,
            state=state,
            prev_action_summary=prev_action,
            prev_visual_feature=prev_visual,
            prev_state=prev_state,
            event_observable=event_obs,
            episode_id="gpu_test",
            env_id="test_env",
            decision_index=i,
            decision_start_frame=i * 10,
            prev_executed_start=prev_exec_start,
            prev_executed_end=prev_exec_end,
            prev_executed_count=prev_exec_count,
            end_reason=end_reason,
            schema_hash=config.token_schema.compute_hash(),
        )
        tokens.append(token)

    return tokens


class TestGRUBackend:
    """Test GRU reference backend on GPU."""

    def test_forward_pass(self, small_config, device):
        """Forward pass completes without error."""
        config = small_config
        config.backend = "gru_reference"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_simple_tokens(4, config, device)

        with torch.no_grad():
            output = encoder.forward_from_tokens(tokens, return_all=False)

        assert output.shape == (config.memory_dim,)
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()

    def test_backward_pass(self, small_config, device):
        """Backward pass computes gradients."""
        config = small_config
        config.backend = "gru_reference"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.train()

        tokens = create_simple_tokens(4, config, device)

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"


class TestMamba2Backend:
    """Test official Mamba2 backend on GPU."""

    def test_forward_pass(self, small_config, device):
        """Forward pass completes without error."""
        config = small_config
        config.backend = "mamba2"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_simple_tokens(4, config, device)

        with torch.no_grad():
            output = encoder.forward_from_tokens(tokens, return_all=False)

        assert output.shape == (config.memory_dim,)
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()

    def test_backward_pass(self, small_config, device):
        """Backward pass computes gradients."""
        config = small_config
        config.backend = "mamba2"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.train()

        tokens = create_simple_tokens(4, config, device)

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for param in encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"


class TestSimplifiedSSMBackend:
    """Test SimplifiedSSM experimental backend on GPU."""

    def test_forward_pass(self, small_config, device):
        """Forward pass completes without error."""
        config = small_config
        config.backend = "simplified_ssm_experimental"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_simple_tokens(4, config, device)

        with torch.no_grad():
            output = encoder.forward_from_tokens(tokens, return_all=False)

        assert output.shape == (config.memory_dim,)
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()

    def test_backward_pass(self, small_config, device):
        """Backward pass computes gradients."""
        config = small_config
        config.backend = "simplified_ssm_experimental"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.train()

        tokens = create_simple_tokens(4, config, device)

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for param in encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"
