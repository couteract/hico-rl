"""Simple GPU initialization test."""

import torch
import pytest
import os

# B01R5P2: Module-level GPU opt-in check - must be EXACTLY "1"
pytestmark_gpu = pytest.mark.skipif(
    os.environ.get("HIMEM_RUN_GPU_TESTS") != "1" or os.environ.get("HIMEM_GPU_LOCK_HELD") != "1",
    reason="GPU tests require HIMEM_RUN_GPU_TESTS=1 and HIMEM_GPU_LOCK_HELD=1"
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.mark.skipif(
    os.environ.get("HIMEM_RUN_GPU_TESTS") != "1" or os.environ.get("HIMEM_GPU_LOCK_HELD") != "1",
    reason="GPU tests require HIMEM_RUN_GPU_TESTS=1 and HIMEM_GPU_LOCK_HELD=1"
)
def test_pytorch_gpu(device):
    """Test basic PyTorch GPU functionality."""
    x = torch.randn(10, 10, device=device)
    y = x @ x.T
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


def test_himem_import():
    """Test that HiMem modules can be imported."""
    from lerobot.memory.himem import (
        HiMemConfig,
        TokenSchema,
        EventFeatureEncoder,
        create_himem_encoder,
    )
    assert HiMemConfig is not None
    assert TokenSchema is not None


def test_feature_encoder_cpu():
    """Test EventFeatureEncoder on CPU."""
    from lerobot.memory.himem import TokenSchema, EventFeatureEncoder

    schema = TokenSchema(
        visual_feature_dim=32,
        visual_delta_dim=32,
        event_observable_dim=16,
    )
    encoder = EventFeatureEncoder(schema)

    # First token (no previous)
    visual = torch.randn(32)
    state = torch.randn(8)
    event_obs = torch.randn(16)

    token = encoder.encode_token(
        visual_feature=visual,
        state=state,
        prev_action_summary=None,
        prev_visual_feature=None,
        prev_state=None,
        event_observable=event_obs,
        episode_id="test",
        env_id="test_env",
        decision_index=0,
        decision_start_frame=0,
        prev_executed_start=None,
        prev_executed_end=None,
        prev_executed_count=None,
        end_reason=None,  # Fixed: decision 0 must have null end_reason
        schema_hash=schema.compute_hash(),
    )

    assert token is not None
    assert token.model_input.shape[0] == schema.token_dim


def test_encoder_init_cpu():
    """Test HiMemEncoder initialization on CPU."""
    from lerobot.memory.himem import HiMemConfig, create_himem_encoder

    config = HiMemConfig(
        window_size=4,
        memory_dim=64,
        backend="gru_reference",
        gru_hidden_dim=64,
    )

    encoder = create_himem_encoder(config)
    assert encoder is not None


@pytest.mark.skipif(
    os.environ.get("HIMEM_RUN_GPU_TESTS") != "1" or os.environ.get("HIMEM_GPU_LOCK_HELD") != "1",
    reason="GPU tests require HIMEM_RUN_GPU_TESTS=1 and HIMEM_GPU_LOCK_HELD=1"
)
def test_encoder_init_gpu(device):
    """Test HiMemEncoder initialization on GPU."""
    from lerobot.memory.himem import HiMemConfig, create_himem_encoder

    config = HiMemConfig(
        window_size=4,
        memory_dim=64,
        backend="gru_reference",
        gru_hidden_dim=64,
    )

    encoder = create_himem_encoder(config).to(device)
    assert next(encoder.parameters()).device.type == "cuda"
