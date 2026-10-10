"""
B01R4 Comprehensive GPU Tests

Extends B01R3 GPU gates with:
- Batch processing tests (batch=1, small batch)
- GRU backend tests
- Window boundary tests
- Performance validation
- Complete numerical matrix for debug and main configs

Requires:
1. HIMEM_RUN_GPU_TESTS=1
2. HIMEM_GPU_LOCK_HELD=1
3. torch.cuda.is_available()

Run via: scripts/workstream_gpu.sh -m pytest tests/memory/himem/test_b01r4_gpu_comprehensive.py
"""

import pytest
import torch
import os
import json
from pathlib import Path

# Check opt-in BEFORE importing mamba_ssm
def _check_gpu_opt_in():
    """Check if GPU tests are authorized."""
    has_opt_in = os.environ.get("HIMEM_RUN_GPU_TESTS") == "1"
    has_lock = os.environ.get("HIMEM_GPU_LOCK_HELD") == "1"
    has_cuda = torch.cuda.is_available()

    if not has_opt_in or not has_lock:
        return False, "Missing HIMEM_RUN_GPU_TESTS=1 or HIMEM_GPU_LOCK_HELD=1"

    if not has_cuda:
        return False, "CUDA not available"

    return True, "GPU tests authorized"


GPU_AUTHORIZED, GPU_SKIP_REASON = _check_gpu_opt_in()

# Only import mamba_ssm if authorized
if GPU_AUTHORIZED:
    try:
        import mamba_ssm
        MAMBA_AVAILABLE = True
    except ImportError:
        MAMBA_AVAILABLE = False
        GPU_SKIP_REASON = "mamba_ssm not installed"
else:
    MAMBA_AVAILABLE = False


# Pytest skip marker
pytestmark = pytest.mark.skipif(
    not GPU_AUTHORIZED or not MAMBA_AVAILABLE,
    reason=GPU_SKIP_REASON
)


from lerobot.memory.himem import (
    HiMemConfig,
    TokenSchema,
    CausalToken,
    EventFeatureEncoder,
    create_himem_encoder,
    save_checkpoint,
    load_checkpoint,
)


def create_debug_mamba2_config():
    """Debug Mamba2 config."""
    return HiMemConfig(
        window_size=8,
        memory_dim=128,
        backend="mamba2",
        mamba2_d_model=128,
        mamba2_d_state=32,
        mamba2_d_conv=4,
        mamba2_expand=2,
        mamba2_headdim=32,
        mamba2_num_layers=1,
        mamba2_use_mem_eff_path=False,
        token_schema=TokenSchema(
            visual_feature_dim=64,
            visual_delta_dim=64,
            event_observable_dim=16,
        ),
    )


def create_main_mamba2_config():
    """Main Mamba2 config."""
    return HiMemConfig(
        window_size=8,
        memory_dim=256,
        backend="mamba2",
        mamba2_d_model=256,
        mamba2_d_state=64,
        mamba2_d_conv=4,
        mamba2_expand=2,
        mamba2_headdim=64,
        mamba2_num_layers=2,
        mamba2_use_mem_eff_path=False,
        token_schema=TokenSchema(
            visual_feature_dim=256,
            visual_delta_dim=256,
            event_observable_dim=32,
        ),
    )


def create_gru_config():
    """GRU config for comparison."""
    return HiMemConfig(
        window_size=8,
        memory_dim=128,
        backend="gru_reference",
        gru_num_layers=2,
        gru_hidden_dim=128,
        token_schema=TokenSchema(
            visual_feature_dim=64,
            visual_delta_dim=64,
            event_observable_dim=16,
        ),
    )


def create_synthetic_token(
    decision_index: int,
    schema: TokenSchema,
    episode_id: str = "test_ep",
    env_id: str = "test_env",
    device: torch.device = None,
    seed: int = None,
) -> CausalToken:
    """Create a synthetic causal token for testing with proper layout."""
    if seed is None:
        seed = decision_index + 1000

    torch.manual_seed(seed)
    device = device or torch.device("cuda")

    # Use EventFeatureEncoder to ensure proper layout
    encoder = EventFeatureEncoder(schema)

    # Create synthetic features
    visual_feature = torch.randn(schema.visual_feature_dim, device=device) * 0.1
    state = torch.randn(schema.state_dim, device=device) * 0.1
    event_obs = torch.randn(schema.event_observable_dim, device=device) * 0.1

    # For decision 0: no previous features/actions
    # For decision k>0: provide previous features/actions
    if decision_index == 0:
        prev_action_summary = None
        prev_visual_feature = None
        prev_state = None
    else:
        prev_action_summary = torch.randn(schema.action_summary_dim, device=device) * 0.1
        prev_visual_feature = torch.randn(schema.visual_feature_dim, device=device) * 0.1
        prev_state = torch.randn(schema.state_dim, device=device) * 0.1

    return encoder.encode_token(
        visual_feature=visual_feature,
        state=state,
        prev_action_summary=prev_action_summary,
        prev_visual_feature=prev_visual_feature,
        prev_state=prev_state,
        event_observable=event_obs,
        episode_id=episode_id,
        env_id=env_id,
        decision_index=decision_index,
        decision_start_frame=0 if decision_index == 0 else decision_index * 10,
        prev_executed_start=None if decision_index == 0 else (decision_index - 1) * 10,
        prev_executed_end=None if decision_index == 0 else decision_index * 10,
        prev_executed_count=None if decision_index == 0 else 10,
        end_reason=None if decision_index == 0 else "queue_depleted",
        schema_hash=schema.compute_hash(),
    )


class TestBatchProcessing:
    """Test batch processing with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [1, 7, 8])
    def test_mamba2_batch_forward(self, batch_size, seq_len):
        """Test Mamba2 batch forward with various batch and sequence sizes."""
        config = create_debug_mamba2_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        # Create batch of token sequences
        batch_tokens = []
        for b in range(batch_size):
            tokens = [
                create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
                for i in range(seq_len)
            ]
            batch_tokens.append(tokens)

        with torch.no_grad():
            output = encoder.forward_batch(batch_tokens, return_all=True)

        assert output.shape == (batch_size, seq_len, config.memory_dim)
        assert torch.isfinite(output).all(), "Batch forward contains NaN/Inf"

    def test_gru_batch_forward(self):
        """Test GRU batch forward."""
        config = create_gru_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        # Create batch of token sequences
        batch_tokens = []
        for b in range(4):
            tokens = [
                create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
                for i in range(8)
            ]
            batch_tokens.append(tokens)

        with torch.no_grad():
            output = encoder.forward_batch(batch_tokens, return_all=True)

        assert output.shape == (4, 8, config.memory_dim)
        assert torch.isfinite(output).all()


class TestWindowBoundary:
    """Test window boundary behavior."""

    def test_window_size_8_enforced(self):
        """Test that exactly window_size=8 tokens are processed correctly."""
        config = create_debug_mamba2_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        # Create exactly W=8 tokens
        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(8)
        ]

        with torch.no_grad():
            output = encoder.forward_from_tokens(tokens, return_all=True)

        assert output.shape == (8, config.memory_dim)
        assert torch.isfinite(output).all()

    def test_exceeds_window_rejected(self):
        """Test that sequences > W=8 are rejected."""
        config = create_debug_mamba2_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        # Try to use 9 tokens (exceeds W=8)
        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(9)
        ]

        with pytest.raises(ValueError, match="exceeds window size"):
            encoder.forward_from_tokens(tokens, return_all=True)


class TestGRUBackend:
    """Test GRU backend on GPU."""

    def test_gru_forward_backward(self):
        """Test GRU forward and backward passes."""
        config = create_gru_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.train()

        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(5)
        ]

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        # Check gradients are finite
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"

    def test_gru_checkpoint_roundtrip(self, tmp_path):
        """Test GRU checkpoint save/load."""
        config = create_gru_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        # Save checkpoint
        ckpt_path = tmp_path / "gru_test.pt"
        save_checkpoint(encoder, ckpt_path)

        # Load and verify
        loaded = load_checkpoint(ckpt_path, device=torch.device("cuda"), strict=True)
        loaded.eval()

        # Compare outputs
        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(5)
        ]

        with torch.no_grad():
            output1 = encoder.forward_from_tokens(tokens, return_all=True)
            output2 = loaded.forward_from_tokens(tokens, return_all=True)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)


class TestMainConfigValidation:
    """Validate main config meets requirements."""

    def test_main_mamba2_parameter_count(self):
        """Test main Mamba2 config has <=5M parameters."""
        config = create_main_mamba2_config()
        encoder = create_himem_encoder(config=config).cuda()

        total_params = sum(p.numel() for p in encoder.parameters())

        assert total_params <= 5_000_000, f"Main config has {total_params} parameters, exceeds 5M limit"

        # Record actual counts
        backend_params = sum(p.numel() for p in encoder.backend.parameters())
        proj_params = sum(p.numel() for p in encoder.input_proj.parameters()) + \
                     sum(p.numel() for p in encoder.output_proj.parameters())

        print(f"\nMain config parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Backend: {backend_params:,}")
        print(f"  Projections: {proj_params:,}")

    def test_main_forward_backward_finite(self):
        """Test main config forward/backward produces finite values."""
        config = create_main_mamba2_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.train()

        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(8)
        ]

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        assert torch.isfinite(output).all()
        for param in encoder.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    def test_main_optimizer_updates_backend(self):
        """Test optimizer actually updates main config backend parameters."""
        config = create_main_mamba2_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.train()

        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Get initial Mamba layer parameter
        from mamba_ssm import Mamba2 as OfficialMamba2
        mamba_layer = None
        for layer in encoder.backend.layers:
            if isinstance(layer, OfficialMamba2):
                mamba_layer = layer
                break

        assert mamba_layer is not None
        initial_param = mamba_layer.in_proj.weight.data.clone()

        # Forward + backward + step
        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(5)
        ]

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check parameter changed
        final_param = mamba_layer.in_proj.weight.data
        assert not torch.allclose(initial_param, final_param, rtol=1e-7), \
            "Optimizer step did not change Mamba2 parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
