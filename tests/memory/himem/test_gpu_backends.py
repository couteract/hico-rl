"""
GPU smoke tests for B01R HiMem backends.

Tests all three backends (gru_reference, mamba2, simplified_ssm_experimental)
for basic forward/backward pass and numerical stability on GPU.

Run with: pytest tests/memory/himem/test_gpu_backends.py
B01R5P2: Uses shared lock via workstream_gpu.sh wrapper.
"""

import pytest
import torch
import os

# B01R5P2: Module-level GPU opt-in check - must be EXACTLY "1"
pytestmark = pytest.mark.skipif(
    os.environ.get("HIMEM_RUN_GPU_TESTS") != "1" or os.environ.get("HIMEM_GPU_LOCK_HELD") != "1",
    reason="GPU tests require HIMEM_RUN_GPU_TESTS=1 and HIMEM_GPU_LOCK_HELD=1"
)

from lerobot.memory.himem import (
    HiMemConfig,
    TokenSchema,
    create_himem_encoder,
    GRUMemoryBackend,
    Mamba2Backend,
    SimplifiedSSMExperimental,
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


def create_synthetic_tokens(num_tokens: int, config: HiMemConfig, device: torch.device):
    """Create synthetic tokens for testing. Returns at most W tokens."""
    from lerobot.memory.himem import CausalToken, EventFeatureEncoder

    feature_enc = EventFeatureEncoder(config.token_schema)

    # Respect window size limit
    actual_num = min(num_tokens, config.window_size)
    tokens = []

    for i in range(actual_num):
        torch.manual_seed(i + 5000)

        visual_feature = torch.randn(config.token_schema.visual_feature_dim, device=device) * 0.1
        state = torch.randn(config.token_schema.state_dim, device=device) * 0.1
        event_observable = torch.randn(config.token_schema.event_observable_dim, device=device) * 0.1

        if i > 0:
            torch.manual_seed(i - 1 + 5000)
            prev_visual = torch.randn(config.token_schema.visual_feature_dim, device=device) * 0.1
            prev_state = torch.randn(config.token_schema.state_dim, device=device) * 0.1

            torch.manual_seed(i + 6000)
            from lerobot.memory.himem import summarize_action_chunk
            actions = torch.randn(10, 7, device=device) * 0.1
            prev_action_summary = summarize_action_chunk(actions)

            prev_executed_start = (i - 1) * 10
            prev_executed_end = i * 10
            prev_executed_count = 10
            end_reason = "timeout"
        else:
            prev_visual = None
            prev_state = None
            prev_action_summary = None
            prev_executed_start = None
            prev_executed_end = None
            prev_executed_count = None
            end_reason = None

        token = feature_enc.encode_token(
            visual_feature=visual_feature,
            state=state,
            prev_action_summary=prev_action_summary,
            prev_visual_feature=prev_visual,
            prev_state=prev_state,
            event_observable=event_observable,
            episode_id="gpu_test",
            env_id="test_env",
            decision_index=i,
            decision_start_frame=i * 10,
            prev_executed_start=prev_executed_start,
            prev_executed_end=prev_executed_end,
            prev_executed_count=prev_executed_count,
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

        tokens = create_synthetic_tokens(8, config, device)

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

        tokens = create_synthetic_tokens(8, config, device)

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

    def test_determinism(self, small_config, device):
        """Multiple forward passes give same result."""
        config = small_config
        config.backend = "gru_reference"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_synthetic_tokens(8, config, device)

        with torch.no_grad():
            output1 = encoder.forward_from_tokens(tokens, return_all=False)
            output2 = encoder.forward_from_tokens(tokens, return_all=False)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)


class TestMamba2Backend:
    """Test official Mamba2 backend on GPU."""

    def test_backend_is_official_mamba2(self, small_config, device):
        """Verify we're using official mamba_ssm.Mamba2."""
        config = small_config
        config.backend = "mamba2"

        encoder = create_himem_encoder(config=config).to(device)

        manifest = encoder.get_backend_manifest()
        assert manifest["backend_type"] == "mamba2"
        assert manifest["implementation"] == "mamba_ssm.Mamba2"
        assert "mamba_ssm_version" in manifest
        assert manifest["use_mem_eff_path"] is False  # causal-conv1d not installed

    def test_forward_pass(self, small_config, device):
        """Forward pass completes without error."""
        config = small_config
        config.backend = "mamba2"
        config.mamba2_use_mem_eff_path = False  # Required

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_synthetic_tokens(8, config, device)

        with torch.no_grad():
            output = encoder.forward_from_tokens(tokens, return_all=False)

        assert output.shape == (config.memory_dim,)
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()

    def test_backward_pass(self, small_config, device):
        """Backward pass computes gradients."""
        config = small_config
        config.backend = "mamba2"
        config.mamba2_use_mem_eff_path = False

        encoder = create_himem_encoder(config=config).to(device)
        encoder.train()

        tokens = create_synthetic_tokens(8, config, device)

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        # Check backend has gradients
        backend_has_grad = False
        for param in encoder.backend.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                backend_has_grad = True
                break

        assert backend_has_grad, "Mamba2 backend has no gradients"

    def test_determinism(self, small_config, device):
        """Multiple forward passes give same result in eval mode."""
        config = small_config
        config.backend = "mamba2"
        config.mamba2_use_mem_eff_path = False

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_synthetic_tokens(8, config, device)

        with torch.no_grad():
            output1 = encoder.forward_from_tokens(tokens, return_all=False)
            output2 = encoder.forward_from_tokens(tokens, return_all=False)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-6)


class TestSimplifiedSSMBackend:
    """Test experimental SimplifiedSSM backend on GPU."""

    def test_forward_pass(self, small_config, device):
        """Forward pass completes without error."""
        config = small_config
        config.backend = "simplified_ssm_experimental"

        encoder = create_himem_encoder(config=config).to(device)
        encoder.eval()

        tokens = create_synthetic_tokens(8, config, device)

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

        tokens = create_synthetic_tokens(8, config, device)

        output = encoder.forward_from_tokens(tokens, return_all=False)
        loss = output.sum()
        loss.backward()

        has_grad = False
        for param in encoder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"

    def test_marked_as_experimental(self, small_config, device):
        """Backend is marked as experimental."""
        config = small_config
        config.backend = "simplified_ssm_experimental"

        encoder = create_himem_encoder(config=config).to(device)

        manifest = encoder.get_backend_manifest()
        assert manifest["backend_type"] == "simplified_ssm_experimental"
        assert "NOT RECOMMENDED" in manifest.get("warning", "")


class TestBackendComparison:
    """Cross-backend tests."""

    def test_all_backends_produce_finite_output(self, small_config, device):
        """All backends produce finite outputs."""
        tokens = create_synthetic_tokens(8, small_config, device)

        for backend_name in ["gru_reference", "mamba2", "simplified_ssm_experimental"]:
            config = small_config
            config.backend = backend_name
            if backend_name == "mamba2":
                config.mamba2_use_mem_eff_path = False

            encoder = create_himem_encoder(config=config).to(device)
            encoder.eval()

            with torch.no_grad():
                output = encoder.forward_from_tokens(tokens, return_all=False)

            assert torch.isfinite(output).all(), f"Backend {backend_name} produced non-finite output"

    def test_backends_differ(self, small_config, device):
        """Different backends produce different outputs (sanity check)."""
        tokens = create_synthetic_tokens(8, small_config, device)

        # Get outputs from two backends
        config_gru = small_config
        config_gru.backend = "gru_reference"
        encoder_gru = create_himem_encoder(config=config_gru).to(device)
        encoder_gru.eval()

        config_mamba = small_config
        config_mamba.backend = "mamba2"
        config_mamba.mamba2_use_mem_eff_path = False
        encoder_mamba = create_himem_encoder(config=config_mamba).to(device)
        encoder_mamba.eval()

        with torch.no_grad():
            output_gru = encoder_gru.forward_from_tokens(tokens, return_all=False)
            output_mamba = encoder_mamba.forward_from_tokens(tokens, return_all=False)

        # Should be different (different architectures)
        diff = (output_gru - output_mamba).abs().max().item()
        assert diff > 1e-3, f"Backends produced too similar outputs: diff = {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
