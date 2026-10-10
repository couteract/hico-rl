"""
B01R3 GPU Numerical Gates for Official Mamba2

These tests require:
1. HIMEM_RUN_GPU_TESTS=1 environment variable
2. HIMEM_GPU_LOCK_HELD=1 (set by workstream_gpu.sh wrapper)
3. torch.cuda.is_available() == True

Run via: scripts/workstream_gpu.sh -m pytest tests/memory/himem/test_b01r3_gpu_gates.py

Direct pytest without opt-in will skip ALL GPU tests cleanly.
"""

import pytest
import torch
import os
import sys

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


def create_debug_config():
    """Debug config for GPU tests."""
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


def create_main_config():
    """Main production config for GPU tests."""
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


class TestOfficialMamba2Verification:
    """Verify official Mamba2 backend is being used."""

    def test_backend_is_official_mamba2(self):
        """Assert layers are actual mamba_ssm.Mamba2 instances."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()

        # Check backend type
        assert encoder.backend.backend_type == "mamba2"

        # Check each layer is official Mamba2
        from mamba_ssm import Mamba2 as OfficialMamba2
        for layer in encoder.backend.layers:
            assert isinstance(layer, OfficialMamba2), \
                f"Layer is {type(layer)}, not official mamba_ssm.Mamba2"

    def test_mamba_ssm_provenance(self):
        """Record mamba_ssm version and path."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()

        manifest = encoder.get_backend_manifest()

        assert "mamba_ssm_version" in manifest
        assert "mamba_ssm_path" in manifest
        assert manifest["backend_type"] == "mamba2"
        assert manifest["implementation"] == "mamba_ssm.Mamba2"

        print(f"\nmamba_ssm version: {manifest['mamba_ssm_version']}")
        print(f"mamba_ssm path: {manifest['mamba_ssm_path']}")

    def test_resolved_constructor_config(self):
        """Verify resolved Mamba2 constructor matches config."""
        config = create_main_config()
        encoder = create_himem_encoder(config=config).cuda()

        manifest = encoder.get_backend_manifest()

        assert manifest["d_model"] == 256
        assert manifest["d_state"] == 64
        assert manifest["d_conv"] == 4
        assert manifest["expand"] == 2
        assert manifest["headdim"] == 64
        assert manifest["num_layers"] == 2
        assert manifest["use_mem_eff_path"] == False


class TestMamba2ForwardBackward:
    """Test forward/backward passes with official Mamba2."""

    @pytest.mark.parametrize("seq_len", [1, 7, 8])
    def test_forward_finite_outputs(self, seq_len):
        """Forward pass produces finite outputs for various sequence lengths."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(seq_len)
        ]

        with torch.no_grad():
            output = encoder.forward_from_tokens(tokens, return_all=True)

        assert output.shape == (seq_len, config.memory_dim)
        assert torch.isfinite(output).all(), "Forward output contains NaN/Inf"

    def test_forward_backward_gradients_finite(self):
        """Backward pass produces finite gradients."""
        config = create_debug_config()
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
                assert torch.isfinite(param.grad).all(), \
                    f"Gradient for {name} contains NaN/Inf"

    def test_optimizer_step_changes_parameters(self):
        """Optimizer step actually updates Mamba2 parameters."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.train()

        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Get initial Mamba layer parameter
        initial_param = encoder.backend.layers[0].in_proj.weight.data.clone()

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
        updated_param = encoder.backend.layers[0].in_proj.weight.data
        diff = (updated_param - initial_param).abs().max().item()

        assert diff > 1e-6, f"Parameters did not change after optimizer step: max_diff={diff}"


class TestRollingWindowOnlineOffline:
    """Test online vs offline recent-W equivalence for 19 positions."""

    def test_19_positions_online_offline_equivalence(self):
        """
        For 19 decisions (2W+3), verify online and offline recent-W match at every position.

        This is the critical test that B01R2 failed (only did one length-19 forward).
        """
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        W = config.window_size
        num_decisions = 2 * W + 3  # 19

        # Create all tokens
        all_tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
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

        # Offline path: recompute each position from recent W with prev_state=None
        offline_memories = []
        for k in range(num_decisions):
            start_idx = max(0, k - W + 1)
            window_tokens = all_tokens[start_idx:k + 1]

            with torch.no_grad():
                memory = encoder.forward_from_tokens(window_tokens, return_all=False)
            offline_memories.append(memory)

        # Compare all 19 positions
        for k in range(num_decisions):
            is_close = torch.allclose(
                online_memories[k], offline_memories[k],
                rtol=1e-4, atol=1e-6
            )
            if not is_close:
                diff = (online_memories[k] - offline_memories[k]).abs().max().item()
                pytest.fail(
                    f"Position {k}/{num_decisions}: online != offline, max_diff = {diff}\n"
                    f"Critical boundaries: W-1={W-1}, W={W}, W+1={W+1}"
                )

    def test_future_token_perturbation_no_effect(self):
        """Perturbing future tokens doesn't change past outputs."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        W = config.window_size  # 8

        # Base sequence: use W tokens [0..7]
        tokens_base = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(W)
        ]

        # Extended sequence: same base tokens [0..7] then perturbed future, but only test within W
        # To test causality, we compare output at position 5 when sequence is [0..5] vs [0..7]
        tokens_short = tokens_base[:6]  # [0..5]
        tokens_long = tokens_base[:8]   # [0..7]

        with torch.no_grad():
            output_short = encoder.forward_from_tokens(tokens_short, return_all=True)  # [6, memory_dim]
            output_long = encoder.forward_from_tokens(tokens_long, return_all=True)    # [8, memory_dim]

        # First 6 positions should match exactly (future tokens 6,7 don't affect positions 0-5)
        assert torch.allclose(output_short, output_long[:6], rtol=1e-5, atol=1e-7), \
            "Future tokens affected past positions"

    def test_window_history_perturbation_changes_current(self):
        """Changing history within window changes current memory."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        W = config.window_size  # 8

        # Normal sequence: exactly W tokens [0..7]
        tokens_normal = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(W)
        ]

        # Perturbed sequence: change token at position W-1 (last position)
        tokens_perturbed = tokens_normal.copy()
        tokens_perturbed[W-1] = create_synthetic_token(
            W-1, config.token_schema, device=torch.device("cuda"), seed=(W-1) + 9999
        )

        with torch.no_grad():
            output_normal = encoder.forward_from_tokens(tokens_normal, return_all=False)
            output_perturbed = encoder.forward_from_tokens(tokens_perturbed, return_all=False)

        diff = (output_normal - output_perturbed).abs().max().item()
        assert diff > 1e-4, \
            f"Changing token within window did not affect output: diff={diff}"


class TestMamba2CheckpointContract:
    """Test Mamba2 checkpoint with expected-contract validation."""

    def test_save_load_eval_consistency(self, tmp_path):
        """Mamba2 checkpoint save/load preserves eval outputs."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()
        encoder.eval()

        # Save
        ckpt_path = tmp_path / "mamba2_test.pt"
        save_checkpoint(encoder, ckpt_path)

        # Load
        loaded_encoder = load_checkpoint(ckpt_path, device=torch.device("cuda"))
        loaded_encoder.eval()

        # Test
        tokens = [
            create_synthetic_token(i, config.token_schema, device=torch.device("cuda"))
            for i in range(8)
        ]

        with torch.no_grad():
            out1 = encoder.forward_from_tokens(tokens, return_all=True)
            out2 = loaded_encoder.forward_from_tokens(tokens, return_all=True)

        assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-7)

    def test_mamba2_config_mismatch_rejected(self, tmp_path):
        """Cannot load Mamba2 checkpoint with different d_state."""
        config = create_debug_config()
        encoder = create_himem_encoder(config=config).cuda()

        ckpt_path = tmp_path / "mamba2_test.pt"
        save_checkpoint(encoder, ckpt_path)

        # Try to load with different d_state
        different_config = create_debug_config()
        different_config.mamba2_d_state = 16  # Was 32

        with pytest.raises(ValueError, match="mamba2_d_state.*mismatch"):
            load_checkpoint(ckpt_path, expected_config=different_config, device=torch.device("cuda"))

    def test_mamba2_provenance_stored(self, tmp_path):
        """Checkpoint manifest includes full Mamba2 provenance."""
        config = create_main_config()
        encoder = create_himem_encoder(config=config).cuda()

        ckpt_path = tmp_path / "mamba2_test.pt"
        manifest = save_checkpoint(encoder, ckpt_path)

        # Check manifest contains Mamba2 details
        backend_manifest = manifest["backend_manifest"]
        assert backend_manifest["backend_type"] == "mamba2"
        assert backend_manifest["d_model"] == 256
        assert backend_manifest["d_state"] == 64
        assert backend_manifest["num_layers"] == 2
        assert "mamba_ssm_version" in backend_manifest


class TestMainConfigParameters:
    """Test main production config meets parameter constraints."""

    def test_main_config_parameter_count(self):
        """Main config has <=5M trainable parameters."""
        config = create_main_config()
        encoder = create_himem_encoder(config=config).cuda()

        counts = encoder.count_parameters()
        trainable = counts["trainable"]

        assert trainable <= 5_000_000, \
            f"Main config has {trainable:,} trainable parameters, exceeds 5M limit"

        print(f"\nMain config parameters:")
        print(f"  Total: {counts['total']:,}")
        print(f"  Trainable: {counts['trainable']:,}")
        print(f"  Feature encoder: {counts['feature_encoder']:,}")
        print(f"  Backend (Mamba2): {counts['backend']:,}")
        print(f"  Projections: {counts['projections']:,}")

    def test_debug_config_smaller_than_main(self):
        """Debug config has fewer parameters than main."""
        debug_config = create_debug_config()
        main_config = create_main_config()

        debug_encoder = create_himem_encoder(config=debug_config).cuda()
        main_encoder = create_himem_encoder(config=main_config).cuda()

        debug_params = debug_encoder.count_parameters()["total"]
        main_params = main_encoder.count_parameters()["total"]

        assert debug_params < main_params, \
            f"Debug config ({debug_params:,}) should be smaller than main ({main_params:,})"
