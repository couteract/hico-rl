"""
Verify that backend=mamba2 uses official mamba_ssm.Mamba2.

This test confirms B01R requirement: "backend=mamba2 必须实际调用官方 mamba_ssm.Mamba2"
"""

import pytest
import torch
import os

# B01R5P2: Module-level GPU opt-in check - must be EXACTLY "1"
pytestmark = pytest.mark.skipif(
    os.environ.get("HIMEM_RUN_GPU_TESTS") != "1" or os.environ.get("HIMEM_GPU_LOCK_HELD") != "1",
    reason="GPU tests require HIMEM_RUN_GPU_TESTS=1 and HIMEM_GPU_LOCK_HELD=1"
)

from lerobot.memory.himem import HiMemConfig, TokenSchema, create_himem_encoder


def test_mamba2_uses_official_implementation():
    """Verify backend=mamba2 uses official mamba_ssm.Mamba2."""
    config = HiMemConfig(
        backend="mamba2",
        memory_dim=64,
        mamba2_d_model=64,
        mamba2_d_state=32,
        mamba2_num_layers=1,
        mamba2_use_mem_eff_path=False,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    # Access the backend
    backend = encoder.backend

    # Check backend type
    assert backend.backend_type == "mamba2", f"Expected backend_type='mamba2', got '{backend.backend_type}'"

    # Check that it imported from mamba_ssm
    assert hasattr(backend, "mamba_ssm_version"), "Backend should have mamba_ssm_version"
    assert hasattr(backend, "mamba_ssm_path"), "Backend should have mamba_ssm_path"

    print(f"✓ Using mamba_ssm version: {backend.mamba_ssm_version}")
    print(f"✓ Imported from: {backend.mamba_ssm_path}")

    # Check that layers are official Mamba2
    from mamba_ssm import Mamba2 as OfficialMamba2

    for i, layer in enumerate(backend.layers):
        assert isinstance(layer, OfficialMamba2), (
            f"Layer {i} is not official Mamba2: {type(layer)}"
        )

    print(f"✓ All {len(backend.layers)} layers are official mamba_ssm.Mamba2")

    # Verify parameters exist
    total_params = sum(p.numel() for p in backend.parameters())
    assert total_params > 0, "Backend should have trainable parameters"

    print(f"✓ Backend has {total_params:,} parameters")


def test_mamba2_forward_uses_official_code():
    """Verify forward pass actually executes official Mamba2 code."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = HiMemConfig(
        backend="mamba2",
        memory_dim=64,
        mamba2_d_model=64,
        mamba2_d_state=32,
        mamba2_num_layers=1,
        mamba2_use_mem_eff_path=False,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config).to("cuda:0")
    encoder.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 4
    token_dim = config.token_schema.token_dim

    x = torch.randn(batch_size, seq_len, token_dim, device="cuda:0")

    # Forward pass
    with torch.no_grad():
        output = encoder(x)

    assert output.shape == (batch_size, config.memory_dim)
    assert output.device.type == "cuda"
    assert torch.isfinite(output).all()

    print(f"✓ Forward pass completed successfully")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")


def test_mamba2_rolling_window_lengths():
    """Test Mamba2 with rolling window lengths: 1, 7, 8 (B01R2 requirement)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = HiMemConfig(
        backend="mamba2",
        memory_dim=128,
        mamba2_d_model=128,
        mamba2_d_state=64,
        mamba2_num_layers=2,
        mamba2_use_mem_eff_path=False,
    )

    encoder = create_himem_encoder(config=config).to("cuda:0")
    encoder.eval()

    token_dim = config.token_schema.token_dim

    # Test critical window lengths
    for seq_len in [1, 7, 8]:
        print(f"\n  Testing seq_len={seq_len}")
        x = torch.randn(2, seq_len, token_dim, device="cuda:0")

        with torch.no_grad():
            output = encoder(x, return_all=False)

        assert output.shape == (2, config.memory_dim)
        assert torch.isfinite(output).all()
        print(f"  ✓ seq_len={seq_len} passed")


def test_mamba2_2w_plus_3_positions():
    """Test 2W+3 full position coverage (W=8, so 2*8+3=19 positions) (B01R2 requirement)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = HiMemConfig(
        backend="mamba2",
        memory_dim=128,
        mamba2_d_model=128,
        mamba2_num_layers=2,
        window_size=8,
    )

    encoder = create_himem_encoder(config=config).to("cuda:0")
    encoder.eval()

    token_dim = config.token_schema.token_dim

    # Test 19 positions (2*8 + 3)
    seq_len = 19
    print(f"\n  Testing seq_len={seq_len} (2W+3 coverage)")

    x = torch.randn(1, seq_len, token_dim, device="cuda:0")

    with torch.no_grad():
        output = encoder(x, return_all=True)

    assert output.shape == (1, seq_len, config.memory_dim)
    assert torch.isfinite(output).all()
    print(f"  ✓ 2W+3 positions all valid")


def test_mamba2_with_optimizer():
    """Test Mamba2 can be optimized (B01R2 requirement)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = HiMemConfig(
        backend="mamba2",
        memory_dim=128,
        mamba2_d_model=128,
        mamba2_num_layers=2,
    )

    encoder = create_himem_encoder(config=config).to("cuda:0")
    encoder.train()

    # Create optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    token_dim = config.token_schema.token_dim
    x = torch.randn(2, 4, token_dim, device="cuda:0")
    target = torch.randn(2, config.memory_dim, device="cuda:0")

    # Forward + backward
    output = encoder(x, return_all=False)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    print(f"  ✓ Optimizer step completed")
    print(f"  ✓ Loss: {loss.item():.6f}")


def test_mamba2_checkpoint_save_load():
    """Test Mamba2 checkpoint save/load (B01R2 requirement)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import tempfile
    import os

    config = HiMemConfig(
        backend="mamba2",
        memory_dim=128,
        mamba2_d_model=128,
        mamba2_num_layers=2,
    )

    # Create encoder
    encoder1 = create_himem_encoder(config=config).to("cuda:0")
    encoder1.eval()

    # Generate test input
    token_dim = config.token_schema.token_dim
    x = torch.randn(1, 4, token_dim, device="cuda:0")

    # Forward pass
    with torch.no_grad():
        output1 = encoder1(x, return_all=False)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        checkpoint_path = f.name

    torch.save({
        "config": config.to_dict(),
        "model_state_dict": encoder1.state_dict(),
    }, checkpoint_path)

    print(f"  ✓ Checkpoint saved to {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    encoder2 = create_himem_encoder(config=config).to("cuda:0")
    encoder2.load_state_dict(checkpoint["model_state_dict"])
    encoder2.eval()

    # Forward pass with loaded model
    with torch.no_grad():
        output2 = encoder2(x, return_all=False)

    # Verify outputs match
    assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)
    print(f"  ✓ Loaded model produces identical output")

    # Cleanup
    os.unlink(checkpoint_path)


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("=" * 60)
    print("Testing official Mamba2 usage")
    print("=" * 60)

    test_mamba2_uses_official_implementation()

    print("\n" + "=" * 60)
    print("Testing Mamba2 forward pass")
    print("=" * 60)

    test_mamba2_forward_uses_official_code()

    print("\n" + "=" * 60)
    print("Testing rolling window lengths (1, 7, 8)")
    print("=" * 60)

    test_mamba2_rolling_window_lengths()

    print("\n" + "=" * 60)
    print("Testing 2W+3 positions")
    print("=" * 60)

    test_mamba2_2w_plus_3_positions()

    print("\n" + "=" * 60)
    print("Testing with optimizer")
    print("=" * 60)

    test_mamba2_with_optimizer()

    print("\n" + "=" * 60)
    print("Testing checkpoint save/load")
    print("=" * 60)

    test_mamba2_checkpoint_save_load()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
