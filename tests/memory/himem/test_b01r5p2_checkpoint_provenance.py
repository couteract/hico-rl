"""
B01R5P2: Checkpoint Provenance Tampering Tests

Tests that strict=True validates backend manifest integrity and rejects tampered checkpoints.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from lerobot.memory.himem import HiMemConfig, TokenSchema, create_himem_encoder
from lerobot.memory.himem.checkpoint import save_checkpoint, load_checkpoint


def test_gru_tampered_torch_version_rejected():
    """GRU checkpoint with tampered torch_version must be rejected in strict mode."""
    config = HiMemConfig(
        backend="gru_reference",
        memory_dim=64,
        gru_hidden_dim=64,
        gru_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Load and tamper
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        original_version = checkpoint["backend_manifest"]["torch_version"]
        checkpoint["backend_manifest"]["torch_version"] = "9.9.9.fake"

        # Save tampered checkpoint
        tampered_path = Path(tmpdir) / "tampered.pt"
        torch.save(checkpoint, tampered_path)

        # Attempt strict load - should reject
        with pytest.raises(ValueError, match="Backend provenance mismatch for 'torch_version'"):
            load_checkpoint(tampered_path, strict=True)


def test_gru_tampered_hidden_dim_rejected():
    """GRU checkpoint with tampered hidden_dim must be rejected in strict mode."""
    config = HiMemConfig(
        backend="gru_reference",
        memory_dim=64,
        gru_hidden_dim=64,
        gru_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Load and tamper
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["backend_manifest"]["hidden_dim"] = 128  # Wrong value

        # Save tampered checkpoint
        tampered_path = Path(tmpdir) / "tampered.pt"
        torch.save(checkpoint, tampered_path)

        # Attempt strict load - should reject
        with pytest.raises(ValueError, match="Backend provenance mismatch for 'hidden_dim'"):
            load_checkpoint(tampered_path, strict=True)


def test_gru_tampered_num_layers_rejected():
    """GRU checkpoint with tampered num_layers must be rejected in strict mode."""
    config = HiMemConfig(
        backend="gru_reference",
        memory_dim=64,
        gru_hidden_dim=64,
        gru_num_layers=2,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Load and tamper
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["backend_manifest"]["num_layers"] = 1  # Wrong value

        # Save tampered checkpoint
        tampered_path = Path(tmpdir) / "tampered.pt"
        torch.save(checkpoint, tampered_path)

        # Attempt strict load - should reject
        with pytest.raises(ValueError, match="Backend provenance mismatch for 'num_layers'"):
            load_checkpoint(tampered_path, strict=True)


def test_gru_tampered_implementation_rejected():
    """GRU checkpoint with tampered implementation must be rejected in strict mode."""
    config = HiMemConfig(
        backend="gru_reference",
        memory_dim=64,
        gru_hidden_dim=64,
        gru_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Load and tamper
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["backend_manifest"]["implementation"] = "fake.FakeGRU"

        # Save tampered checkpoint
        tampered_path = Path(tmpdir) / "tampered.pt"
        torch.save(checkpoint, tampered_path)

        # Attempt strict load - should reject
        with pytest.raises(ValueError, match="Backend provenance mismatch for 'implementation'"):
            load_checkpoint(tampered_path, strict=True)


def test_gru_missing_required_field_rejected():
    """GRU checkpoint missing required provenance field must be rejected."""
    config = HiMemConfig(
        backend="gru_reference",
        memory_dim=64,
        gru_hidden_dim=64,
        gru_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        save_checkpoint(encoder, checkpoint_path)

        # Load and remove required field
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        del checkpoint["backend_manifest"]["torch_version"]

        # Save corrupted checkpoint
        corrupted_path = Path(tmpdir) / "corrupted.pt"
        torch.save(checkpoint, corrupted_path)

        # Attempt strict load - should reject
        with pytest.raises(ValueError, match="missing required field 'torch_version'"):
            load_checkpoint(corrupted_path, strict=True)


def test_gru_valid_checkpoint_accepted():
    """Valid GRU checkpoint with complete provenance should load successfully."""
    config = HiMemConfig(
        backend="gru_reference",
        memory_dim=64,
        gru_hidden_dim=64,
        gru_num_layers=2,
        token_schema=TokenSchema(
            visual_feature_dim=32,
            visual_delta_dim=32,
            event_observable_dim=16,
        ),
    )

    encoder = create_himem_encoder(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.pt"
        manifest = save_checkpoint(encoder, checkpoint_path)

        # Strict load should succeed
        loaded = load_checkpoint(checkpoint_path, strict=True)

        # Verify backend type matches
        assert loaded.get_backend_manifest()["backend_type"] == "gru_reference"
        assert loaded.get_backend_manifest()["hidden_dim"] == 64
        assert loaded.get_backend_manifest()["num_layers"] == 2
