"""
Checkpoint management for B01R.

Implements save/load with validation of:
- Model config and schema hashes
- Backend provenance
- State dict consistency
- Deterministic reload verification
"""

import torch
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from .config import HiMemConfig
from .encoder import HiMemEncoder


def compute_state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Compute SHA256 hash of model state dict.

    Args:
        state_dict: Model state dictionary

    Returns:
        Hex digest of hash
    """
    # Serialize state dict deterministically
    state_bytes = []
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        state_bytes.append(key.encode())
        state_bytes.append(tensor.cpu().numpy().tobytes())

    combined = b"".join(state_bytes)
    return hashlib.sha256(combined).hexdigest()


def save_checkpoint(
    model: HiMemEncoder,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Save model checkpoint with full provenance.

    Args:
        model: HiMemEncoder instance
        path: Save path (will create parent dirs)
        metadata: Optional additional metadata

    Returns:
        Checkpoint manifest
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get config and backend info
    config = model.get_config()
    backend_manifest = model.get_backend_manifest()
    param_counts = model.count_parameters()

    # Get state dict
    state_dict = model.state_dict()
    state_dict_hash = compute_state_dict_hash(state_dict)

    # Build checkpoint
    checkpoint = {
        "config": config.to_dict(),
        "config_hash": config.compute_hash(),
        "token_schema_hash": config.token_schema.compute_hash(),
        "backend_manifest": backend_manifest,
        "parameter_counts": param_counts,
        "state_dict": state_dict,
        "state_dict_hash": state_dict_hash,
        "protocol_version": "3.0",
        "schema_version": "b01r_causal",
    }

    # Add optional metadata
    if metadata:
        checkpoint["metadata"] = metadata

    # Save
    torch.save(checkpoint, path)

    # Create manifest file
    manifest = {
        "checkpoint_path": str(path),
        "config_hash": checkpoint["config_hash"],
        "token_schema_hash": checkpoint["token_schema_hash"],
        "state_dict_hash": state_dict_hash,
        "backend_type": config.backend,
        "backend_manifest": backend_manifest,
        "parameter_counts": param_counts,
        "protocol_version": checkpoint["protocol_version"],
        "schema_version": checkpoint["schema_version"],
    }

    manifest_path = path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
    expected_config: Optional[HiMemConfig] = None,
    expected_backend: Optional[str] = None,
    expected_window_size: Optional[int] = None,
    expected_schema_hash: Optional[str] = None,
    expected_provenance: Optional[Dict[str, Any]] = None,
) -> HiMemEncoder:
    """
    Load model checkpoint with validation.

    Args:
        path: Checkpoint path
        device: Target device (default: CPU)
        strict: If True, enforce strict validation
        expected_config: Expected configuration (validates compatibility)
        expected_backend: Expected backend type
        expected_window_size: Expected window size
        expected_schema_hash: Expected token schema hash
        expected_provenance: Expected backend provenance (implementation, path, version, etc.)

    Returns:
        Loaded HiMemEncoder instance

    Raises:
        ValueError: If validation fails in strict mode or expected contract violated
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = device or torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct config
    config_dict = checkpoint["config"]

    # Validate expected contract BEFORE constructing model
    if expected_config is not None:
        # B01R5: Compare complete resolved config, not just selected fields
        expected_dict = expected_config.to_dict()

        # Compare all config fields (excluding runtime-specific like rtol/atol which may vary)
        comparable_fields = [
            "window_size", "memory_dim", "backend",
            "gru_hidden_dim", "gru_num_layers",
            "mamba2_d_model", "mamba2_d_state", "mamba2_d_conv", "mamba2_expand",
            "mamba2_headdim", "mamba2_num_layers", "mamba2_use_mem_eff_path",
            "simplified_ssm_state_dim", "simplified_ssm_num_layers",
            "dropout", "layer_norm_eps"
        ]

        for field in comparable_fields:
            if field in config_dict and field in expected_dict:
                if config_dict[field] != expected_dict[field]:
                    raise ValueError(
                        f"Config field '{field}' mismatch: "
                        f"checkpoint has {config_dict[field]}, expected {expected_dict[field]}"
                    )

        # Check token schema hash
        expected_hash = expected_config.token_schema.compute_hash()
        checkpoint_hash = checkpoint.get("token_schema_hash")
        if checkpoint_hash != expected_hash:
            raise ValueError(
                f"Token schema mismatch: checkpoint hash {checkpoint_hash[:16]}..., "
                f"expected {expected_hash[:16]}..."
            )

    # Validate individual expected parameters
    if expected_backend is not None and config_dict["backend"] != expected_backend:
        raise ValueError(
            f"Backend mismatch: checkpoint has '{config_dict['backend']}', "
            f"expected '{expected_backend}'"
        )

    if expected_window_size is not None and config_dict["window_size"] != expected_window_size:
        raise ValueError(
            f"Window size mismatch: checkpoint has {config_dict['window_size']}, "
            f"expected {expected_window_size}"
        )

    if expected_schema_hash is not None:
        checkpoint_hash = checkpoint.get("token_schema_hash")
        if checkpoint_hash != expected_schema_hash:
            raise ValueError(
                f"Schema hash mismatch: checkpoint has {checkpoint_hash[:16]}..., "
                f"expected {expected_schema_hash[:16]}..."
            )

    # Reconstruct token schema
    token_schema_dict = config_dict.get("token_schema", {})
    from .config import TokenSchema
    token_schema = TokenSchema(
        visual_feature_dim=token_schema_dict.get("visual_feature_dim", 256),
        state_dim=token_schema_dict.get("state_dim", 8),
        action_summary_dim=token_schema_dict.get("action_summary_dim", 14),
        has_previous_action=token_schema_dict.get("has_previous_action", True),
        visual_delta_dim=token_schema_dict.get("visual_delta_dim", 256),
        state_delta_dim=token_schema_dict.get("state_delta_dim", 8),
        event_observable_dim=token_schema_dict.get("event_observable_dim", 32),
    )

    config = HiMemConfig(
        window_size=config_dict["window_size"],
        memory_dim=config_dict["memory_dim"],
        backend=config_dict["backend"],
        token_schema=token_schema,
        gru_hidden_dim=config_dict.get("gru_hidden_dim", 256),
        gru_num_layers=config_dict.get("gru_num_layers", 1),
        mamba2_d_model=config_dict.get("mamba2_d_model", 256),
        mamba2_d_state=config_dict.get("mamba2_d_state", 64),
        mamba2_d_conv=config_dict.get("mamba2_d_conv", 4),
        mamba2_expand=config_dict.get("mamba2_expand", 2),
        mamba2_headdim=config_dict.get("mamba2_headdim", 64),
        mamba2_num_layers=config_dict.get("mamba2_num_layers", 2),
        mamba2_use_mem_eff_path=config_dict.get("mamba2_use_mem_eff_path", False),
        simplified_ssm_state_dim=config_dict.get("simplified_ssm_state_dim", 64),
        simplified_ssm_num_layers=config_dict.get("simplified_ssm_num_layers", 1),
        dropout=config_dict.get("dropout", 0.1),
        layer_norm_eps=config_dict.get("layer_norm_eps", 1e-5),
        rtol=config_dict.get("rtol", 1e-5),
        atol=config_dict.get("atol", 1e-7),
    )

    # Validate config hash
    expected_config_hash = checkpoint["config_hash"]
    actual_config_hash = config.compute_hash()
    if strict and expected_config_hash != actual_config_hash:
        raise ValueError(
            f"Config hash mismatch: expected {expected_config_hash}, "
            f"got {actual_config_hash}"
        )

    # Validate schema hash
    expected_schema_hash = checkpoint["token_schema_hash"]
    actual_schema_hash = config.token_schema.compute_hash()
    if strict and expected_schema_hash != actual_schema_hash:
        raise ValueError(
            f"Token schema hash mismatch: expected {expected_schema_hash}, "
            f"got {actual_schema_hash}"
        )

    # Create model
    model = HiMemEncoder(config).to(device)

    # Load state dict
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=strict)

    # Validate state dict hash
    if strict:
        expected_state_hash = checkpoint["state_dict_hash"]
        actual_state_hash = compute_state_dict_hash(model.state_dict())
        if expected_state_hash != actual_state_hash:
            raise ValueError(
                f"State dict hash mismatch after loading: "
                f"expected {expected_state_hash}, got {actual_state_hash}"
            )

    # Validate backend type
    loaded_backend = model.get_backend_manifest()["backend_type"]
    expected_backend_type = checkpoint["backend_manifest"]["backend_type"]
    if strict and loaded_backend != expected_backend_type:
        raise ValueError(
            f"Backend mismatch: expected {expected_backend_type}, got {loaded_backend}"
        )

    # B01R5P2: Validate backend provenance unconditionally in strict mode
    # This validates that stored manifest matches actual constructed backend's required provenance
    if strict:
        checkpoint_manifest = checkpoint["backend_manifest"]
        loaded_manifest = model.get_backend_manifest()

        backend_type = checkpoint_manifest.get("backend_type")

        # Required fields for all backends
        required_fields = ["backend_type", "implementation"]

        # Backend-specific required provenance fields
        if backend_type == "gru_reference":
            required_fields.extend(["torch_version", "hidden_dim", "num_layers"])
        elif backend_type == "mamba2":
            required_fields.extend([
                "mamba_ssm_version", "mamba_ssm_path",
                "d_model", "d_state", "d_conv", "expand", "headdim", "num_layers"
            ])
        elif backend_type == "simplified_ssm_experimental":
            required_fields.extend(["state_dim", "num_layers"])

        # Validate required fields exist in checkpoint manifest
        for field in required_fields:
            if field not in checkpoint_manifest:
                raise ValueError(
                    f"Checkpoint backend manifest missing required field '{field}' for backend '{backend_type}'"
                )

        # Validate checkpoint manifest matches loaded backend manifest
        for field in required_fields:
            checkpoint_value = checkpoint_manifest.get(field)
            loaded_value = loaded_manifest.get(field)
            if checkpoint_value != loaded_value:
                raise ValueError(
                    f"Backend provenance mismatch for '{field}': "
                    f"checkpoint has '{checkpoint_value}', loaded backend has '{loaded_value}'"
                )

    # Validate expected provenance if provided (B01R5: strict internal consistency)
    if expected_provenance is not None:
        checkpoint_manifest = checkpoint["backend_manifest"]

        # Required backend manifest fields for strict validation
        required_fields = ["backend_type", "implementation"]

        # Backend-specific required fields
        backend_type = checkpoint_manifest.get("backend_type")
        if backend_type == "mamba2":
            required_fields.extend(["mamba_ssm_version", "mamba_ssm_path"])
        elif backend_type == "gru_reference":
            required_fields.extend(["torch_version"])

        # Validate required fields exist
        for field in required_fields:
            if field not in checkpoint_manifest:
                raise ValueError(
                    f"Backend manifest missing required field '{field}' for {backend_type}"
                )

        # Validate expected values match
        for key, expected_value in expected_provenance.items():
            checkpoint_value = checkpoint_manifest.get(key)
            if checkpoint_value != expected_value:
                raise ValueError(
                    f"Expected provenance mismatch for '{key}': "
                    f"checkpoint has '{checkpoint_value}', expected '{expected_value}'"
                )

    return model


def verify_checkpoint_determinism(
    checkpoint_path: Path,
    test_input: torch.Tensor,
    device: Optional[torch.device] = None,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> Dict[str, Any]:
    """
    Verify that checkpoint loading is deterministic.

    Loads the checkpoint twice and compares outputs on the same input.

    Args:
        checkpoint_path: Path to checkpoint
        test_input: Test input tensor [seq_len, token_dim]
        device: Target device
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Verification results dict
    """
    device = device or torch.device("cpu")
    test_input = test_input.to(device)

    # Load first time
    model1 = load_checkpoint(checkpoint_path, device=device, strict=True)
    model1.eval()

    # Load second time
    model2 = load_checkpoint(checkpoint_path, device=device, strict=True)
    model2.eval()

    # Create dummy tokens (synthetic test)
    from .config import CausalToken
    schema = model1.config.token_schema
    tokens = []
    for i in range(test_input.size(0)):
        # B01R5P2: Properly format decision 0 with zeros in required positions
        if i == 0:
            token_input = torch.zeros_like(test_input[i])
            # Set non-zero values only for visual_feature, state, and event_observable
            token_input[:schema.visual_feature_dim] = test_input[i][:schema.visual_feature_dim]
            offset = schema.visual_feature_dim
            token_input[offset:offset + schema.state_dim] = test_input[i][offset:offset + schema.state_dim]
            # event_observable is at the end
            offset = (schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim +
                      1 + schema.visual_delta_dim + schema.state_delta_dim)
            token_input[offset:] = test_input[i][offset:]
        else:
            token_input = test_input[i].clone()
            # Set has_previous to exactly 1.0
            offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim
            token_input[offset] = 1.0

        token = CausalToken(
            model_input=token_input,
            audit_metadata={
                "env_id": "test_env",
                "episode_id": "test_episode",
                "decision_index": i,
                "decision_start_frame": 0 if i == 0 else i * 10,
                "prev_executed_start": None if i == 0 else (i - 1) * 10,
                "prev_executed_end": None if i == 0 else i * 10,
                "prev_executed_count": None if i == 0 else 10,
                "end_reason": None if i == 0 else "queue_depleted",
                "schema_version": "b01r_causal",
                "schema_hash": model1.config.token_schema.compute_hash(),
            }
        )
        tokens.append(token)

    # Forward pass
    with torch.no_grad():
        output1 = model1.forward_from_tokens(tokens, return_all=True)
        output2 = model2.forward_from_tokens(tokens, return_all=True)

    # Compare
    is_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    max_diff = (output1 - output2).abs().max().item()

    return {
        "deterministic": is_close,
        "max_absolute_difference": max_diff,
        "rtol": rtol,
        "atol": atol,
        "output_shape": list(output1.shape),
    }
