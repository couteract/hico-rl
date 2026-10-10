"""
HiMem configuration and protocol definitions for B01R.

This module defines:
- Token schema (causal model inputs vs audit metadata)
- Backend configurations (gru_reference, mamba2, simplified_ssm_experimental)
- Protocol version and validation rules
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any
import hashlib
import json


# Protocol version for B01R
HIMEM_PROTOCOL_VERSION = "3.0"
HIMEM_SCHEMA_VERSION = "b01r_causal"


@dataclass
class TokenSchema:
    """
    Defines the causal token structure for memory computation.

    The k-th memory m_k is computed BEFORE generating action chunk k,
    so x_k can only contain information available at that time.
    """

    # Causal model inputs (what goes into the network)
    visual_feature_dim: int = 256  # Frozen vision backbone output
    state_dim: int = 8  # Current robot state (strictly 8D)
    action_summary_dim: int = 14  # Previous executed chunk summary (mean + last)
    has_previous_action: bool = True  # First decision = False, rest = True
    visual_delta_dim: int = 256  # Current vs previous visual feature delta
    state_delta_dim: int = 8  # Current vs previous state delta
    event_observable_dim: int = 32  # Observable event vector (no reward/success/failure)

    def __post_init__(self):
        """Validate schema construction."""
        # Enforce exact dimensions
        if self.state_dim != 8:
            raise ValueError(
                f"state_dim must be exactly 8 (robot state dimension), got {self.state_dim}"
            )

        if self.action_summary_dim != 14:
            raise ValueError(
                f"action_summary_dim must be exactly 14 (mean7 + last7), got {self.action_summary_dim}"
            )

        # All dimensions must be positive integers
        for field_name in ["visual_feature_dim", "visual_delta_dim", "event_observable_dim"]:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive integer, got {value}")

        # state_delta_dim must match state_dim
        if self.state_delta_dim != self.state_dim:
            raise ValueError(
                f"state_delta_dim must equal state_dim, got state_delta_dim={self.state_delta_dim}, "
                f"state_dim={self.state_dim}"
            )

        # visual_delta_dim must match visual_feature_dim (B01R5 requirement)
        if self.visual_delta_dim != self.visual_feature_dim:
            raise ValueError(
                f"visual_delta_dim must equal visual_feature_dim, got visual_delta_dim={self.visual_delta_dim}, "
                f"visual_feature_dim={self.visual_feature_dim}"
            )

    # Total token dimension
    @property
    def token_dim(self) -> int:
        return (
            self.visual_feature_dim +
            self.state_dim +
            self.action_summary_dim +
            1 +  # has_previous_action flag
            self.visual_delta_dim +
            self.state_delta_dim +
            self.event_observable_dim
        )

    # Audit metadata (NOT model inputs)
    # These fields are for validation and tracing only
    audit_fields: list[str] = field(default_factory=lambda: [
        "env_id",
        "episode_id",
        "decision_index",
        "decision_start_frame",
        "prev_executed_start",
        "prev_executed_end",
        "prev_executed_count",
        "end_reason",
        "schema_version",
        "schema_hash"
    ])

    def compute_hash(self) -> str:
        """Compute SHA256 hash of schema for checkpoint validation."""
        schema_dict = {
            "visual_feature_dim": self.visual_feature_dim,
            "state_dim": self.state_dim,
            "action_summary_dim": self.action_summary_dim,
            "visual_delta_dim": self.visual_delta_dim,
            "state_delta_dim": self.state_delta_dim,
            "event_observable_dim": self.event_observable_dim,
            "version": HIMEM_SCHEMA_VERSION
        }
        schema_json = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()


@dataclass
class HiMemConfig:
    """
    Configuration for HiMem Event-Mamba module.

    B01R implements rolling window (W=8) with zero-state recomputation.
    """

    # Window size (maximum recent tokens to keep)
    window_size: int = 8

    # Token schema
    token_schema: TokenSchema = field(default_factory=TokenSchema)

    # Memory output dimension
    memory_dim: int = 256

    # Backend selection
    backend: Literal["gru_reference", "mamba2", "simplified_ssm_experimental"] = "gru_reference"

    # Backend-specific parameters
    # GRU reference
    gru_hidden_dim: int = 256
    gru_num_layers: int = 1

    # Mamba2 (official mamba_ssm)
    mamba2_d_model: int = 256
    mamba2_d_state: int = 64
    mamba2_d_conv: int = 4
    mamba2_expand: int = 2
    mamba2_headdim: int = 64
    mamba2_num_layers: int = 2
    mamba2_use_mem_eff_path: bool = False  # False because causal-conv1d not installed

    # SimplifiedSSM experimental (pure PyTorch)
    simplified_ssm_state_dim: int = 64
    simplified_ssm_num_layers: int = 1

    # Training parameters
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Numerical tolerance for determinism tests
    rtol: float = 1e-5
    atol: float = 1e-7

    def __post_init__(self):
        """Validate configuration consistency."""
        # Enforce memory_dim == mamba2_d_model when using mamba2 backend
        if self.backend == "mamba2" and self.memory_dim != self.mamba2_d_model:
            raise ValueError(
                f"When using mamba2 backend, memory_dim must equal mamba2_d_model. "
                f"Got memory_dim={self.memory_dim}, mamba2_d_model={self.mamba2_d_model}"
            )

        # Enforce memory_dim == gru_hidden_dim when using gru_reference backend
        if self.backend == "gru_reference" and self.memory_dim != self.gru_hidden_dim:
            raise ValueError(
                f"When using gru_reference backend, memory_dim must equal gru_hidden_dim. "
                f"Got memory_dim={self.memory_dim}, gru_hidden_dim={self.gru_hidden_dim}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["token_schema_hash"] = self.token_schema.compute_hash()
        result["protocol_version"] = HIMEM_PROTOCOL_VERSION
        result["schema_version"] = HIMEM_SCHEMA_VERSION
        return result

    def compute_hash(self) -> str:
        """Compute configuration hash for checkpoint validation."""
        config_dict = self.to_dict()
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()


@dataclass
class CausalToken:
    """
    A single causal token for memory computation.

    Attributes:
        model_input: The actual tensor fed to the network (causal information only)
        audit_metadata: Dict containing episode_id, decision_index, etc. (not fed to network)
    """

    # This will be a torch.Tensor in practice
    model_input: Any  # shape: [token_dim]
    audit_metadata: Dict[str, Any]

    def validate(self, schema: TokenSchema) -> None:
        """
        Validate token structure and causality constraints.

        Raises:
            ValueError: If validation fails
        """
        import torch

        # Check model_input is a tensor
        if not isinstance(self.model_input, torch.Tensor):
            raise ValueError(f"model_input must be torch.Tensor, got {type(self.model_input)}")

        # Check exact rank (must be rank 1)
        if self.model_input.ndim != 1:
            raise ValueError(
                f"model_input must be rank 1 (1D tensor), got rank {self.model_input.ndim} "
                f"with shape {self.model_input.shape}"
            )

        # Check dimension
        if self.model_input.shape[0] != schema.token_dim:
            raise ValueError(
                f"Token dimension mismatch: expected {schema.token_dim}, "
                f"got {self.model_input.shape[0]}"
            )

        # Check dtype is floating point
        if not self.model_input.dtype.is_floating_point:
            raise ValueError(
                f"model_input must be floating point dtype, got {self.model_input.dtype}"
            )

        # Check finite values
        if not torch.isfinite(self.model_input).all():
            raise ValueError("Token contains NaN or Inf values")

        # Check required audit fields
        for field_name in schema.audit_fields:
            if field_name not in self.audit_metadata:
                raise ValueError(f"Missing required audit field: {field_name}")

        # Check for forbidden fields (future/RL-specific information)
        forbidden_fields = [
            "reward", "success", "failure", "done", "advantage", "value",
            "action_chunk_probabilities", "acp", "current_action_end_frame",
            "current_action_count", "future_executed_count"
        ]
        for forbidden in forbidden_fields:
            if forbidden in self.audit_metadata:
                raise ValueError(
                    f"Forbidden field '{forbidden}' found in audit_metadata. "
                    f"Token k can only contain information available BEFORE generating action chunk k."
                )

        # Validate schema version
        if self.audit_metadata.get("schema_version") != HIMEM_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {HIMEM_SCHEMA_VERSION}, "
                f"got {self.audit_metadata.get('schema_version')}"
            )

        # Validate schema hash
        expected_hash = schema.compute_hash()
        actual_hash = self.audit_metadata.get("schema_hash")
        if actual_hash != expected_hash:
            raise ValueError(
                f"Schema hash mismatch: expected {expected_hash[:8]}..., "
                f"got {actual_hash[:8] if actual_hash else 'None'}..."
            )

        # Validate decision_index is non-negative integer (reject bool)
        decision_index = self.audit_metadata.get("decision_index")
        if not isinstance(decision_index, int) or isinstance(decision_index, bool) or decision_index < 0:
            raise ValueError(
                f"decision_index must be non-negative int (not bool), got {decision_index} (type: {type(decision_index).__name__})"
            )

        # Validate decision_start_frame is non-negative integer (reject bool)
        decision_start_frame = self.audit_metadata.get("decision_start_frame")
        if not isinstance(decision_start_frame, int) or isinstance(decision_start_frame, bool) or decision_start_frame < 0:
            raise ValueError(
                f"decision_start_frame must be non-negative int (not bool), got {decision_start_frame} (type: {type(decision_start_frame).__name__})"
            )

        # First decision (index 0) must start at frame 0
        if decision_index == 0 and decision_start_frame != 0:
            raise ValueError(
                f"First decision (index 0) must start at frame 0, got {decision_start_frame}"
            )

        # Validate end_reason
        end_reason = self.audit_metadata.get("end_reason")
        if decision_index == 0:
            # First decision must have null end_reason
            if end_reason is not None:
                raise ValueError(
                    f"First decision (index 0) must have null end_reason, got {end_reason}"
                )
        else:
            # Subsequent decisions must have recognized end_reason (B01R5: add timeout)
            valid_end_reasons = [
                "queue_depleted", "early_replan", "terminal", "truncated", "timeout"
            ]
            if end_reason not in valid_end_reasons:
                raise ValueError(
                    f"end_reason must be one of {valid_end_reasons}, got {end_reason}"
                )

        # Validate execution interval consistency
        prev_start = self.audit_metadata.get("prev_executed_start")
        prev_end = self.audit_metadata.get("prev_executed_end")
        prev_count = self.audit_metadata.get("prev_executed_count")

        # Validate execution interval consistency
        prev_start = self.audit_metadata.get("prev_executed_start")
        prev_end = self.audit_metadata.get("prev_executed_end")
        prev_count = self.audit_metadata.get("prev_executed_count")

        if decision_index == 0:
            # First decision: no previous execution
            if prev_start is not None or prev_end is not None or prev_count is not None:
                raise ValueError(
                    f"First decision (index 0) must have null execution interval, "
                    f"got start={prev_start}, end={prev_end}, count={prev_count}"
                )
        else:
            # Subsequent decisions: must have valid execution interval
            if prev_start is None or prev_end is None or prev_count is None:
                raise ValueError(
                    f"Decision {decision_index} must have complete execution interval, "
                    f"got start={prev_start}, end={prev_end}, count={prev_count}"
                )

            # Validate interval bounds (reject bool for all integer fields)
            if not isinstance(prev_start, int) or isinstance(prev_start, bool):
                raise ValueError(
                    f"prev_executed_start must be int (not bool), got {prev_start} (type: {type(prev_start).__name__})"
                )
            if not isinstance(prev_end, int) or isinstance(prev_end, bool):
                raise ValueError(
                    f"prev_executed_end must be int (not bool), got {prev_end} (type: {type(prev_end).__name__})"
                )
            if not isinstance(prev_count, int) or isinstance(prev_count, bool):
                raise ValueError(
                    f"prev_executed_count must be int (not bool), got {prev_count} (type: {type(prev_count).__name__})"
                )

            if prev_start < 0:
                raise ValueError(f"prev_executed_start must be non-negative, got {prev_start}")

            if prev_end <= prev_start:
                raise ValueError(
                    f"Invalid execution interval: [{prev_start}, {prev_end}) is empty or reversed"
                )

            # Validate count is strictly positive and matches interval
            if prev_count <= 0:
                raise ValueError(f"prev_executed_count must be strictly positive, got {prev_count}")

            expected_count = prev_end - prev_start
            if prev_count != expected_count:
                raise ValueError(
                    f"Execution count mismatch: interval [{prev_start}, {prev_end}) "
                    f"implies count {expected_count}, got {prev_count}"
                )

        # Validate env_id and episode_id are strings
        env_id = self.audit_metadata.get("env_id")
        episode_id = self.audit_metadata.get("episode_id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError(f"env_id must be non-empty string, got {env_id}")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"episode_id must be non-empty string, got {episode_id}")

        # B01R5P2: Validate layout constraints on model_input tensor
        # Extract slices according to schema definition
        offset = 0
        visual_feature = self.model_input[offset:offset + schema.visual_feature_dim]
        offset += schema.visual_feature_dim

        state = self.model_input[offset:offset + schema.state_dim]
        offset += schema.state_dim

        action_summary = self.model_input[offset:offset + schema.action_summary_dim]
        offset += schema.action_summary_dim

        has_previous = self.model_input[offset]
        offset += 1

        visual_delta = self.model_input[offset:offset + schema.visual_delta_dim]
        offset += schema.visual_delta_dim

        state_delta = self.model_input[offset:offset + schema.state_delta_dim]
        offset += schema.state_delta_dim

        # For decision 0: action_summary, has_previous, visual_delta, state_delta must all be EXACTLY zero
        # B01R5P2: Use exact zero comparison, not tolerance-based allclose
        if decision_index == 0:
            import torch
            if not (action_summary == 0.0).all():
                raise ValueError(
                    f"Decision 0 must have exact zero action_summary, got non-zero values"
                )
            if has_previous.item() != 0.0:
                raise ValueError(
                    f"Decision 0 must have has_previous=0, got {has_previous.item()}"
                )
            if not (visual_delta == 0.0).all():
                raise ValueError(
                    f"Decision 0 must have exact zero visual_delta, got non-zero values"
                )
            if not (state_delta == 0.0).all():
                raise ValueError(
                    f"Decision 0 must have exact zero state_delta, got non-zero values"
                )
        else:
            # For decision k>0: has_previous must be exactly 1
            if has_previous.item() != 1.0:
                raise ValueError(
                    f"Decision {decision_index} must have has_previous=1, got {has_previous.item()}"
                )


# Default configurations
def get_debug_config() -> HiMemConfig:
    """Smaller config for testing."""
    return HiMemConfig(
        memory_dim=128,
        gru_hidden_dim=128,
        mamba2_d_model=128,
        mamba2_num_layers=1,
        token_schema=TokenSchema(
            visual_feature_dim=128,
            visual_delta_dim=128
        )
    )


def get_production_config() -> HiMemConfig:
    """Production config for real training."""
    return HiMemConfig(
        memory_dim=256,
        gru_hidden_dim=256,
        mamba2_d_model=256,
        mamba2_d_state=64,
        mamba2_d_conv=4,
        mamba2_expand=2,
        mamba2_headdim=64,
        mamba2_num_layers=2,
        backend="mamba2"
    )
