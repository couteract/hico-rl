"""
Main HiMem encoder for B01R.

Integrates token encoding, rolling window management, and memory backend.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from .config import HiMemConfig, CausalToken, TokenSchema
from .event_features import EventFeatureEncoder
from .backends import create_backend
from .state_manager import RollingWindowStateManager


class HiMemEncoder(nn.Module):
    """
    HiMem Event-Mamba encoder with rolling window semantics.

    B01R implementation:
    - Per-environment ring buffer of up to W=8 recent tokens
    - Each query recomputes from zero state: m_k = Backend_zero(recent_W)[-1]
    - No persistent hidden state across windows
    """

    def __init__(self, config: HiMemConfig):
        super().__init__()
        self.config = config

        # Token encoder
        self.feature_encoder = EventFeatureEncoder(config.token_schema)

        # Input projection if token_dim != memory_dim
        token_dim = config.token_schema.token_dim
        if token_dim != config.memory_dim:
            self.input_proj = nn.Linear(token_dim, config.memory_dim)
        else:
            self.input_proj = nn.Identity()

        # Memory backend
        self.backend = create_backend(
            backend_type=config.backend,
            input_dim=config.memory_dim,
            output_dim=config.memory_dim,
            num_layers=(
                config.gru_num_layers if config.backend == "gru_reference"
                else config.mamba2_num_layers if config.backend == "mamba2"
                else config.simplified_ssm_num_layers
            ),
            d_state=config.mamba2_d_state,
            d_conv=config.mamba2_d_conv,
            expand=config.mamba2_expand,
            headdim=config.mamba2_headdim,
            use_mem_eff_path=config.mamba2_use_mem_eff_path,
            state_dim=config.simplified_ssm_state_dim,  # FIXED: was simplified_ssm_num_layers
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )

        # Output projection to memory_dim (if needed)
        self.output_proj = nn.Linear(config.memory_dim, config.memory_dim)

        # State manager (not serialized with model)
        self.state_manager: Optional[RollingWindowStateManager] = None

    def create_state_manager(self) -> RollingWindowStateManager:
        """Create a new state manager instance."""
        return RollingWindowStateManager(self.config)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Standard forward pass for raw token tensors.

        Args:
            x: Token tensors, shape [batch, seq_len, token_dim] or [seq_len, token_dim]
            return_all: If True, return all positions; if False, return only last

        Returns:
            If return_all: [batch, seq_len, memory_dim] or [seq_len, memory_dim]
            Otherwise: [batch, memory_dim] or [memory_dim]
        """
        # Validate input tensor
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be torch.Tensor, got {type(x)}")

        if x.ndim not in [2, 3]:
            raise ValueError(
                f"Input must be rank 2 [seq_len, token_dim] or rank 3 [batch, seq_len, token_dim], "
                f"got rank {x.ndim} with shape {x.shape}"
            )

        if not x.dtype.is_floating_point:
            raise ValueError(f"Input must be floating point dtype, got {x.dtype}")

        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or Inf values")

        # Check dimension matches schema
        expected_token_dim = self.config.token_schema.token_dim
        if x.shape[-1] != expected_token_dim:
            raise ValueError(
                f"Token dimension mismatch: expected {expected_token_dim}, got {x.shape[-1]}"
            )

        # Check non-empty sequence
        seq_dim = -2 if x.ndim == 3 else 0
        if x.shape[seq_dim] == 0:
            raise ValueError("Cannot process empty sequence")

        # Handle both batched and unbatched inputs
        if x.ndim == 2:
            # [seq_len, token_dim] -> [1, seq_len, token_dim]
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Project to memory_dim
        x = self.input_proj(x)  # [batch, seq_len, memory_dim]

        # Pass through backend (from zero state)
        output, _ = self.backend(x, prev_state=None)  # [batch, seq_len, memory_dim]

        # Output projection
        memory = self.output_proj(output)  # [batch, seq_len, memory_dim]

        if not return_all:
            # Return only last position
            memory = memory[:, -1, :]  # [batch, memory_dim]

        if squeeze_batch:
            memory = memory.squeeze(0)

        return memory

    def forward_from_tokens(
        self,
        tokens: List[CausalToken],
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Compute memory from a sequence of tokens.

        This is the core rolling window computation:
        m_k = Encoder_zero(tokens[max(0, k-W+1):k+1])[-1]

        Args:
            tokens: List of CausalToken objects (must be <= W window size)
            return_all: If True, return all sequence positions; if False, return only last

        Returns:
            If return_all: [seq_len, memory_dim]
            Otherwise: [memory_dim]
        """
        if not tokens:
            raise ValueError("Cannot encode empty token sequence")

        # Enforce window size limit
        if len(tokens) > self.config.window_size:
            raise ValueError(
                f"Token sequence length {len(tokens)} exceeds window size {self.config.window_size}. "
                f"Use only the most recent W={self.config.window_size} tokens."
            )

        # Validate all tokens first
        for i, token in enumerate(tokens):
            token.validate(self.config.token_schema)

        # Validate sequence continuity (always, not just when state_manager attached)
        self._validate_token_sequence(tokens)

        # Stack tokens into batch
        token_tensors = self.feature_encoder(tokens)  # [seq_len, token_dim]
        token_tensors = token_tensors.unsqueeze(0)  # [1, seq_len, token_dim]

        # Project to memory_dim
        x = self.input_proj(token_tensors)  # [1, seq_len, memory_dim]

        # Pass through backend (from zero state)
        output, _ = self.backend(x, prev_state=None)  # [1, seq_len, memory_dim]

        # Output projection
        memory = self.output_proj(output)  # [1, seq_len, memory_dim]

        # Remove batch dimension
        memory = memory.squeeze(0)  # [seq_len, memory_dim]

        if return_all:
            return memory
        else:
            return memory[-1]  # [memory_dim]

    def forward_batch(
        self,
        batch_tokens: List[List[CausalToken]],
        return_all: bool = False,
        pad_value: float = 0.0
    ) -> torch.Tensor:
        """
        Compute memory for a batch of token sequences.

        Sequences may have different lengths. They will be padded to the
        longest sequence in the batch.

        Args:
            batch_tokens: List of token sequences (each must be <= W)
            return_all: If True, return all positions; if False, return only last valid
            pad_value: Value to use for padding

        Returns:
            If return_all: [batch, max_seq_len, memory_dim] (padded)
            Otherwise: [batch, memory_dim] (last valid position per sequence)
        """
        if not batch_tokens:
            raise ValueError("Empty batch")

        batch_size = len(batch_tokens)
        lengths = [len(seq) for seq in batch_tokens]
        max_len = max(lengths)

        # Validate each sequence independently (includes W check and sequence validation)
        for i, tokens in enumerate(batch_tokens):
            if not tokens:
                raise ValueError(f"Sequence {i} in batch is empty")

        # Process each sequence and collect results
        all_memories = []
        for tokens, length in zip(batch_tokens, lengths):
            memory = self.forward_from_tokens(tokens, return_all=True)  # [seq_len, memory_dim]

            # Pad to max_len if needed
            if length < max_len:
                pad_size = max_len - length
                padding = torch.full(
                    (pad_size, memory.size(-1)),
                    pad_value,
                    device=memory.device,
                    dtype=memory.dtype
                )
                memory = torch.cat([memory, padding], dim=0)

            all_memories.append(memory)

        # Stack into batch
        batch_memory = torch.stack(all_memories, dim=0)  # [batch, max_len, memory_dim]

        if return_all:
            return batch_memory
        else:
            # Extract last valid position for each sequence
            last_valid = []
            for i, length in enumerate(lengths):
                last_valid.append(batch_memory[i, length - 1])
            return torch.stack(last_valid, dim=0)  # [batch, memory_dim]

    def _validate_token_sequence(self, tokens: List[CausalToken]) -> None:
        """
        Validate token sequence for protocol compliance.

        This enforces sequence validation independent of state_manager attachment.

        Args:
            tokens: List of tokens to validate

        Raises:
            ValueError: If validation fails
        """
        if not tokens:
            raise ValueError("Cannot validate empty token sequence")

        # Extract metadata
        env_ids = [t.audit_metadata["env_id"] for t in tokens]
        episode_ids = [t.audit_metadata["episode_id"] for t in tokens]
        decision_indices = [t.audit_metadata["decision_index"] for t in tokens]

        # All tokens must be from same environment and episode
        unique_envs = set(env_ids)
        if len(unique_envs) > 1:
            raise ValueError(
                f"Cross-env tokens not allowed in forward_from_tokens: {unique_envs}"
            )

        unique_episodes = set(episode_ids)
        if len(unique_episodes) > 1:
            raise ValueError(
                f"Cross-episode tokens not allowed in forward_from_tokens: {unique_episodes}"
            )

        # Check for duplicate decisions
        seen_indices = set()
        for idx in decision_indices:
            if idx in seen_indices:
                raise ValueError(
                    f"Duplicate decision index {idx} in sequence"
                )
            seen_indices.add(idx)

        # Check chronological order (must be consecutive)
        for i in range(len(decision_indices) - 1):
            if decision_indices[i + 1] != decision_indices[i] + 1:
                raise ValueError(
                    f"Non-consecutive decisions: {decision_indices[i]} -> {decision_indices[i+1]}"
                )

        # Validate adjacent interval continuity
        for i in range(1, len(tokens)):
            prev_token = tokens[i - 1]
            curr_token = tokens[i]

            prev_start = prev_token.audit_metadata["decision_start_frame"]
            curr_prev_start = curr_token.audit_metadata["prev_executed_start"]
            curr_prev_end = curr_token.audit_metadata["prev_executed_end"]
            curr_start = curr_token.audit_metadata["decision_start_frame"]

            # Current token's previous interval must match previous decision's actual interval
            if curr_prev_start != prev_start:
                raise ValueError(
                    f"Interval discontinuity: decision {decision_indices[i]} claims "
                    f"prev_start={curr_prev_start}, but previous decision {decision_indices[i-1]} "
                    f"started at {prev_start}"
                )

            if curr_prev_end != curr_start:
                raise ValueError(
                    f"Interval discontinuity: decision {decision_indices[i]} at frame {curr_start} "
                    f"claims prev_end={curr_prev_end}, but they must match"
                )

    def get_config(self) -> HiMemConfig:
        """Return the configuration."""
        return self.config

    def get_backend_manifest(self) -> Dict[str, Any]:
        """Get backend provenance information."""
        manifest = self.backend.get_backend_manifest()
        manifest["config_hash"] = self.config.compute_hash()
        manifest["token_schema_hash"] = self.config.token_schema.compute_hash()
        return manifest

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Per-component counts
        feature_encoder_params = sum(p.numel() for p in self.feature_encoder.parameters())
        backend_params = sum(p.numel() for p in self.backend.parameters())
        proj_params = sum(p.numel() for p in self.input_proj.parameters()) + sum(
            p.numel() for p in self.output_proj.parameters()
        )

        return {
            "total": total,
            "trainable": trainable,
            "feature_encoder": feature_encoder_params,
            "backend": backend_params,
            "projections": proj_params,
        }


def create_himem_encoder(
    config: Optional[HiMemConfig] = None,
    backend: Optional[str] = None,
    **kwargs
) -> HiMemEncoder:
    """
    Factory function to create HiMem encoder.

    Args:
        config: Full configuration object (takes precedence)
        backend: Backend name (if config not provided)
        **kwargs: Additional config overrides

    Returns:
        HiMemEncoder instance
    """
    if config is None:
        config = HiMemConfig(backend=backend or "gru_reference", **kwargs)

    return HiMemEncoder(config)
