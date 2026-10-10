"""
Event feature encoder for B01R.

Constructs causal tokens from observations, states, and action history.
The k-th token x_k contains only information available BEFORE generating action chunk k.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .config import TokenSchema, CausalToken


class EventFeatureEncoder(nn.Module):
    """
    Encodes decision events into causal tokens for memory computation.

    The k-th memory m_k is computed before action chunk k is generated.
    Therefore x_k can only contain:
    - Current frozen visual feature (provided by caller)
    - Current state (8D)
    - Previous executed action chunk summary (mean + last)
    - Visual delta (current vs previous)
    - State delta (current vs previous)
    - Observable event vector (no reward/success/failure)

    Forbidden inputs:
    - Current chunk's future end_frame/executed_count
    - Predicted but not executed action tail
    - Current decision's success/failure/reward/done
    """

    def __init__(self, schema: TokenSchema):
        super().__init__()
        self.schema = schema

        # No learnable parameters in B01R - uses deterministic encoding
        # Real vision encoder and action summary will be provided by caller

    def encode_token(
        self,
        visual_feature: torch.Tensor,  # [visual_feature_dim] - from frozen backbone
        state: torch.Tensor,  # [8] - current robot state
        prev_action_summary: Optional[torch.Tensor],  # [14] - previous executed chunk
        prev_visual_feature: Optional[torch.Tensor],  # [visual_feature_dim]
        prev_state: Optional[torch.Tensor],  # [8]
        event_observable: torch.Tensor,  # [event_observable_dim] - no reward/success
        episode_id: str,
        env_id: str,
        decision_index: int,
        decision_start_frame: int,
        prev_executed_start: Optional[int],
        prev_executed_end: Optional[int],
        prev_executed_count: Optional[int],
        end_reason: str,
        schema_hash: str,
    ) -> CausalToken:
        """
        Encode a single decision event into a causal token.

        Args:
            visual_feature: Current frozen visual feature from backbone
            state: Current 8D robot state
            prev_action_summary: Summary of previous executed chunk (mean + last)
                                First decision: None → zeros
            prev_visual_feature: Previous visual feature for delta computation
            prev_state: Previous state for delta computation
            event_observable: Observable event vector (no future info)
            episode_id: Episode identifier for audit
            decision_index: Decision index within episode (0-indexed)
            decision_start_frame: Frame when this decision began
            prev_executed_start: Start frame of previous executed chunk
            prev_executed_end: End frame of previous executed chunk
            prev_executed_count: Number of frames in previous executed chunk
            end_reason: Reason for previous chunk ending
            schema_hash: Schema version hash for validation

        Returns:
            CausalToken with model_input tensor and audit_metadata
        """
        # Validate inputs - use explicit exceptions, not assert
        if visual_feature.ndim != 1:
            raise ValueError(f"visual_feature must be 1D, got shape {visual_feature.shape}")
        if visual_feature.shape[0] != self.schema.visual_feature_dim:
            raise ValueError(
                f"visual_feature dim mismatch: expected {self.schema.visual_feature_dim}, "
                f"got {visual_feature.shape[0]}"
            )

        if state.ndim != 1:
            raise ValueError(f"state must be 1D, got shape {state.shape}")
        if state.shape[0] != self.schema.state_dim:
            raise ValueError(
                f"state dim mismatch: expected {self.schema.state_dim}, got {state.shape[0]}"
            )

        if event_observable.ndim != 1:
            raise ValueError(f"event_observable must be 1D, got shape {event_observable.shape}")
        if event_observable.shape[0] != self.schema.event_observable_dim:
            raise ValueError(
                f"event_observable dim mismatch: expected {self.schema.event_observable_dim}, "
                f"got {event_observable.shape[0]}"
            )

        # Check all tensors are on same device
        device = visual_feature.device
        if state.device != device:
            raise ValueError(f"state device {state.device} != visual_feature device {device}")
        if event_observable.device != device:
            raise ValueError(f"event_observable device {event_observable.device} != {device}")

        # Check all tensors have same floating dtype
        dtype = visual_feature.dtype
        if not dtype.is_floating_point:
            raise ValueError(f"visual_feature must be floating point, got {dtype}")
        if state.dtype != dtype:
            raise ValueError(f"state dtype {state.dtype} != visual_feature dtype {dtype}")
        if event_observable.dtype != dtype:
            raise ValueError(f"event_observable dtype {event_observable.dtype} != {dtype}")

        # Check all inputs are finite
        if not torch.isfinite(visual_feature).all():
            raise ValueError("visual_feature contains NaN or Inf")
        if not torch.isfinite(state).all():
            raise ValueError("state contains NaN or Inf")
        if not torch.isfinite(event_observable).all():
            raise ValueError("event_observable contains NaN or Inf")

        # Validate first decision semantics
        if decision_index == 0:
            # First decision must have all previous inputs as None
            if prev_action_summary is not None:
                raise ValueError(
                    "First decision (index 0) must have prev_action_summary=None, "
                    f"got tensor with shape {prev_action_summary.shape}"
                )
            if prev_visual_feature is not None:
                raise ValueError(
                    "First decision (index 0) must have prev_visual_feature=None, "
                    f"got tensor with shape {prev_visual_feature.shape}"
                )
            if prev_state is not None:
                raise ValueError(
                    "First decision (index 0) must have prev_state=None, "
                    f"got tensor with shape {prev_state.shape}"
                )
        else:
            # Non-first decisions should have previous inputs
            if prev_action_summary is None:
                raise ValueError(
                    f"Decision {decision_index} must have non-null prev_action_summary"
                )
            if prev_visual_feature is None:
                raise ValueError(
                    f"Decision {decision_index} must have non-null prev_visual_feature"
                )
            if prev_state is None:
                raise ValueError(
                    f"Decision {decision_index} must have non-null prev_state"
                )

        # Previous action summary (first decision = zeros)
        if prev_action_summary is None:
            prev_action_summary = torch.zeros(
                self.schema.action_summary_dim, device=device, dtype=dtype
            )
            has_previous = torch.zeros(1, device=device, dtype=dtype)
        else:
            if prev_action_summary.ndim != 1:
                raise ValueError(
                    f"prev_action_summary must be 1D, got shape {prev_action_summary.shape}"
                )
            if prev_action_summary.shape[0] != self.schema.action_summary_dim:
                raise ValueError(
                    f"prev_action_summary dim mismatch: expected {self.schema.action_summary_dim}, "
                    f"got {prev_action_summary.shape[0]}"
                )
            if prev_action_summary.device != device:
                raise ValueError(
                    f"prev_action_summary device {prev_action_summary.device} != {device}"
                )
            if prev_action_summary.dtype != dtype:
                raise ValueError(
                    f"prev_action_summary dtype {prev_action_summary.dtype} != {dtype}"
                )
            if not torch.isfinite(prev_action_summary).all():
                raise ValueError("prev_action_summary contains NaN or Inf")
            has_previous = torch.ones(1, device=device, dtype=dtype)


        # Visual delta (first decision = zeros)
        if prev_visual_feature is None:
            visual_delta = torch.zeros(
                self.schema.visual_delta_dim, device=device, dtype=dtype
            )
        else:
            if prev_visual_feature.ndim != 1:
                raise ValueError(
                    f"prev_visual_feature must be 1D, got shape {prev_visual_feature.shape}"
                )
            if prev_visual_feature.shape[0] != self.schema.visual_feature_dim:
                raise ValueError(
                    f"prev_visual_feature dim mismatch: expected {self.schema.visual_feature_dim}, "
                    f"got {prev_visual_feature.shape[0]}"
                )
            if prev_visual_feature.device != device:
                raise ValueError(
                    f"prev_visual_feature device {prev_visual_feature.device} != {device}"
                )
            if prev_visual_feature.dtype != dtype:
                raise ValueError(
                    f"prev_visual_feature dtype {prev_visual_feature.dtype} != {dtype}"
                )
            if not torch.isfinite(prev_visual_feature).all():
                raise ValueError("prev_visual_feature contains NaN or Inf")
            visual_delta = visual_feature - prev_visual_feature

        # State delta (first decision = zeros)
        if prev_state is None:
            state_delta = torch.zeros(
                self.schema.state_delta_dim, device=device, dtype=dtype
            )
        else:
            if prev_state.ndim != 1:
                raise ValueError(f"prev_state must be 1D, got shape {prev_state.shape}")
            if prev_state.shape[0] != self.schema.state_dim:
                raise ValueError(
                    f"prev_state dim mismatch: expected {self.schema.state_dim}, "
                    f"got {prev_state.shape[0]}"
                )
            if prev_state.device != device:
                raise ValueError(f"prev_state device {prev_state.device} != {device}")
            if prev_state.dtype != dtype:
                raise ValueError(f"prev_state dtype {prev_state.dtype} != {dtype}")
            if not torch.isfinite(prev_state).all():
                raise ValueError("prev_state contains NaN or Inf")
            state_delta = state - prev_state

        # Concatenate all causal components
        model_input = torch.cat([
            visual_feature,        # [visual_feature_dim]
            state,                 # [8]
            prev_action_summary,   # [14]
            has_previous,          # [1]
            visual_delta,          # [visual_delta_dim]
            state_delta,           # [8]
            event_observable,      # [event_observable_dim]
        ], dim=-1)

        # Audit metadata (not fed to network)
        audit_metadata = {
            "env_id": env_id,
            "episode_id": episode_id,
            "decision_index": decision_index,
            "decision_start_frame": decision_start_frame,
            "prev_executed_start": prev_executed_start,
            "prev_executed_end": prev_executed_end,
            "prev_executed_count": prev_executed_count,
            "end_reason": end_reason,
            "schema_version": "b01r_causal",
            "schema_hash": schema_hash,
        }

        token = CausalToken(
            model_input=model_input,
            audit_metadata=audit_metadata
        )

        # Validate token
        token.validate(self.schema)

        return token

    def forward(self, tokens: list[CausalToken]) -> torch.Tensor:
        """
        Stack tokens into a batch for sequence processing.

        Args:
            tokens: List of CausalToken objects

        Returns:
            Stacked tensor of shape [batch_size, token_dim]
        """
        if not tokens:
            raise ValueError("Cannot encode empty token list")

        # Validate all tokens before stacking
        for token in tokens:
            token.validate(self.schema)

        model_inputs = [token.model_input for token in tokens]
        return torch.stack(model_inputs, dim=0)


def summarize_action_chunk(actions: torch.Tensor) -> torch.Tensor:
    """
    Summarize an executed action chunk into a fixed-size vector.

    Args:
        actions: Executed actions, shape [num_frames, 7]

    Returns:
        Summary vector of shape [14] = [mean_7d, last_7d]

    Raises:
        ValueError: If actions is empty or invalid
    """
    if not isinstance(actions, torch.Tensor):
        raise ValueError(f"actions must be torch.Tensor, got {type(actions)}")

    if actions.ndim != 2:
        raise ValueError(f"actions must be rank 2 [num_frames, 7], got rank {actions.ndim}")

    if actions.shape[-1] != 7:
        raise ValueError(f"actions must be 7D, got {actions.shape[-1]}")

    if actions.shape[0] == 0:
        raise ValueError(
            "Cannot summarize empty action chunk. "
            "prev_executed_count must be > 0 for non-first decisions."
        )

    if not torch.isfinite(actions).all():
        raise ValueError("actions contains NaN or Inf values")

    mean_action = actions.mean(dim=0)  # [7]
    last_action = actions[-1]  # [7]

    return torch.cat([mean_action, last_action], dim=-1)  # [14]
