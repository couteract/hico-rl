"""
B01R5P2: Exact Zero Validation Tests

Tests that decision 0 layout validation uses exact zero comparison, not tolerance.
Manager counterexample: 1e-9 values should be rejected.
"""

import pytest
import torch
from lerobot.memory.himem.config import TokenSchema, CausalToken, HIMEM_SCHEMA_VERSION


def test_decision0_action_summary_tiny_nonzero_rejected():
    """Decision 0 with action_summary=1e-9 must be rejected (exact zero required)."""
    schema = TokenSchema()

    token_input = torch.zeros(schema.token_dim)

    # Set action_summary to tiny non-zero value (1e-9)
    offset = schema.visual_feature_dim + schema.state_dim
    token_input[offset:offset + schema.action_summary_dim] = 1e-9

    metadata = {
        "env_id": "test",
        "episode_id": "ep1",
        "decision_index": 0,
        "decision_start_frame": 0,
        "prev_executed_start": None,
        "prev_executed_end": None,
        "prev_executed_count": None,
        "end_reason": None,
        "schema_version": HIMEM_SCHEMA_VERSION,
        "schema_hash": schema.compute_hash(),
    }

    token = CausalToken(model_input=token_input, audit_metadata=metadata)

    with pytest.raises(ValueError, match="Decision 0 must have exact zero action_summary"):
        token.validate(schema)


def test_decision0_visual_delta_tiny_nonzero_rejected():
    """Decision 0 with visual_delta=1e-9 must be rejected (exact zero required)."""
    schema = TokenSchema()

    token_input = torch.zeros(schema.token_dim)

    # Set visual_delta to tiny non-zero value
    offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim + 1
    token_input[offset:offset + schema.visual_delta_dim] = 1e-9

    metadata = {
        "env_id": "test",
        "episode_id": "ep1",
        "decision_index": 0,
        "decision_start_frame": 0,
        "prev_executed_start": None,
        "prev_executed_end": None,
        "prev_executed_count": None,
        "end_reason": None,
        "schema_version": HIMEM_SCHEMA_VERSION,
        "schema_hash": schema.compute_hash(),
    }

    token = CausalToken(model_input=token_input, audit_metadata=metadata)

    with pytest.raises(ValueError, match="Decision 0 must have exact zero visual_delta"):
        token.validate(schema)


def test_decision0_state_delta_tiny_nonzero_rejected():
    """Decision 0 with state_delta=1e-9 must be rejected (exact zero required)."""
    schema = TokenSchema()

    token_input = torch.zeros(schema.token_dim)

    # Set state_delta to tiny non-zero value
    offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim + 1 + schema.visual_delta_dim
    token_input[offset:offset + schema.state_delta_dim] = 1e-9

    metadata = {
        "env_id": "test",
        "episode_id": "ep1",
        "decision_index": 0,
        "decision_start_frame": 0,
        "prev_executed_start": None,
        "prev_executed_end": None,
        "prev_executed_count": None,
        "end_reason": None,
        "schema_version": HIMEM_SCHEMA_VERSION,
        "schema_hash": schema.compute_hash(),
    }

    token = CausalToken(model_input=token_input, audit_metadata=metadata)

    with pytest.raises(ValueError, match="Decision 0 must have exact zero state_delta"):
        token.validate(schema)
