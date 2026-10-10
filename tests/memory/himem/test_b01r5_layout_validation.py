"""
B01R5 Layout Validation Tests

Tests for strict decision 0 layout constraints and schema validation.
"""

import pytest
import torch
from lerobot.memory.himem.config import TokenSchema, CausalToken, HIMEM_SCHEMA_VERSION


def test_visual_delta_dim_must_match_visual_feature_dim():
    """B01R5: visual_delta_dim must equal visual_feature_dim"""
    with pytest.raises(ValueError, match="visual_delta_dim must equal visual_feature_dim"):
        TokenSchema(visual_feature_dim=256, visual_delta_dim=128)


def test_decision_index_rejects_bool():
    """B01R5: decision_index must reject bool (even though bool is int subclass in Python)"""
    schema = TokenSchema()
    token_input = torch.randn(schema.token_dim)

    metadata = {
        "env_id": "test",
        "episode_id": "ep1",
        "decision_index": False,  # bool, not int
        "decision_start_frame": 0,
        "prev_executed_start": None,
        "prev_executed_end": None,
        "prev_executed_count": None,
        "end_reason": None,
        "schema_version": HIMEM_SCHEMA_VERSION,
        "schema_hash": schema.compute_hash(),
    }

    token = CausalToken(model_input=token_input, audit_metadata=metadata)

    with pytest.raises(ValueError, match="decision_index must be non-negative int.*not bool"):
        token.validate(schema)


def test_timeout_is_valid_end_reason():
    """B01R5: timeout is now a valid end_reason (in addition to truncated)"""
    schema = TokenSchema()
    token_input = torch.randn(schema.token_dim)

    # Set has_previous to exactly 1.0 (required for decision 1)
    offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim
    token_input[offset] = 1.0

    metadata = {
        "env_id": "test",
        "episode_id": "ep1",
        "decision_index": 1,
        "decision_start_frame": 5,
        "prev_executed_start": 0,
        "prev_executed_end": 5,
        "prev_executed_count": 5,
        "end_reason": "timeout",  # Valid in B01R5
        "schema_version": HIMEM_SCHEMA_VERSION,
        "schema_hash": schema.compute_hash(),
    }

    token = CausalToken(model_input=token_input, audit_metadata=metadata)
    token.validate(schema)  # Should not raise


def test_decision_0_must_have_zero_action_summary():
    """B01R5: Decision 0 layout must have zeros in action_summary positions"""
    from lerobot.memory.himem.event_features import EventFeatureEncoder

    schema = TokenSchema()
    encoder = EventFeatureEncoder(schema)

    device = torch.device("cpu")
    dtype = torch.float32

    visual_feature = torch.randn(schema.visual_feature_dim, device=device, dtype=dtype)
    state = torch.randn(schema.state_dim, device=device, dtype=dtype)
    event_observable = torch.randn(schema.event_observable_dim, device=device, dtype=dtype)

    # First decision: all previous inputs None
    token = encoder.encode_token(
        visual_feature=visual_feature,
        state=state,
        prev_action_summary=None,
        prev_visual_feature=None,
        prev_state=None,
        event_observable=event_observable,
        episode_id="ep1",
        env_id="env0",
        decision_index=0,
        decision_start_frame=0,
        prev_executed_start=None,
        prev_executed_end=None,
        prev_executed_count=None,
        end_reason=None,
        schema_hash=schema.compute_hash(),
    )

    # Extract action_summary portion (after visual_feature and state)
    offset = schema.visual_feature_dim + schema.state_dim
    action_summary_slice = token.model_input[offset:offset + schema.action_summary_dim]

    # Must be all zeros for decision 0
    assert torch.allclose(action_summary_slice, torch.zeros_like(action_summary_slice)), \
        f"Decision 0 action_summary must be zeros, got {action_summary_slice}"


def test_decision_0_must_have_zero_has_previous():
    """B01R5: Decision 0 layout must have has_previous=0"""
    from lerobot.memory.himem.event_features import EventFeatureEncoder

    schema = TokenSchema()
    encoder = EventFeatureEncoder(schema)

    device = torch.device("cpu")
    dtype = torch.float32

    visual_feature = torch.randn(schema.visual_feature_dim, device=device, dtype=dtype)
    state = torch.randn(schema.state_dim, device=device, dtype=dtype)
    event_observable = torch.randn(schema.event_observable_dim, device=device, dtype=dtype)

    token = encoder.encode_token(
        visual_feature=visual_feature,
        state=state,
        prev_action_summary=None,
        prev_visual_feature=None,
        prev_state=None,
        event_observable=event_observable,
        episode_id="ep1",
        env_id="env0",
        decision_index=0,
        decision_start_frame=0,
        prev_executed_start=None,
        prev_executed_end=None,
        prev_executed_count=None,
        end_reason=None,
        schema_hash=schema.compute_hash(),
    )

    # Extract has_previous flag (after visual, state, action_summary)
    offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim
    has_previous = token.model_input[offset]

    # Must be 0 for decision 0
    assert has_previous.item() == 0.0, f"Decision 0 has_previous must be 0, got {has_previous.item()}"


def test_decision_0_must_have_zero_deltas():
    """B01R5: Decision 0 layout must have visual_delta=0 and state_delta=0"""
    from lerobot.memory.himem.event_features import EventFeatureEncoder

    schema = TokenSchema()
    encoder = EventFeatureEncoder(schema)

    device = torch.device("cpu")
    dtype = torch.float32

    visual_feature = torch.randn(schema.visual_feature_dim, device=device, dtype=dtype)
    state = torch.randn(schema.state_dim, device=device, dtype=dtype)
    event_observable = torch.randn(schema.event_observable_dim, device=device, dtype=dtype)

    token = encoder.encode_token(
        visual_feature=visual_feature,
        state=state,
        prev_action_summary=None,
        prev_visual_feature=None,
        prev_state=None,
        event_observable=event_observable,
        episode_id="ep1",
        env_id="env0",
        decision_index=0,
        decision_start_frame=0,
        prev_executed_start=None,
        prev_executed_end=None,
        prev_executed_count=None,
        end_reason=None,
        schema_hash=schema.compute_hash(),
    )

    # Extract delta portions
    offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim + 1
    visual_delta = token.model_input[offset:offset + schema.visual_delta_dim]
    state_delta = token.model_input[offset + schema.visual_delta_dim:offset + schema.visual_delta_dim + schema.state_delta_dim]

    # Must be all zeros for decision 0
    assert torch.allclose(visual_delta, torch.zeros_like(visual_delta)), \
        f"Decision 0 visual_delta must be zeros"
    assert torch.allclose(state_delta, torch.zeros_like(state_delta)), \
        f"Decision 0 state_delta must be zeros"


def test_subsequent_decision_must_have_has_previous_one():
    """B01R5: Decision k>0 must have has_previous=1"""
    from lerobot.memory.himem.event_features import EventFeatureEncoder

    schema = TokenSchema()
    encoder = EventFeatureEncoder(schema)

    device = torch.device("cpu")
    dtype = torch.float32

    visual_feature = torch.randn(schema.visual_feature_dim, device=device, dtype=dtype)
    prev_visual_feature = torch.randn(schema.visual_feature_dim, device=device, dtype=dtype)
    state = torch.randn(schema.state_dim, device=device, dtype=dtype)
    prev_state = torch.randn(schema.state_dim, device=device, dtype=dtype)
    prev_action_summary = torch.randn(schema.action_summary_dim, device=device, dtype=dtype)
    event_observable = torch.randn(schema.event_observable_dim, device=device, dtype=dtype)

    token = encoder.encode_token(
        visual_feature=visual_feature,
        state=state,
        prev_action_summary=prev_action_summary,
        prev_visual_feature=prev_visual_feature,
        prev_state=prev_state,
        event_observable=event_observable,
        episode_id="ep1",
        env_id="env0",
        decision_index=1,
        decision_start_frame=5,
        prev_executed_start=0,
        prev_executed_end=5,
        prev_executed_count=5,
        end_reason="queue_depleted",
        schema_hash=schema.compute_hash(),
    )

    # Extract has_previous flag
    offset = schema.visual_feature_dim + schema.state_dim + schema.action_summary_dim
    has_previous = token.model_input[offset]

    # Must be 1 for decision k>0
    assert has_previous.item() == 1.0, f"Decision k>0 has_previous must be 1, got {has_previous.item()}"
