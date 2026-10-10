"""B02Q29R3 prior action-schema contract.

This contract deliberately separates the historical environment-submitted action
from the bounded zero-residual actor base used by Performance V2.
"""
from __future__ import annotations

import numpy as np


ATOL = 2e-6
RTOL = 0.0


def as_action(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("ACTION_SCHEMA")
    return result


def bounded_base(raw_base7):
    raw = as_action(raw_base7)
    return np.concatenate((np.clip(raw[:6], -1.0, 1.0), raw[6:])).astype(np.float32)


def support(raw_base7, behavior_submitted_action7):
    raw = as_action(raw_base7)
    behavior = as_action(behavior_submitted_action7)
    bounded = bounded_base(raw)
    delta = np.abs(behavior[:6] - bounded[:6]).astype(np.float32)
    return bool(np.all(np.abs(behavior[:6]) <= 1.0)), float(delta.max(initial=0.0))


def make_row(raw_base7, behavior_submitted_action7):
    raw = as_action(raw_base7)
    behavior = as_action(behavior_submitted_action7)
    bounded = bounded_base(raw)
    inside, delta = support(raw, behavior)
    return {
        "raw_base7": raw,
        "behavior_submitted_action7": behavior,
        "submitted_action7": behavior.copy(),
        "bounded_base7": bounded,
        "v2_zero_residual_action7": bounded.copy(),
        "behavior_arm_in_v2_actor_support": inside,
        "behavior_arm_clip_delta": np.float32(delta),
        "critic_replay_action_semantics": "ACTUAL_BEHAVIOR_SUBMITTED_ACTION7",
        "latent_u": None,
        "behavior_logprob": None,
        "latent_u_valid": False,
        "behavior_logprob_valid": False,
    }


def validate(row):
    required = {
        "raw_base7", "behavior_submitted_action7", "submitted_action7",
        "bounded_base7", "v2_zero_residual_action7",
        "behavior_arm_in_v2_actor_support", "behavior_arm_clip_delta",
        "critic_replay_action_semantics", "latent_u", "behavior_logprob",
        "latent_u_valid", "behavior_logprob_valid",
    }
    if not required.issubset(row):
        raise ValueError("PRIOR_ACTION_SCHEMA_MISSING")
    expected = make_row(row["raw_base7"], row["behavior_submitted_action7"])
    for key in ("raw_base7", "behavior_submitted_action7", "submitted_action7", "bounded_base7", "v2_zero_residual_action7"):
        actual = as_action(row[key])
        if not np.array_equal(actual, expected[key]):
            raise ValueError("PRIOR_ACTION_RELATION_" + key.upper())
    if row["behavior_arm_in_v2_actor_support"] is not expected["behavior_arm_in_v2_actor_support"]:
        raise ValueError("ACTION_SUPPORT_FLAG")
    if not np.isclose(float(row["behavior_arm_clip_delta"]), float(expected["behavior_arm_clip_delta"]), atol=ATOL, rtol=RTOL):
        raise ValueError("ACTION_SUPPORT_DELTA")
    if row["critic_replay_action_semantics"] != expected["critic_replay_action_semantics"]:
        raise ValueError("CRITIC_ACTION_SEMANTICS")
    if row["latent_u"] is not None or row["behavior_logprob"] is not None or row["latent_u_valid"] or row["behavior_logprob_valid"]:
        raise ValueError("BEHAVIOR_LATENT_NOT_IDENTIFIABLE")
    return True


def queue_action_matches(planned_postprocessed, later_behavior):
    planned = as_action(planned_postprocessed)
    later = as_action(later_behavior)
    if not np.array_equal(planned, later):
        raise ValueError("QUEUE_ENV_POST_ACTION_MISMATCH")
    return True
