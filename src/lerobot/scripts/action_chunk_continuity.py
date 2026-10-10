"""Opt-in, arm-only continuity blend for the first frames of a new action chunk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContinuityResult:
    action: np.ndarray
    blended: bool
    blend_weight: float


def policy_queue_is_empty(policy: Any) -> bool:
    """Read the policy's real action queue before inference to identify refill."""
    return len(getattr(policy, "_queues", {}).get("action", ())) == 0


def policy_queue_cursor(policy: Any, n_action_steps: int) -> int:
    """Return the consumed cursor after select_action has popped its output."""
    queued = len(getattr(policy, "_queues", {}).get("action", ()))
    return max(0, int(n_action_steps) - queued - 1)


def should_stop_continuity_pilot(enabled: bool, intervention_active: bool) -> bool:
    """A continuity pilot ends after its first safety takeover frame."""
    return bool(enabled and intervention_active)


def native_domain_distance(action: np.ndarray) -> np.ndarray:
    """Elementwise distance to SO follower native domains for use_degrees=false."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (6,):
        raise ValueError(f"Expected a 6D SO-101 command, got {value.shape}.")
    lower = np.asarray([-100.0] * 5 + [0.0])
    upper = np.asarray([100.0] * 6)
    return np.maximum(lower - value, 0.0) + np.maximum(value - upper, 0.0)


def computed_bus_bounded_native_action(action: np.ndarray) -> np.ndarray:
    """Predict the `_unnormalize` range clamp; this is not a bus readback."""
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError(f"Expected a finite 6D SO-101 command, got {value.shape}.")
    return np.clip(value, np.asarray([-100.0] * 5 + [0.0]), np.asarray([100.0] * 6))


def continuity_record_fields(
    *,
    raw_policy_action: np.ndarray | None,
    continuity_intended_action: np.ndarray,
    controller_input_action: np.ndarray,
    robot_returned_goal_action: np.ndarray,
    follower_feedback_state_pre_action: np.ndarray,
    queue_cursor: int,
    queue_refill: bool,
    blend_applied: bool,
    blend_alpha: float,
    inference_latency_s: float,
) -> dict[str, np.ndarray]:
    """Build distinct, validated 6D recording fields for one control frame."""
    values = {
        "complementary_info.continuity_intended_action": continuity_intended_action,
        "complementary_info.controller_input_action": controller_input_action,
        "complementary_info.robot_returned_goal_action": robot_returned_goal_action,
        "complementary_info.follower_feedback_state": follower_feedback_state_pre_action,
    }
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite 6D SO-101 vector; got {vector.shape}.")
        result[name] = vector.copy()

    raw_valid = raw_policy_action is not None
    raw = np.zeros(6, dtype=np.float32) if not raw_valid else np.asarray(raw_policy_action, dtype=np.float32)
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError(f"Raw policy action must be a finite 6D SO-101 vector; got {raw.shape}.")
    result["complementary_info.policy_action"] = raw.copy()
    result["complementary_info.policy_action_valid"] = np.asarray([raw_valid], dtype=np.int64)
    result["complementary_info.action_queue_cursor"] = np.asarray([queue_cursor], dtype=np.int64)
    result["complementary_info.action_chunk_boundary"] = np.asarray([queue_refill], dtype=np.int64)
    result["complementary_info.boundary_continuity_applied"] = np.asarray([blend_applied], dtype=np.int64)
    result["complementary_info.continuity_blend_weight"] = np.asarray([blend_alpha], dtype=np.float32)
    result["complementary_info.policy_inference_latency_s"] = np.asarray([inference_latency_s], dtype=np.float32)
    return result


class ActionChunkContinuity:
    """Blend only the first three arm targets after a policy action-queue refill.

    The caller must call :meth:`note_sent` with the actual command returned by
    ``robot.send_action``. Gripper index 5 is intentionally always passed raw.
    """

    WEIGHTS = (1.0 / 3.0, 2.0 / 3.0, 1.0)

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self._previous_sent: np.ndarray | None = None
        self._blend_cursor: int | None = None

    def note_sent(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (6,):
            raise ValueError(f"Expected a 6D SO-101 command, got {command.shape}.")
        if not np.isfinite(command).all():
            raise ValueError("Sent SO-101 command must be finite.")
        self._previous_sent = command.copy()

    def process(self, raw_action: np.ndarray, *, chunk_boundary: bool) -> ContinuityResult:
        raw = np.asarray(raw_action, dtype=np.float64)
        if raw.shape != (6,):
            raise ValueError(f"Expected a 6D SO-101 action, got {raw.shape}.")
        if not np.isfinite(raw).all():
            raise ValueError("Raw SO-101 action must be finite.")

        if chunk_boundary:
            self._blend_cursor = 0 if self.enabled and self._previous_sent is not None else None

        if not self.enabled or self._blend_cursor is None:
            return ContinuityResult(raw.copy(), False, 0.0)

        cursor = self._blend_cursor
        if cursor >= len(self.WEIGHTS):
            self._blend_cursor = None
            return ContinuityResult(raw.copy(), False, 0.0)

        alpha = self.WEIGHTS[cursor]
        # The previous *sent* command is the recursive state. The gripper stays raw.
        assert self._previous_sent is not None
        filtered = raw.copy()
        if alpha == 1.0:
            filtered[:5] = raw[:5]
        else:
            filtered[:5] = self._previous_sent[:5] + alpha * (raw[:5] - self._previous_sent[:5])
        self._blend_cursor += 1
        return ContinuityResult(filtered, True, alpha)
