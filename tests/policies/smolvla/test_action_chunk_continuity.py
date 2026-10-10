import numpy as np
import pytest

from lerobot.scripts.action_chunk_continuity import (
    ActionChunkContinuity,
    computed_bus_bounded_native_action,
    continuity_record_fields,
    native_domain_distance,
    policy_queue_cursor,
    policy_queue_is_empty,
    should_stop_continuity_pilot,
)


def test_disabled_is_exact_parity():
    raw = np.array([1.25, -2.5, 3.75, 4.0, -5.0, 61.0])
    controller = ActionChunkContinuity(enabled=False)
    controller.note_sent(np.zeros(6))
    result = controller.process(raw, chunk_boundary=True)
    np.testing.assert_array_equal(result.action, raw)
    assert not result.blended
    assert result.blend_weight == 0.0


def test_first_chunk_is_not_blended():
    controller = ActionChunkContinuity(enabled=True)
    raw = np.arange(6, dtype=float)
    result = controller.process(raw, chunk_boundary=True)
    np.testing.assert_array_equal(result.action, raw)
    assert not result.blended


def test_refill_blends_three_arm_frames_and_keeps_gripper_raw():
    controller = ActionChunkContinuity(enabled=True)
    previous = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    controller.note_sent(previous)
    raw = np.array([30, 40, 50, 60, 70, 80], dtype=float)

    first = controller.process(raw, chunk_boundary=True)
    np.testing.assert_allclose(first.action, [10, 20, 30, 40, 50, 80])
    assert first.blended and first.blend_weight == pytest.approx(1 / 3)
    controller.note_sent(first.action)

    second_raw = raw + 3
    second = controller.process(second_raw, chunk_boundary=False)
    np.testing.assert_allclose(second.action[:5], first.action[:5] + (2 / 3) * (second_raw[:5] - first.action[:5]))
    assert second.action[5] == second_raw[5]
    assert second.blend_weight == pytest.approx(2 / 3)
    controller.note_sent(second.action)

    third_raw = raw + 6
    third = controller.process(third_raw, chunk_boundary=False)
    np.testing.assert_array_equal(third.action, third_raw)
    assert third.blended and third.blend_weight == 1.0
    controller.note_sent(third.action)

    fourth_raw = raw + 9
    fourth = controller.process(fourth_raw, chunk_boundary=False)
    np.testing.assert_array_equal(fourth.action, fourth_raw)
    assert not fourth.blended


def test_queue_refill_starts_a_new_blend_and_reset_clears_state():
    controller = ActionChunkContinuity(enabled=True)
    raw = np.ones(6)
    controller.note_sent(np.zeros(6))
    assert controller.process(raw, chunk_boundary=True).blended
    for _ in range(9):
        controller.note_sent(raw)
        controller.process(raw, chunk_boundary=False)
    controller.note_sent(np.full(6, 2.0))
    refill = controller.process(np.full(6, 3.0), chunk_boundary=True)
    assert refill.blended
    np.testing.assert_allclose(refill.action[:5], 2.0 + 1 / 3)
    assert refill.action[5] == 3.0

    controller.reset()
    after_reset = controller.process(raw, chunk_boundary=True)
    np.testing.assert_array_equal(after_reset.action, raw)
    assert not after_reset.blended


@pytest.mark.parametrize("raw", [np.zeros(5), np.zeros((1, 6))])
def test_rejects_wrong_shapes(raw):
    with pytest.raises(ValueError):
        ActionChunkContinuity(enabled=True).process(raw, chunk_boundary=True)


def test_rejects_nonfinite_values():
    raw = np.zeros(6)
    raw[2] = np.nan
    with pytest.raises(ValueError):
        ActionChunkContinuity(enabled=True).process(raw, chunk_boundary=True)


def test_policy_queue_refill_and_cursor_use_live_queue_state():
    class Policy:
        _queues = {"action": []}

    policy = Policy()
    assert policy_queue_is_empty(policy)
    policy._queues["action"].extend([object() for _ in range(9)])
    assert not policy_queue_is_empty(policy)
    assert policy_queue_cursor(policy, 10) == 0
    policy._queues["action"].pop(0)
    assert policy_queue_cursor(policy, 10) == 1


def test_record_fields_preserve_raw_intended_controller_returned_and_pre_action_state():
    raw = np.arange(6, dtype=np.float32)
    intended = raw + 10
    controller_input = raw + 20
    returned_goal = raw + 30
    feedback = raw + 40
    fields = continuity_record_fields(
        raw_policy_action=raw,
        continuity_intended_action=intended,
        controller_input_action=controller_input,
        robot_returned_goal_action=returned_goal,
        follower_feedback_state_pre_action=feedback,
        queue_cursor=0,
        queue_refill=True,
        blend_applied=True,
        blend_alpha=1 / 3,
        inference_latency_s=0.01,
    )
    np.testing.assert_array_equal(fields["complementary_info.policy_action"], raw)
    np.testing.assert_array_equal(fields["complementary_info.continuity_intended_action"], intended)
    np.testing.assert_array_equal(fields["complementary_info.controller_input_action"], controller_input)
    np.testing.assert_array_equal(fields["complementary_info.robot_returned_goal_action"], returned_goal)
    np.testing.assert_array_equal(fields["complementary_info.follower_feedback_state"], feedback)
    assert fields["complementary_info.policy_action_valid"].item() == 1
    assert len({id(fields[key]) for key in (
        "complementary_info.policy_action",
        "complementary_info.continuity_intended_action",
        "complementary_info.controller_input_action",
        "complementary_info.robot_returned_goal_action",
        "complementary_info.follower_feedback_state",
    )}) == 5


def test_record_fields_mark_policy_raw_missing_during_takeover():
    zeros = np.zeros(6, dtype=np.float32)
    fields = continuity_record_fields(
        raw_policy_action=None,
        continuity_intended_action=zeros,
        controller_input_action=zeros,
        robot_returned_goal_action=zeros,
        follower_feedback_state_pre_action=zeros,
        queue_cursor=-1,
        queue_refill=False,
        blend_applied=False,
        blend_alpha=0.0,
        inference_latency_s=0.0,
    )
    assert fields["complementary_info.policy_action_valid"].item() == 0


def test_native_range_distance_and_computed_bus_clamp_contract():
    raw = np.asarray([100.2, -101.0, 0.0, 50.0, 0.0, -10.0])
    computed = computed_bus_bounded_native_action(raw)
    np.testing.assert_array_equal(computed, [100.0, -100.0, 0.0, 50.0, 0.0, 0.0])
    assert np.all(native_domain_distance(computed) == 0)
    assert np.all(native_domain_distance(computed) <= native_domain_distance(raw))


def test_safety_takeover_ends_continuity_pilot_without_policy_release():
    assert should_stop_continuity_pilot(True, True)
    assert not should_stop_continuity_pilot(True, False)
    assert not should_stop_continuity_pilot(False, True)
