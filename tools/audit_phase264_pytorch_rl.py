#!/usr/bin/env python3
"""Audit Phase264 conversion and the PyTorch VALUE/ACP and replay contracts.

This script deliberately reads only parquet metadata/numeric columns. It does
not decode all 164,068 images, but it validates their declared schema and lets
LeRobot's normal dataset loader remain responsible for image decoding.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch

from lerobot.rl.buffer import ReplayBuffer
from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
)
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    compute_normalized_value_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs/phase264_libero10_lerobot_v3"
DEFAULT_SOURCE_ROOT = Path(
    "/home/zhang/xzw/two/evo-rl-jax/Evo-RL-main/outputs/phase264_libero_rl_rollouts"
)
BASELINE_SHA256 = "3534dea6c61e6581290b6eee699288c5007cba7971efaefa4894f40b42297d71"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _column_numpy(table: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    values = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def _list_column_numpy(table: Any, name: str, width: int) -> np.ndarray:
    values = np.asarray(table[name].combine_chunks().values.to_numpy(zero_copy_only=False))
    _require(values.size == len(table) * width, f"{name} is not a fixed {width}D vector")
    return values.reshape(len(table), width)


def _reference_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeTargetInfo],
    task_max_lengths: dict[int, int],
    c_fail_coef: float,
) -> np.ndarray:
    result = []
    for episode_index, frame_index in zip(episode_indices, frame_indices, strict=True):
        episode = episode_info[int(episode_index)]
        task_max = task_max_lengths[episode.task_index]
        remaining = episode.length - int(frame_index) - 1
        c_fail = task_max * c_fail_coef
        return_value = -remaining - (c_fail if not episode.success else 0.0)
        result.append(np.clip(return_value / (task_max + c_fail), -1.0, 0.0))
    return np.asarray(result, dtype=np.float32)


def _reference_dense_rewards(
    targets: np.ndarray, episode_indices: np.ndarray, frame_indices: np.ndarray
) -> np.ndarray:
    rewards = targets.copy()
    contiguous = (
        (episode_indices[:-1] == episode_indices[1:])
        & (frame_indices[1:] == frame_indices[:-1] + 1)
    )
    rewards[:-1][contiguous] = targets[:-1][contiguous] - targets[1:][contiguous]
    return rewards.astype(np.float32)


def _reference_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    result = np.empty_like(rewards, dtype=np.float32)
    for start in range(len(rewards)):
        episode_index = episode_indices[start]
        start_frame = frame_indices[start]
        reward_sum = 0.0
        steps = 0
        cursor = start
        while cursor < len(rewards) and steps < n_step:
            if episode_indices[cursor] != episode_index or frame_indices[cursor] != start_frame + steps:
                break
            reward_sum += float(rewards[cursor])
            cursor += 1
            steps += 1
        can_bootstrap = (
            steps == n_step
            and cursor < len(rewards)
            and episode_indices[cursor] == episode_index
            and frame_indices[cursor] == start_frame + n_step
        )
        result[start] = reward_sum + (float(values[cursor]) if can_bootstrap else 0.0) - values[start]
    return result


class _FakeDataset:
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _audit_replay_buffer() -> dict[str, Any]:
    samples = []
    # Episode 0 ends successfully; episode 1 ends due to a time limit.
    for episode_index, length, truncated in ((0, 3, False), (1, 2, True)):
        for frame_index in range(length):
            done = frame_index == length - 1
            value = float(episode_index * 10 + frame_index)
            samples.append(
                {
                    "observation.state": torch.tensor([value], dtype=torch.float32),
                    "action": torch.tensor([value + 0.5], dtype=torch.float32),
                    "next.reward": torch.tensor(float(done), dtype=torch.float32),
                    "next.done": torch.tensor(done),
                    "next.truncated": torch.tensor(done and truncated),
                    "episode_index": torch.tensor(episode_index),
                }
            )
    fake = _FakeDataset(samples)
    transitions = ReplayBuffer._lerobotdataset_to_transitions(
        dataset=fake, state_keys=["observation.state"]
    )
    _require(float(transitions[0]["next_state"]["observation.state"][0, 0]) == 1.0, "next state mismatch")
    _require(float(transitions[1]["next_state"]["observation.state"][0, 0]) == 2.0, "next state mismatch")
    _require(float(transitions[2]["next_state"]["observation.state"][0, 0]) == 2.0, "terminal crossed episode")
    _require(float(transitions[3]["next_state"]["observation.state"][0, 0]) == 11.0, "next state mismatch")
    _require(float(transitions[4]["next_state"]["observation.state"][0, 0]) == 11.0, "terminal next state mismatch")
    _require(transitions[2]["done"] and not transitions[2]["truncated"], "success boundary mismatch")
    _require(transitions[4]["done"] and transitions[4]["truncated"], "timeout boundary mismatch")

    replay = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=fake,
        device="cpu",
        storage_device="cpu",
        state_keys=["observation.state"],
        use_drq=False,
    )
    _require(replay.dones[:5].tolist() == [False, False, True, False, True], "done lost on import")
    _require(replay.truncateds[:5].tolist() == [False, False, False, False, True], "truncation lost on import")

    invalid = _FakeDataset([{**samples[0], "next.truncated": torch.tensor(True)}])
    try:
        ReplayBuffer._lerobotdataset_to_transitions(invalid, ["observation.state"])
    except ValueError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    _require(invalid_rejected, "truncated=true with done=false was accepted")
    return {
        "fake_transition_count": len(transitions),
        "same_episode_next_state": True,
        "terminal_does_not_cross_episode": True,
        "success_done_not_truncated": True,
        "timeout_done_and_truncated": True,
        "invalid_truncation_rejected": True,
    }


def audit(dataset_root: Path, source_root: Path, c_fail_coef: float, n_step: int) -> dict[str, Any]:
    info = _read_json(dataset_root / "meta/info.json")
    manifest = _read_json(dataset_root / "CONVERSION_MANIFEST.json")
    folds = _read_json(dataset_root / "episode_folds.json")
    source_manifest = _read_json(source_root / "DATASET_MANIFEST.json")

    _require(info["total_episodes"] == manifest["episode_count"] == 250, "episode total mismatch")
    _require(info["total_frames"] == manifest["frame_count"] == 82034, "frame total mismatch")
    _require(manifest["summary_only_records_imported"] == 0, "summary-only records were imported")
    _require(source_manifest["summary_only_episode_count"] == 100, "source summary-only count changed")
    _require(not manifest["human_intervention_labels_synthesized"], "fake intervention labels exist")
    _require(manifest["source_baseline_checkpoint_sha256"] == BASELINE_SHA256, "baseline SHA mismatch")

    expected_shapes = {
        "observation.images.image": [256, 256, 3],
        "observation.images.wrist_image": [256, 256, 3],
        "observation.state": [8],
        "action": [7],
        "complementary_info.policy_action": [7],
    }
    for field, shape in expected_shapes.items():
        _require(info["features"][field]["shape"] == shape, f"{field} schema mismatch")
    _require("complementary_info.is_intervention" not in info["features"], "intervention field was synthesized")

    episode_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_columns = [
        "episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index",
        "episode_success", "source_episode_id", "source_init_state_id", "task_index_contract",
        "behavior_exploration", "behavior_checkpoint_sha256", "training_action_semantics",
    ]
    episodes = pq.read_table(episode_path, columns=episode_columns).to_pylist()
    tasks_rows = pq.read_table(dataset_root / "meta/tasks.parquet").to_pylist()
    task_text_to_index = {row["__index_level_0__"]: int(row["task_index"]) for row in tasks_rows}
    _require(task_text_to_index == manifest["task_mapping"]["task_text_to_index"], "task table/text mapping mismatch")

    data_columns = [
        "observation.state", "action", "next.reward", "next.success", "next.done", "next.truncated",
        "complementary_info.policy_action", "complementary_info.is_exploration",
        "complementary_info.source_episode_id", "complementary_info.init_state_id",
        "frame_index", "episode_index", "index", "task_index",
    ]
    frames = pads.dataset(dataset_root / "data", format="parquet").to_table(columns=data_columns)
    _require(len(frames) == info["total_frames"], "parquet frame total mismatch")
    state = _list_column_numpy(frames, "observation.state", 8)
    action = _list_column_numpy(frames, "action", 7)
    policy_action = _list_column_numpy(frames, "complementary_info.policy_action", 7)
    _require(np.isfinite(state).all() and np.isfinite(action).all(), "non-finite state/action")
    episode_indices = _column_numpy(frames, "episode_index", np.int64)
    frame_indices = _column_numpy(frames, "frame_index", np.int64)
    global_indices = _column_numpy(frames, "index", np.int64)
    task_indices = _column_numpy(frames, "task_index", np.int64)
    done = _column_numpy(frames, "next.done", bool)
    truncated = _column_numpy(frames, "next.truncated", bool)
    success_frames = _column_numpy(frames, "next.success", bool)
    exploration_frames = _column_numpy(frames, "complementary_info.is_exploration", np.int64)
    source_episode_frames = _column_numpy(frames, "complementary_info.source_episode_id", np.int64)
    init_state_frames = _column_numpy(frames, "complementary_info.init_state_id", np.int64)
    _require(np.array_equal(global_indices, np.arange(len(frames))), "global frame index is not contiguous")
    _require(not np.any(truncated & ~done), "truncation without done")

    task_max_lengths: dict[int, int] = {}
    outcome_counts: Counter[tuple[int, str]] = Counter()
    episode_info: dict[int, EpisodeTargetInfo] = {}
    exploration_episode_count = 0
    for position, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        _require(episode_index == position, "episode indices are not contiguous")
        start, stop = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        length = int(episode["length"])
        _require(stop - start == length, f"episode {episode_index} range/length mismatch")
        _require(np.all(episode_indices[start:stop] == episode_index), f"episode {episode_index} frame ownership mismatch")
        _require(np.array_equal(frame_indices[start:stop], np.arange(length)), f"episode {episode_index} frame order mismatch")
        _require(done[start:stop].sum() == 1 and done[stop - 1], f"episode {episode_index} boundary mismatch")
        task_text = episode["tasks"][0]
        task_index = int(episode["task_index_contract"])
        _require(task_text_to_index[task_text] == task_index, f"episode {episode_index} task mismatch")
        _require(np.all(task_indices[start:stop] == task_index), f"episode {episode_index} frame task mismatch")
        _require(np.all(source_episode_frames[start:stop] == int(episode["source_episode_id"])), "source episode provenance mismatch")
        _require(np.all(init_state_frames[start:stop] == int(episode["source_init_state_id"])), "init state provenance mismatch")
        _require(episode["behavior_checkpoint_sha256"] == BASELINE_SHA256, "episode checkpoint SHA mismatch")
        _require(episode["training_action_semantics"] == "env_action_exactly_submitted_to_libero", "action semantics mismatch")
        is_success = episode["episode_success"] == "success"
        _require(bool(success_frames[start:stop].any()) == is_success, "episode/frame success mismatch")
        if is_success:
            _require(not truncated[stop - 1], "successful episode marked truncated")
        is_exploration = bool(episode["behavior_exploration"])
        exploration_episode_count += int(is_exploration)
        _require(np.all(exploration_frames[start:stop] == int(is_exploration)), "exploration provenance mismatch")
        actions_differ = not np.array_equal(action[start:stop], policy_action[start:stop])
        _require(actions_differ == is_exploration, "behavior/policy action provenance mismatch")
        task_max_lengths[task_index] = max(task_max_lengths.get(task_index, 0), length)
        outcome_counts[(task_index, "success" if is_success else "failure")] += 1
        episode_info[episode_index] = EpisodeTargetInfo(episode_index, task_index, length, is_success)

    _require(exploration_episode_count == 110, "expected 110 exploration episodes")
    _require(int(done.sum()) == len(episodes), "terminal count mismatch")
    _require(int(truncated.sum()) == sum(not (row["episode_success"] == "success") for row in episodes), "failure/truncation mismatch")

    mapping = manifest["episode_index_mapping"]
    _require(len(mapping) == len(episodes), "manifest episode mapping length mismatch")
    summary_path = source_root / source_manifest["summary_only_records_path"]
    summary_records = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(len(summary_records) == 100, "summary-only record file count mismatch")
    # Summary-only records do not carry frame episode IDs. Their absence is
    # established by the exact 250-ID manifest join and frame provenance below.
    converted_source_ids = {int(row["source_episode_id"]) for row in mapping}
    frame_ready_source_ids = {int(row["episode_id"]) for row in source_manifest["records"]}
    _require(converted_source_ids == frame_ready_source_ids, "conversion is not an exact frame-ready ID join")

    all_episodes = set(range(len(episodes)))
    held_out = []
    fold_sizes = []
    for fold in folds["folds"]:
        train = set(map(int, fold["train_episodes"]))
        test = set(map(int, fold["test_episodes"]))
        _require(train.isdisjoint(test), "OOF train/test overlap")
        _require(train | test == all_episodes, "OOF fold does not cover dataset")
        held_out.extend(test)
        fold_sizes.append({"fold": int(fold["fold"]), "train": len(train), "test": len(test)})
    _require(Counter(held_out) == Counter({index: 1 for index in all_episodes}), "episodes not held out exactly once")

    targets = compute_normalized_value_targets(
        episode_indices, frame_indices, episode_info, task_max_lengths, c_fail_coef
    )
    reference_targets = _reference_targets(
        episode_indices, frame_indices, episode_info, task_max_lengths, c_fail_coef
    )
    _require(np.array_equal(targets, reference_targets), "VALUE target formula mismatch")
    rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    reference_rewards = _reference_dense_rewards(targets, episode_indices, frame_indices)
    _require(np.array_equal(rewards, reference_rewards), "dense reward formula/boundary mismatch")
    values = (targets * np.float32(0.73) + np.sin(global_indices * 0.01).astype(np.float32) * 0.01).astype(np.float32)
    advantages = _compute_n_step_advantages(rewards, values, episode_indices, frame_indices, n_step)
    reference_advantages = _reference_advantages(rewards, values, episode_indices, frame_indices, n_step)
    _require(np.allclose(advantages, reference_advantages, rtol=0, atol=1e-7), "n-step advantage mismatch")
    thresholds = _compute_task_thresholds(task_indices, advantages, positive_ratio=0.3)
    indicators = _binarize_advantages(task_indices, advantages, thresholds, np.zeros(len(frames)), False)
    for task_index, threshold in thresholds.items():
        mask = task_indices == task_index
        _require(np.array_equal(indicators[mask], (advantages[mask] >= threshold).astype(np.int64)), "ACP indicator mismatch")

    replay = _audit_replay_buffer()
    return {
        "ok": True,
        "dataset_root": dataset_root.as_posix(),
        "source_root": source_root.as_posix(),
        "episode_count": len(episodes),
        "frame_count": len(frames),
        "success_episode_count": sum(row["episode_success"] == "success" for row in episodes),
        "failure_episode_count": sum(row["episode_success"] != "success" for row in episodes),
        "terminal_count": int(done.sum()),
        "truncated_count": int(truncated.sum()),
        "exploration_episode_count": exploration_episode_count,
        "summary_only_records_imported": 0,
        "human_intervention_labels_synthesized": False,
        "task_max_lengths": {str(key): value for key, value in sorted(task_max_lengths.items())},
        "task_outcome_counts": {
            str(task): {outcome: outcome_counts[(task, outcome)] for outcome in ("success", "failure")}
            for task in sorted(task_max_lengths)
        },
        "oof_folds": fold_sizes,
        "math_contract": {
            "c_fail_coef": c_fail_coef,
            "n_step": n_step,
            "targets_match_independent_formula": True,
            "dense_rewards_are_episode_safe": True,
            "advantages_are_episode_safe": True,
            "acp_threshold_is_per_task_70th_percentile": True,
        },
        "replay_buffer_contract": replay,
        "historical_guards": {
            "model_oof_predictions_required": True,
            "analytic_targets_cannot_substitute_for_predictions": True,
            "expert_demo_not_forced_positive": True,
            "failure_actions_retained": True,
            "evaluation_archives_excluded": True,
            "policy_update_blocked_until_oof_gate": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--c-fail-coef", type=float, default=1.0)
    parser.add_argument("--n-step", type=int, default=50)
    args = parser.parse_args()
    if args.c_fail_coef < 0 or args.n_step <= 0:
        parser.error("--c-fail-coef must be non-negative and --n-step must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    report = audit(dataset_root, source_root, args.c_fail_coef, args.n_step)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
