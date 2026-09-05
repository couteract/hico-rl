#!/usr/bin/env python3
"""Finalize globally thresholded ACP labels from complete OOF VALUE predictions.

Run this only after all folds have written both the VALUE prediction and its
``value_oof_fold`` provenance field. The script rejects incomplete or wrongly
tagged OOF coverage before modifying the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
    _write_columns_in_place,
)
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    compute_normalized_value_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs/phase264_libero10_lerobot_v3"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _numpy(table: Any, field: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[field].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def finalize(
    dataset_root: Path,
    folds_path: Path,
    value_field: str,
    fold_field: str,
    advantage_field: str,
    indicator_field: str,
    intervention_field: str,
    n_step: int,
    positive_ratio: float,
    c_fail_coef: float,
    dry_run: bool,
) -> dict[str, Any]:
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    episode_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(
        episode_path,
        columns=["episode_index", "length", "episode_success", "task_index_contract"],
    ).to_pylist()
    expected_fold_by_episode: dict[int, int] = {}
    for fold in folds["folds"]:
        fold_index = int(fold["fold"])
        for episode_index in map(int, fold["test_episodes"]):
            _require(episode_index not in expected_fold_by_episode, "episode appears in multiple held-out folds")
            expected_fold_by_episode[episode_index] = fold_index
    _require(len(expected_fold_by_episode) == len(episodes), "held-out folds do not cover every episode")

    columns = ["index", "episode_index", "frame_index", "task_index", value_field, fold_field]
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    has_intervention = intervention_field in info["features"]
    if has_intervention:
        columns.append(intervention_field)
    parquet_dataset = pads.dataset(dataset_root / "data", format="parquet")
    missing = sorted(set(columns) - set(parquet_dataset.schema.names))
    if missing:
        raise RuntimeError(
            "OOF finalization is blocked because fold inference has not written required fields: "
            f"{missing}"
        )
    frames = parquet_dataset.to_table(columns=columns)
    absolute_indices = _numpy(frames, "index", np.int64)
    episode_indices = _numpy(frames, "episode_index", np.int64)
    frame_indices = _numpy(frames, "frame_index", np.int64)
    task_indices = _numpy(frames, "task_index", np.int64)
    values = _numpy(frames, value_field, np.float32)
    observed_folds = _numpy(frames, fold_field, np.int64)
    _require(np.isfinite(values).all(), "OOF VALUE predictions are missing or non-finite")
    expected_folds = np.asarray(
        [expected_fold_by_episode[int(episode)] for episode in episode_indices], dtype=np.int64
    )
    mismatches = int(np.sum(observed_folds != expected_folds))
    _require(mismatches == 0, f"{mismatches} frames have the wrong/missing OOF fold provenance")

    episode_info: dict[int, EpisodeTargetInfo] = {}
    task_max_lengths: dict[int, int] = {}
    for row in episodes:
        episode_index = int(row["episode_index"])
        task_index = int(row["task_index_contract"])
        length = int(row["length"])
        success = row["episode_success"] == "success"
        episode_info[episode_index] = EpisodeTargetInfo(episode_index, task_index, length, success)
        task_max_lengths[task_index] = max(task_max_lengths.get(task_index, 0), length)

    targets = compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_lengths,
        c_fail_coef=c_fail_coef,
    )
    rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    advantages = _compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=n_step,
    )
    thresholds = _compute_task_thresholds(task_indices, advantages, positive_ratio)
    interventions = (
        _numpy(frames, intervention_field, np.float32)
        if has_intervention
        else np.zeros(len(frames), dtype=np.float32)
    )
    # LIBERO has no genuine operator takeover. Existing real-robot datasets may
    # retain intervention provenance, but it is not forced positive here.
    indicators = _binarize_advantages(
        task_indices, advantages, thresholds, interventions, force_intervention_positive=False
    )

    report = {
        "ok": True,
        "dry_run": dry_run,
        "dataset_root": dataset_root.as_posix(),
        "frame_count": len(frames),
        "episode_count": len(episodes),
        "oof_fold_counts": {
            str(fold): int(np.sum(observed_folds == fold)) for fold in sorted(set(observed_folds.tolist()))
        },
        "oof_fold_mismatch_count": mismatches,
        "task_max_lengths": {str(task): length for task, length in sorted(task_max_lengths.items())},
        "task_thresholds": {str(task): value for task, value in sorted(thresholds.items())},
        "task_positive_ratios": {
            str(task): float(np.mean(indicators[task_indices == task]))
            for task in sorted(task_max_lengths)
        },
        "value_field": value_field,
        "advantage_field": advantage_field,
        "indicator_field": indicator_field,
        "n_step": n_step,
        "positive_ratio_target": positive_ratio,
        "c_fail_coef": c_fail_coef,
        "global_per_task_thresholds": True,
        "intervention_forced_positive": False,
    }
    if not dry_run:
        _write_columns_in_place(
            dataset_root=dataset_root,
            absolute_indices=absolute_indices,
            columns={
                advantage_field: advantages.astype(np.float32),
                indicator_field: indicators.astype(np.int64),
            },
            feature_infos={
                advantage_field: {"dtype": "float32", "shape": (1,), "names": None},
                indicator_field: {"dtype": "int64", "shape": (1,), "names": None},
            },
        )
        report_path = dataset_root / "OOF_ACP_FINALIZATION.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--folds-path", type=Path, default=None)
    parser.add_argument("--value-field", default="complementary_info.value")
    parser.add_argument("--fold-field", default="complementary_info.value_oof_fold")
    parser.add_argument("--advantage-field", default="complementary_info.advantage")
    parser.add_argument("--indicator-field", default="complementary_info.acp_indicator")
    parser.add_argument("--intervention-field", default="complementary_info.is_intervention")
    parser.add_argument("--n-step", type=int, default=50)
    parser.add_argument("--positive-ratio", type=float, default=0.3)
    parser.add_argument("--c-fail-coef", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n_step <= 0 or not 0 <= args.positive_ratio <= 1 or args.c_fail_coef < 0:
        parser.error("invalid n-step, positive-ratio, or c-fail-coef")
    dataset_root = args.dataset_root.expanduser().resolve()
    folds_path = (
        args.folds_path.expanduser().resolve()
        if args.folds_path is not None
        else dataset_root / "episode_folds.json"
    )
    report = finalize(
        dataset_root=dataset_root,
        folds_path=folds_path,
        value_field=args.value_field,
        fold_field=args.fold_field,
        advantage_field=args.advantage_field,
        indicator_field=args.indicator_field,
        intervention_field=args.intervention_field,
        n_step=args.n_step,
        positive_ratio=args.positive_ratio,
        c_fail_coef=args.c_fail_coef,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
