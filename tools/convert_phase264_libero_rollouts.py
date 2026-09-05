#!/usr/bin/env python3
"""Audit and convert Phase264 LIBERO NPZ rollouts to LeRobotDataset v3.

The source rollout view stays read-only.  The converted dataset stores the
actual action submitted to LIBERO as ``action`` and retains the unperturbed
policy action as provenance.  No human-intervention labels are synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    "/home/zhang/xzw/two/evo-rl-jax/Evo-RL-main/outputs/phase264_libero_rl_rollouts"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/phase264_libero10_lerobot_v3"
BASELINE_SHA256 = "3534dea6c61e6581290b6eee699288c5007cba7971efaefa4894f40b42297d71"
LIBERO_TASK_TEXT_TO_INDEX = {
    "put the white mug on the left plate and put the yellow and white mug on the right plate": 0,
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": 1,
    "put the yellow and white mug in the microwave and close it": 2,
    "turn on the stove and put the moka pot on it": 3,
    "put both the alphabet soup and the cream cheese box in the basket": 4,
    "put both the alphabet soup and the tomato sauce in the basket": 5,
    "put both moka pots on the stove": 6,
    "put both the cream cheese box and the butter in the basket": 7,
    "put the black bowl in the bottom drawer of the cabinet and close it": 8,
    "pick up the book and place it in the back compartment of the caddy": 9,
}
REQUIRED_ARRAYS = {
    "camera1",
    "camera2",
    "robot_state",
    "policy_action",
    "env_action",
    "reward",
    "success",
    "terminated",
    "truncated",
    "frame_index",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_source(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source_root / "DATASET_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("baseline_checkpoint_sha256") != BASELINE_SHA256:
        raise ValueError("Source manifest does not reference the pinned Phase139 baseline.")
    records = list(manifest.get("records", []))
    if not records:
        raise ValueError("Source manifest has no frame-ready records.")
    return manifest, records


def _audit_record(source_root: Path, record: dict[str, Any], verify_sha256: bool) -> dict[str, Any]:
    transition_path = source_root / record["view_transitions"]
    metadata_path = source_root / record["view_metadata"]
    if not transition_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete rollout view for source episode {record['episode_id']}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "episode_id": int(record["episode_id"]),
        "episode_success": bool(record["success"]),
        "task_text": str(record["task_text"]),
        "suite_task_id": int(record["suite_task_id"]),
        "checkpoint_sha256": BASELINE_SHA256,
        "real_libero_rollout": True,
        "mock_environment": False,
        "training_eligible": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Episode {record['episode_id']} metadata mismatch for {key}: "
                f"expected={expected!r} actual={metadata.get(key)!r}"
            )
    canonical = metadata.get("canonical_task_resolution", {}).get("canonical_task_id")
    if int(canonical) != int(record["canonical_task_id"]):
        raise ValueError(f"Episode {record['episode_id']} canonical task mapping is inconsistent.")
    if verify_sha256 and _sha256(transition_path) != record["transitions_sha256"]:
        raise ValueError(f"Episode {record['episode_id']} transition SHA256 changed.")

    with np.load(transition_path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_ARRAYS - set(archive.files))
        if missing:
            raise ValueError(f"Episode {record['episode_id']} is missing arrays: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in REQUIRED_ARRAYS}

    frame_count = int(record["frame_count"])
    expected_shapes = {
        "camera1": (frame_count, 256, 256, 3),
        "camera2": (frame_count, 256, 256, 3),
        "robot_state": (frame_count, 8),
        "policy_action": (frame_count, 7),
        "env_action": (frame_count, 7),
        "reward": (frame_count,),
        "success": (frame_count,),
        "terminated": (frame_count,),
        "truncated": (frame_count,),
        "frame_index": (frame_count,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(
                f"Episode {record['episode_id']} {key} shape mismatch: "
                f"expected={shape} actual={arrays[key].shape}"
            )
    if arrays["camera1"].dtype != np.uint8 or arrays["camera2"].dtype != np.uint8:
        raise TypeError(f"Episode {record['episode_id']} cameras must be uint8.")
    if any(arrays[key].dtype != np.bool_ for key in ("success", "terminated", "truncated")):
        raise TypeError(f"Episode {record['episode_id']} terminal arrays must be bool.")
    if not np.array_equal(arrays["frame_index"], np.arange(frame_count, dtype=np.int64)):
        raise ValueError(f"Episode {record['episode_id']} frame indices are not contiguous.")
    if any(not np.isfinite(arrays[key]).all() for key in ("robot_state", "policy_action", "env_action", "reward")):
        raise ValueError(f"Episode {record['episode_id']} contains non-finite numeric data.")
    if np.any((arrays["terminated"] | arrays["truncated"])[:-1]):
        raise ValueError(f"Episode {record['episode_id']} has an early terminal boundary.")
    if not bool(arrays["terminated"][-1] or arrays["truncated"][-1]):
        raise ValueError(f"Episode {record['episode_id']} has no closed terminal boundary.")
    if bool(arrays["terminated"][-1] and arrays["truncated"][-1]):
        raise ValueError(f"Episode {record['episode_id']} is both terminated and truncated.")
    if bool(arrays["success"].any()) != bool(record["success"]):
        raise ValueError(f"Episode {record['episode_id']} success label disagrees with frames.")
    if record["success"] and not bool(arrays["success"][-1] and arrays["terminated"][-1]):
        raise ValueError(f"Successful episode {record['episode_id']} does not terminate on success.")

    exploration = bool(record["exploration"])
    actions_differ = not np.array_equal(arrays["policy_action"], arrays["env_action"])
    if exploration != actions_differ:
        raise ValueError(
            f"Episode {record['episode_id']} exploration flag/action provenance mismatch: "
            f"exploration={exploration} actions_differ={actions_differ}"
        )
    return {"metadata": metadata, "arrays": arrays, "actions_differ": actions_differ}


def _build_folds(records: list[dict[str, Any]], episode_map: dict[int, int], num_folds: int) -> dict[str, Any]:
    strata: dict[tuple[int, bool], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_index = LIBERO_TASK_TEXT_TO_INDEX[str(record["task_text"])]
        strata[(task_index, bool(record["success"]))].append(record)

    test_by_fold: list[list[int]] = [[] for _ in range(num_folds)]
    for (task_id, success), rows in sorted(strata.items()):
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"phase264-fold-v2:{task_id}:{int(success)}:{row['init_state_id']}:{row['episode_id']}".encode()
            ).hexdigest()
        )
        for index, row in enumerate(rows):
            test_by_fold[index % num_folds].append(episode_map[int(row["episode_id"])])

    all_episodes = set(episode_map.values())
    folds = []
    for fold_index, test_episodes in enumerate(test_by_fold):
        test_set = set(test_episodes)
        train_episodes = sorted(all_episodes - test_set)
        summary = Counter(
            (LIBERO_TASK_TEXT_TO_INDEX[str(row["task_text"])], "success" if row["success"] else "failure")
            for row in records
            if episode_map[int(row["episode_id"])] in test_set
        )
        folds.append(
            {
                "fold": fold_index,
                "train_episodes": train_episodes,
                "test_episodes": sorted(test_set),
                "test_counts": {
                    f"task_{task_id}_{outcome}": count
                    for (task_id, outcome), count in sorted(summary.items())
                },
            }
        )
    flattened = [episode for fold in folds for episode in fold["test_episodes"]]
    if sorted(flattened) != sorted(all_episodes):
        raise RuntimeError("Cross-validation folds do not cover every episode exactly once.")
    return {
        "schema_version": 1,
        "strategy": "deterministic round-robin within PyTorch LIBERO task_index x episode_success strata",
        "group_key": "source init_state_id (all source init states are unique in Phase264)",
        "num_folds": num_folds,
        "folds": folds,
    }


def _dataset_features() -> dict[str, dict[str, Any]]:
    return {
        "observation.images.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["eef_x", "eef_y", "eef_z", "axis_x", "axis_y", "axis_z", "gripper_0", "gripper_1"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
        "complementary_info.is_exploration": {"dtype": "int64", "shape": (1,), "names": None},
        "complementary_info.source_episode_id": {"dtype": "int64", "shape": (1,), "names": None},
        "complementary_info.init_state_id": {"dtype": "int64", "shape": (1,), "names": None},
    }


def _convert(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    records: list[dict[str, Any]],
    audits: dict[int, dict[str, Any]],
    source_manifest_sha256: str,
    fps: int,
    num_folds: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="libero_sim",
        features=_dataset_features(),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    episode_map: dict[int, int] = {}
    try:
        for output_episode_index, record in enumerate(records):
            source_episode_id = int(record["episode_id"])
            episode_map[source_episode_id] = output_episode_index
            metadata = audits[source_episode_id]["metadata"]
            # Reload one archive at a time for conversion. Keeping every NPZ
            # array resident would require roughly the full 11 GB source size.
            converted_audit = _audit_record(source_root, record, verify_sha256=False)
            arrays = converted_audit["arrays"]
            task_text = str(record["task_text"])
            task_index = LIBERO_TASK_TEXT_TO_INDEX.get(task_text)
            if task_index is None:
                raise ValueError(f"Unknown PyTorch LIBERO-10 task text: {task_text!r}")
            exploration = bool(record["exploration"])
            frame_count = int(record["frame_count"])
            for frame_index in range(frame_count):
                # JAX archives retain simulator orientation.  PyTorch LIBERO's
                # environment processor flips H and W before policy inference.
                main_image = np.ascontiguousarray(arrays["camera1"][frame_index, ::-1, ::-1])
                wrist_image = np.ascontiguousarray(arrays["camera2"][frame_index, ::-1, ::-1])
                terminated = bool(arrays["terminated"][frame_index])
                truncated = bool(arrays["truncated"][frame_index])
                dataset.add_frame(
                    {
                        "task": task_text,
                        "observation.images.image": main_image,
                        "observation.images.wrist_image": wrist_image,
                        "observation.state": arrays["robot_state"][frame_index].astype(np.float32, copy=False),
                        # The behavior action caused the next state.  This is
                        # env_action for both clean and exploration episodes.
                        "action": arrays["env_action"][frame_index].astype(np.float32, copy=False),
                        "next.reward": np.asarray([arrays["reward"][frame_index]], dtype=np.float32),
                        "next.success": np.asarray([arrays["success"][frame_index]], dtype=np.bool_),
                        "next.done": np.asarray([terminated or truncated], dtype=np.bool_),
                        "next.truncated": np.asarray([truncated], dtype=np.bool_),
                        "complementary_info.policy_action": arrays["policy_action"][frame_index].astype(
                            np.float32, copy=False
                        ),
                        "complementary_info.is_exploration": np.asarray([int(exploration)], dtype=np.int64),
                        "complementary_info.source_episode_id": np.asarray([source_episode_id], dtype=np.int64),
                        "complementary_info.init_state_id": np.asarray(
                            [int(record["init_state_id"])], dtype=np.int64
                        ),
                    }
                )
            exploration_contract = metadata.get("behavior_exploration_contract") or {
                "enabled": False,
                "kind": "none",
            }
            dataset.save_episode(
                parallel_encoding=False,
                extra_episode_metadata={
                    "episode_success": "success" if record["success"] else "failure",
                    "source_episode_id": source_episode_id,
                    "source_init_state_id": int(record["init_state_id"]),
                    "task_index_contract": task_index,
                    "source_canonical_task_id": int(record["canonical_task_id"]),
                    "source_dataset_task_id": int(metadata["dataset_task_id"]),
                    "suite_task_id": int(record["suite_task_id"]),
                    "collection_round": int(metadata.get("collection_round", 0)),
                    "behavior_exploration": exploration,
                    "behavior_policy_id": str(record["behavior_policy_id"]),
                    "behavior_checkpoint_sha256": str(record["behavior_checkpoint_sha256"]),
                    "source_transitions_sha256": str(record["transitions_sha256"]),
                    "source_mapping_version": str(record["mapping_version"]),
                    "exploration_contract_json": json.dumps(exploration_contract, sort_keys=True),
                    "training_action_semantics": "env_action_exactly_submitted_to_libero",
                },
            )
            print(
                f"converted {output_episode_index + 1}/{len(records)} "
                f"source_episode={source_episode_id} frames={frame_count}",
                flush=True,
            )
            del arrays, converted_audit
    finally:
        dataset.finalize()

    folds = _build_folds(records, episode_map, num_folds)
    _write_json(output_root / "episode_folds.json", folds)
    conversion_manifest = {
        "schema_version": 1,
        "source_root": source_root.resolve().as_posix(),
        "source_manifest_sha256": source_manifest_sha256,
        "source_baseline_checkpoint_sha256": BASELINE_SHA256,
        "repo_id": repo_id,
        "fps": fps,
        "episode_count": len(records),
        "frame_count": sum(int(record["frame_count"]) for record in records),
        "summary_only_records_imported": 0,
        "human_intervention_labels_synthesized": False,
        "task_mapping": {
            "rule": "exact task-text join to the PyTorch LIBERO-10 dataset task table",
            "task_text_to_index": LIBERO_TASK_TEXT_TO_INDEX,
            "source_canonical_task_id_is_provenance_only": True,
        },
        "image_mapping": {
            "camera1": "observation.images.image",
            "camera2": "observation.images.wrist_image",
            "transform": "flip height and width (simulator_raw -> policy_input)",
        },
        "action_mapping": {
            "env_action": "action",
            "policy_action": "complementary_info.policy_action",
        },
        "episode_index_mapping": [
            {
                "source_episode_id": int(record["episode_id"]),
                "lerobot_episode_index": episode_map[int(record["episode_id"])],
                "task_index": LIBERO_TASK_TEXT_TO_INDEX[str(record["task_text"])],
                "source_canonical_task_id": int(record["canonical_task_id"]),
                "success": bool(record["success"]),
                "exploration": bool(record["exploration"]),
            }
            for record in records
        ],
    }
    _write_json(output_root / "CONVERSION_MANIFEST.json", conversion_manifest)
    return conversion_manifest


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", default="local/phase264-libero10-rl")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip expensive archive hashing. Shape, dtype, terminal, and provenance checks still run.",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.num_folds < 2:
        parser.error("--fps must be positive and --num-folds must be at least 2")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest, records = _load_source(source_root)
    unknown_texts = sorted({str(row["task_text"]) for row in records} - set(LIBERO_TASK_TEXT_TO_INDEX))
    if unknown_texts:
        raise ValueError(f"Source contains task text absent from the PyTorch LIBERO-10 task map: {unknown_texts}")
    records.sort(key=lambda row: (LIBERO_TASK_TEXT_TO_INDEX[str(row["task_text"])], int(row["episode_id"])))
    if args.max_episodes is not None:
        records = records[: args.max_episodes]

    audits: dict[int, dict[str, Any]] = {}
    task_texts: dict[int, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        audited = _audit_record(source_root, record, verify_sha256=not args.skip_sha256)
        source_episode_id = int(record["episode_id"])
        audits[source_episode_id] = {"metadata": audited["metadata"]}
        del audited
        task_texts[LIBERO_TASK_TEXT_TO_INDEX[str(record["task_text"])]].add(str(record["task_text"]))
        print(f"audited {index + 1}/{len(records)} source_episode={source_episode_id}", flush=True)
    ambiguous_tasks = {task: sorted(texts) for task, texts in task_texts.items() if len(texts) != 1}
    if ambiguous_tasks:
        raise ValueError(f"PyTorch task indices have ambiguous text: {ambiguous_tasks}")

    audit_report = {
        "ok": True,
        "source_root": source_root.as_posix(),
        "source_manifest_sha256": _sha256(source_root / "DATASET_MANIFEST.json"),
        "audited_episode_count": len(records),
        "audited_frame_count": sum(int(record["frame_count"]) for record in records),
        "success_episode_count": sum(bool(record["success"]) for record in records),
        "failure_episode_count": sum(not bool(record["success"]) for record in records),
        "exploration_episode_count": sum(bool(record["exploration"]) for record in records),
        "summary_only_records_excluded": int(manifest.get("summary_only_episode_count", 0)),
        "sha256_verified": not args.skip_sha256,
        "human_intervention_required": False,
        "training_action": "env_action",
        "task_index_source": "exact PyTorch LIBERO task-text mapping",
        "legacy_source_task_id_count": sum(
            int(record["canonical_task_id"])
            != LIBERO_TASK_TEXT_TO_INDEX[str(record["task_text"])]
            for record in records
        ),
        "pytorch_task_counts": {
            str(task_index): {
                "success": sum(
                    LIBERO_TASK_TEXT_TO_INDEX[str(record["task_text"])] == task_index
                    and bool(record["success"])
                    for record in records
                ),
                "failure": sum(
                    LIBERO_TASK_TEXT_TO_INDEX[str(record["task_text"])] == task_index
                    and not bool(record["success"])
                    for record in records
                ),
            }
            for task_index in sorted(LIBERO_TASK_TEXT_TO_INDEX.values())
        },
    }
    print(json.dumps(audit_report, indent=2, sort_keys=True), flush=True)
    if args.audit_only:
        return audit_report

    result = _convert(
        source_root=source_root,
        output_root=output_root,
        repo_id=args.repo_id,
        records=records,
        audits=audits,
        source_manifest_sha256=audit_report["source_manifest_sha256"],
        fps=args.fps,
        num_folds=args.num_folds,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
