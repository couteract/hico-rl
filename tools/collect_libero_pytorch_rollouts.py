#!/usr/bin/env python3
"""Collect policy rollouts from the PyTorch LIBERO interface.

This is the online counterpart of ``convert_phase264_libero_rollouts.py``. It
stores the action actually submitted to LIBERO and the post-processor-ready
observation, so the result can be used as a LeRobot v3 training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


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


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _features() -> dict:
    return {
        "observation.images.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.policy_action": {"dtype": "float32", "shape": (7,), "names": None},
        "complementary_info.is_exploration": {"dtype": "int64", "shape": (1,), "names": None},
        "complementary_info.source_episode_id": {"dtype": "int64", "shape": (1,), "names": None},
        "complementary_info.init_state_id": {"dtype": "int64", "shape": (1,), "names": None},
    }


def _one(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value[0]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    task_ids = [int(item) for item in args.task_ids.split(",") if item.strip()]
    if not task_ids:
        parser.error("--task-ids must contain at least one integer")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    env_cfg = LiberoEnvConfig(
        task=args.task_suite,
        task_ids=task_ids,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=args.episode_length,
    )
    envs_by_suite = make_env(env_cfg, n_envs=1)
    envs = envs_by_suite[args.task_suite]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg)
    rename_map = {f"{OBS_IMAGES}.image2": f"{OBS_IMAGES}.wrist_image"}
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"rename_observations_processor": {"rename_map": rename_map}},
    )

    output = LeRobotDataset.create(
        repo_id="local/libero-online-pytorch-rl",
        root=args.output_root,
        fps=env_cfg.fps,
        robot_type="libero_sim",
        features=_features(),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    episodes = []
    try:
        with torch.inference_mode():
            for task_id in task_ids:
                env = envs[task_id]
                task_text = env.envs[0].task_description
                if task_text not in LIBERO_TASK_TEXT_TO_INDEX:
                    raise KeyError(f"LIBERO task text is absent from the PyTorch task map: {task_text!r}")
                task_index = LIBERO_TASK_TEXT_TO_INDEX[task_text]
                for local_ep in range(args.episodes_per_task):
                    seed = args.seed + len(episodes)
                    obs, _ = env.reset(seed=[seed])
                    policy.reset()
                    frame_count = 0
                    success = False
                    while True:
                        raw = preprocess_observation(obs)
                        raw = add_envs_task(env, raw)
                        processed = env_pre(raw)
                        policy_input = pre(processed)
                        raw_policy_action = policy.select_action(policy_input)
                        # Match lerobot_eval's processor contract exactly: the policy
                        # postprocessor consumes a tensor, while the environment
                        # postprocessor consumes an action transition dictionary.
                        policy_action = post(raw_policy_action)
                        env_action = env_post({ACTION: policy_action})[ACTION]
                        action_np = env_action.detach().cpu().numpy().astype(np.float32)[0]
                        policy_np = policy_action.detach().cpu().numpy().astype(np.float32)[0]

                        image = processed[f"{OBS_IMAGES}.image"][0].detach().cpu().numpy()
                        wrist_key = f"{OBS_IMAGES}.wrist_image"
                        if wrist_key not in processed:
                            wrist_key = f"{OBS_IMAGES}.image2"
                        wrist = processed[wrist_key][0].detach().cpu().numpy()
                        state = processed[OBS_STATE][0].detach().cpu().numpy().astype(np.float32)
                        if image.shape[0] == 3:
                            image = np.transpose(image, (1, 2, 0))
                            wrist = np.transpose(wrist, (1, 2, 0))
                        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                        wrist = np.clip(wrist * 255.0, 0, 255).astype(np.uint8)
                        obs, reward, terminated, truncated, info = env.step(action_np[None, :])
                        done = bool(_one(terminated) or _one(truncated))
                        if done:
                            final_info = info.get("final_info") if isinstance(info, dict) else None
                            if not isinstance(final_info, dict) or "is_success" not in final_info:
                                raise RuntimeError(
                                    "LIBERO closed an episode without Gymnasium >=1.0 final_info.is_success."
                                )
                            success = bool(_one(final_info["is_success"]))
                        else:
                            success = False
                        output.add_frame({
                            "task": task_text,
                            "observation.images.image": image,
                            "observation.images.wrist_image": wrist,
                            "observation.state": state,
                            "action": action_np,
                            "next.reward": np.asarray([float(_one(reward))], dtype=np.float32),
                            "next.success": np.asarray([success], dtype=np.bool_),
                            "next.done": np.asarray([done], dtype=np.bool_),
                            "next.truncated": np.asarray([bool(_one(truncated))], dtype=np.bool_),
                            "complementary_info.policy_action": policy_np,
                            "complementary_info.is_exploration": np.asarray([0], dtype=np.int64),
                            "complementary_info.source_episode_id": np.asarray([len(episodes)], dtype=np.int64),
                            "complementary_info.init_state_id": np.asarray([seed], dtype=np.int64),
                        })
                        frame_count += 1
                        if done:
                            break
                    output.save_episode(parallel_encoding=False, extra_episode_metadata={
                        "episode_success": "success" if success else "failure",
                        "task_index_contract": task_index,
                        "suite_task_id": task_id,
                        "task_mapping_version": "exact_task_text_pytorch_libero10_v1",
                        "source_episode_id": len(episodes),
                        "source_init_state_id": seed,
                        "seed": seed,
                        "behavior_policy_id": str(args.policy_path),
                        "behavior_checkpoint_sha256": args.checkpoint_sha256 or _sha256(args.policy_path / "model.safetensors"),
                        "training_action_semantics": "env_action_exactly_submitted_to_libero",
                        "human_intervention": False,
                    })
                    episodes.append({
                        "episode_index": len(episodes),
                        "task_index": task_index,
                        "suite_task_id": task_id,
                        "task_text": task_text,
                        "success": success,
                        "frames": frame_count,
                        "seed": seed,
                    })
                    print(json.dumps(episodes[-1]), flush=True)
    finally:
        output.finalize()
        close_envs(envs_by_suite)

    (args.output_root / "ONLINE_COLLECTION_MANIFEST.json").write_text(json.dumps({
        "schema_version": 1,
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "episodes": episodes,
        "episode_count": len(episodes),
        "human_intervention_labels_synthesized": False,
        "task_mapping": {
            "rule": "exact task-text join to the PyTorch LIBERO-10 task table",
            "task_text_to_index": LIBERO_TASK_TEXT_TO_INDEX,
            "suite_task_id_is_provenance_only": True,
        },
        "action_mapping": {"env_action": "action", "policy_action": "complementary_info.policy_action"},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
