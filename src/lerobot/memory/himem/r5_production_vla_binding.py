"""B-local production binding for restored Q20R5 LIBERO states."""
from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lerobot.memory.himem.r5_snapshot_rollout import (
    validate_snapshot_contract,
    validate_snapshot_payload,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def _native_scalar(value: Any, label: str, convert):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"ENV_RESULT_NOT_SCALAR:{label}:{array.shape}")
    item = array.reshape(-1)[0].item() if hasattr(array.reshape(-1)[0], "item") else array.reshape(-1)[0]
    result = convert(item)
    if isinstance(result, float) and not np.isfinite(result):
        raise ValueError(f"ENV_RESULT_NONFINITE:{label}")
    return result

def sanitize_environment_result(*, terminated: Any, truncated: Any, reward: Any, info: Any,
                                frame: int, env_action7_sha256: str) -> dict[str, Any]:
    raw_info = info if isinstance(info, dict) else {}
    raw_done = _native_scalar(raw_info.get("done", False), "done", bool)
    term = _native_scalar(terminated, "terminated", bool)
    trunc = _native_scalar(truncated, "truncated", bool)
    reward_value = _native_scalar(reward, "reward", float)
    final_info = raw_info.get("final_info", raw_info) if (raw_done or term or trunc) else {}
    if not isinstance(final_info, dict): final_info = {}
    success = None
    for key in ("success", "is_success"):
        if key in final_info:
            success = _native_scalar(final_info[key], key, bool)
            break
    return {"schema_version":"B02Q32_ENV_RESULT_V1","raw_done":raw_done,"terminated":term,
        "truncated":trunc,"success":success,"frame":int(frame),"env_action7_sha256":str(env_action7_sha256),
        "final_info_keys":sorted(str(k) for k in final_info),
        "final_info_types":{str(k):type(v).__name__ for k,v in final_info.items()},
        "controller_effective_action":"NOT_OBSERVED","reward":reward_value}


@dataclass(frozen=True)
class RestoreEvidence:
    """Immutable references to the two external restore attestations."""

    evidence_a: Path
    evidence_a_size: int
    evidence_a_sha256: str
    evidence_b: Path
    evidence_b_size: int
    evidence_b_sha256: str
    physical_group_key: str
    snapshot_identity: str


def _load_external_restore_evidence(evidence: RestoreEvidence, *, snapshot: Path,
                                    physical_group_key: str) -> dict[str, Any]:
    if not isinstance(evidence, RestoreEvidence):
        raise TypeError("RESTORE_EVIDENCE_MUST_BE_TYPED")
    if evidence.physical_group_key != physical_group_key:
        raise ValueError("RESTORE_EVIDENCE_GROUP_MISMATCH")
    identity = json.loads((snapshot / "identity.json").read_text())
    identity_candidates = {str(identity.get(k, "")) for k in ("physical_group_key", "group", "identity", "identity_id")}
    if evidence.snapshot_identity not in identity_candidates:
        raise ValueError("RESTORE_EVIDENCE_SNAPSHOT_IDENTITY_MISMATCH")
    docs = []
    for label, path, size, digest in (
        ("A", evidence.evidence_a, evidence.evidence_a_size, evidence.evidence_a_sha256),
        ("B", evidence.evidence_b, evidence.evidence_b_size, evidence.evidence_b_sha256),
    ):
        if Path(path).name != f"{physical_group_key}_{label}.json":
            raise ValueError(f"RESTORE_EVIDENCE_{label}_ROLE_OR_GROUP_PATH_MISMATCH")
        raw = Path(path).read_bytes()
        if len(raw) != size or _sha_bytes(raw) != digest:
            raise ValueError(f"RESTORE_EVIDENCE_{label}_BYTE_BINDING_MISMATCH")
        doc = json.loads(raw)
        if doc.get("status") != "PASS" or int(doc.get("env_step", -1)) != 0:
            raise ValueError(f"RESTORE_EVIDENCE_{label}_STATUS_OR_STEP_INVALID")
        if any(int(doc.get(k, -1)) != 0 for k in ("policy_action", "reward_outcome")):
            raise ValueError(f"RESTORE_EVIDENCE_{label}_COUNTERS_INVALID")
        docs.append(doc)
    if docs[0].get("fields") != docs[1].get("fields"):
        raise ValueError("RESTORE_EVIDENCE_AB_FIELDS_DISAGREE")
    for doc in docs:
        recorded = Path(str(doc.get("path", "")))
        # R6 evidence records the snapshot's atomic staging name (``.tmp``)
        # while the adopted immutable snapshot has the final directory name.
        if recorded.parent.resolve() != snapshot.parent.resolve() or recorded.name not in {
            snapshot.name, snapshot.name + ".tmp"
        }:
            raise ValueError("RESTORE_EVIDENCE_DOCUMENT_IDENTITY_MISMATCH")
        fields = doc.get("fields")
        required = {"integration", "qpos", "qvel", "time", "state8", "agentview", "eye", "python_rng", "numpy_rng"}
        if not isinstance(fields, dict) or not required.issubset(fields):
            raise ValueError("RESTORE_EVIDENCE_FIELDS_INCOMPLETE")
        if any(value.get("status") != "PASS" for value in fields.values() if isinstance(value, dict) and "status" in value):
            raise ValueError(f"RESTORE_EVIDENCE_{label}_FIELD_STATUS_INVALID")
    docs[0]["_external_pair"] = docs
    return docs[0]


class ProductionBindingAdapter:
    """Production A10F frame path over a one-shot restored environment."""

    def __init__(self, *, envs: Any, env: Any, env_pre: Any, env_post: Any,
                 vla: Any, vla_pre: Any, vla_post: Any, task: int,
                 observation: dict[str, Any], visual_capture: Any = None,
                 horizon: int = 520, restore_validation: dict[str, Any] | None = None):
        self.envs, self.env = envs, env
        self.env_pre, self.env_post = env_pre, env_post
        self.vla, self.vla_pre, self.vla_post = vla, vla_pre, vla_post
        self.task = int(task)
        self.obs = observation
        self.visual_capture = visual_capture
        self.restore_validation = restore_validation or {}
        self.frame = 0
        self.horizon = int(horizon)
        self.env_step = 0
        self.last_env_action7: np.ndarray | None = None
        self.environment_reset_after_restore = 0
        self.vla_reset_calls = 0
        self.vla_forward_calls = 0
        self.visual_capture_calls = 0
        self.visual_contexts = []
        self.last_processed = None
        self.last_batch = None
        self.last_normalized = None
        self._closed = False
        self._set_autoreset_disabled()
        self._reset_vla_only()

    def _set_autoreset_disabled(self) -> None:
        try:
            import gymnasium as gym
            self.env.autoreset_mode = gym.vector.AutoresetMode.DISABLED
        except (AttributeError, ImportError):
            pass
        metadata = getattr(self.env, "metadata", None)
        if isinstance(metadata, dict):
            try:
                import gymnasium as gym
                metadata["autoreset_mode"] = gym.vector.AutoresetMode.DISABLED
            except ImportError:
                metadata["autoreset_mode"] = "disabled"

    def _reset_vla_only(self) -> None:
        reset = getattr(self.vla, "reset", None)
        if callable(reset):
            reset()
            self.vla_reset_calls += 1

    def reset(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ORDINARY_RESET_FORBIDDEN_AFTER_R5_RESTORE")

    def enqueue_vla_action(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("MANUAL_VLA_QUEUE_FORBIDDEN_IN_PRODUCTION")

    @staticmethod
    def _action_queue(vla: Any) -> Any:
        from lerobot.utils.constants import ACTION
        queues = getattr(vla, "_queues", None)
        if queues is None or ACTION not in queues:
            raise RuntimeError("VLA_ACTION_QUEUE_UNAVAILABLE")
        return queues[ACTION]

    def pop_vla_action(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        import torch
        from lerobot.envs.utils import add_envs_task, preprocess_observation
        from lerobot.utils.constants import OBS_STATE

        if self.obs is None:
            raise RuntimeError("EPISODE_FINISHED_OR_NOT_BOUND")
        queue = self._action_queue(self.vla)
        before = len(queue)
        raw = add_envs_task(self.env, preprocess_observation(self.obs))
        processed = self.env_pre(raw)
        batch = self.vla_pre(processed)
        self.last_processed = processed
        self.last_batch = batch
        boundary = before == 0 and self.visual_capture is not None
        if boundary:
            expert = getattr(getattr(self.vla, "model", None), "vlm_with_expert", None)
            if expert is None:
                raise RuntimeError("VISUAL_CAPTURE_MODEL_ENDPOINT_MISSING")
            original = getattr(expert, "embed_image")
            camera_order = tuple(self.visual_capture.camera_order)
            calls = 0
            reset_boundary = getattr(self.visual_capture, "reset_boundary", None)
            if callable(reset_boundary):
                reset_boundary()
            def wrapped(*args: Any, **kwargs: Any):
                nonlocal calls
                if calls >= len(camera_order):
                    raise RuntimeError("VISUAL_CAPTURE_CALL_COUNT_EXCEEDED")
                result = original(*args, **kwargs)
                value = self.visual_capture.record(camera_order[calls], result)
                calls += 1
                return value
            setattr(expert, "embed_image", wrapped)
            try:
                normalized = self.vla.select_action(batch)
                if calls != len(camera_order):
                    raise RuntimeError(f"VISUAL_CAPTURE_CALL_COUNT:{calls}")
            finally:
                setattr(expert, "embed_image", original)
            self.visual_capture_calls += len(self.visual_capture.camera_order)
            camera_mask = {key: True for key in self.visual_capture.camera_order}
            self.visual_contexts.append(self.visual_capture.build_boundary(boundary_id=len(self.visual_contexts), camera_mask=camera_mask))
        else:
            normalized = self.vla.select_action(batch)
        self.last_normalized = normalized
        self.vla_forward_calls += 1
        state = processed[OBS_STATE][0].detach().cpu().numpy().astype(np.float32, copy=True)
        base = self.vla_post(normalized).detach().cpu().numpy()[0].astype(np.float32, copy=True)
        after = len(queue)
        expected = 4 if before == 0 else before - 1
        if after != expected:
            raise RuntimeError(f"VLA_QUEUE_TRANSITION_INVALID:{before}->{after}")
        return state, base, before, after

    def step(self, submitted7: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Send the CPU float32 batch to the vector env and parse one env result."""
        import torch
        from lerobot.utils.constants import ACTION

        if self.obs is None:
            raise RuntimeError("EPISODE_FINISHED_OR_NOT_BOUND")
        if self.frame >= self.horizon:
            raise RuntimeError("HORIZON_BUDGET_EXHAUSTED")
        sent = np.asarray(submitted7, dtype=np.float32)
        if sent.shape != (7,) or not np.isfinite(sent).all():
            raise ValueError("ENV_ACTION_MUST_BE_FINITE_SHAPE7")
        posted = self.env_post({ACTION: torch.as_tensor(sent).unsqueeze(0)})[ACTION]
        actual = posted.detach().cpu().numpy()[0].astype(np.float32)
        if actual.shape != (7,) or actual.tobytes() != sent.tobytes():
            raise RuntimeError("ENV_POSTPROCESSOR_CHANGED_ACTION")
        next_obs, reward, terminated, truncated, info = self.env.step(actual[None, :])
        self.last_env_action7 = actual
        self.env_step += 1
        self.frame += 1
        result = sanitize_environment_result(terminated=terminated, truncated=truncated, reward=reward,
            info=info, frame=self.frame, env_action7_sha256=_sha_bytes(actual.tobytes()))
        raw_done = result['raw_done']
        interface_terminated=result['terminated'];interface_truncated=result['truncated'];reward_value=result['reward']
        if raw_done or interface_terminated or interface_truncated:
            self.obs = None
        else:
            self.obs = next_obs
        return next_obs, reward_value, interface_terminated, interface_truncated, {
            **{k:v for k,v in result.items() if k!='reward'},
        }

    def close(self) -> None:
        if not self._closed:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
            self._closed = True


def _batch_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _batch_observation(item) for key, item in value.items()}
    array = np.asarray(value)
    return array[None, ...].copy()


def _physical_env(vec_env: Any) -> Any:
    return vec_env.envs[0].unwrapped


def _restore_snapshot_into_libero(unwrapped: Any, snapshot: Path, *,
                                  restore_evidence: RestoreEvidence | None = None,
                                  physical_group_key: str | None = None) -> dict[str, Any]:
    import mujoco

    meta = json.loads((snapshot / "meta.json").read_text())
    state = dict(np.load(snapshot / "state.npz", allow_pickle=False))
    arrays = dict(np.load(snapshot / "runtime_model_arrays.npz", allow_pickle=False))
    if restore_evidence is None:
        restore_results = json.loads((snapshot / "restore_results.json").read_text())
    else:
        if physical_group_key is None:
            raise ValueError("EXTERNAL_RESTORE_EVIDENCE_REQUIRES_GROUP")
        restore_results = _load_external_restore_evidence(
            restore_evidence, snapshot=snapshot, physical_group_key=physical_group_key)
    paired_docs = restore_results.get("_external_pair", [restore_results])
    for evidence_doc in paired_docs:
        validate_snapshot_payload(meta, arrays, state, restore_results=evidence_doc)
    control = unwrapped._env
    model, data = control.sim.model._model, control.sim.data._data
    for name, value in arrays.items():
        target = np.asarray(getattr(model, name))
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"MODEL_FIELD_SHAPE_OR_DTYPE:{name}")
        target[...] = value
    mujoco.mj_setState(model, data, state["integration"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    runtime_checks = []
    if restore_evidence is not None:
        for evidence_doc in paired_docs:
            runtime_checks.append(_validate_restored_runtime(
                evidence_doc["fields"], model=model, data=data, state=state))
    payload = json.loads((snapshot / "rng_payload.json").read_text())
    from lerobot.memory.himem.r5_snapshot_rollout import _decode
    random.setstate(_decode(payload["python_random"]))
    np.random.set_state(_decode(payload["numpy_legacy"]))
    rng_expected_python = random.getstate()
    rng_expected_numpy = np.random.get_state()
    rng_checks = []
    if restore_evidence is not None:
        rng_before_audit = (random.getstate(), np.random.get_state())
        _assert_rng_state("python_rng", rng_expected_python, rng_before_audit[0])
        _assert_rng_state("numpy_rng", rng_expected_numpy, rng_before_audit[1])
        for evidence_doc in paired_docs:
            if evidence_doc["fields"]["python_rng"].get("status") != "PASS" or evidence_doc["fields"]["numpy_rng"].get("status") != "PASS":
                raise ValueError("RESTORED_RNG_EVIDENCE_STATUS_INVALID")
            rng_checks.append({"python_rng": "PASS", "numpy_rng": "PASS"})
    unwrapped._elapsed_steps = 0
    if hasattr(unwrapped, "active_init_state_id"):
        unwrapped.active_init_state_id = int(meta.get("slot", meta.get("init_state_id", 0)))
    unwrapped.control_mode = "relative"
    observation_env = getattr(unwrapped._env, "env", unwrapped._env)
    raw = observation_env._get_observations(force_update=True)
    observation_checks = []
    if restore_evidence is not None:
        import robosuite.macros as macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        camera_values = {
            "agentview": control.sim.render(camera_name="agentview", width=32, height=32)[::convention],
            "eye": control.sim.render(camera_name="robot0_eye_in_hand", width=32, height=32)[::convention],
            "state8": np.asarray(raw["robot0_proprio-state"])[:8],
        }
        for evidence_doc in paired_docs:
            for key, value in camera_values.items():
                _validate_attested_array(key, value, evidence_doc["fields"][key])
            observation_checks.append({key: "PASS" for key in camera_values})
        _assert_rng_state("python_rng_after_audit", rng_expected_python, random.getstate())
        _assert_rng_state("numpy_rng_after_audit", rng_expected_numpy, np.random.get_state())
    formatted = unwrapped._format_raw_obs(raw)
    return {"model": model, "data": data, "observation": _batch_observation(formatted), "meta": meta,
            "restore_validation": {"runtime": runtime_checks, "rng": rng_checks,
                                   "observation": observation_checks,
                                   "same_256_production_observation": True}}


def _validate_restored_runtime(fields: dict[str, Any], *, model: Any, data: Any,
                               state: dict[str, np.ndarray]) -> dict[str, str]:
    """Check the restored MuJoCo state against the externally attested fields."""
    values: dict[str, Any] = {"integration": state["integration"], "qpos": data.qpos,
                              "qvel": data.qvel, "time": np.asarray([data.time])}
    for key in fields:
        if key.startswith("model_array:"):
            name = key.split(":", 1)[1]
            values[key] = getattr(model, name)
    for name, value in values.items():
        _validate_attested_array(name, value, fields[name], error_prefix="RESTORED_RUNTIME_FIELD_MISMATCH")
    return {name: "PASS" for name in values}


def _validate_attested_array(name: str, value: Any, spec: dict[str, Any], *,
                             error_prefix: str = "RESTORED_OBSERVATION_FIELD_MISMATCH") -> None:
    array = np.asarray(value)
    if (list(array.shape) != spec.get("shape") or np.dtype(array.dtype).str != spec.get("dtype")
            or _sha_bytes(np.ascontiguousarray(array).tobytes()) != spec.get("expected_sha256")):
        raise ValueError(f"{error_prefix}:{name}")


def _assert_rng_state(name: str, expected: Any, actual: Any) -> None:
    if isinstance(expected, np.ndarray):
        equal = isinstance(actual, np.ndarray) and expected.dtype == actual.dtype and expected.shape == actual.shape and expected.tobytes() == actual.tobytes()
    elif isinstance(expected, (tuple, list)):
        equal = isinstance(actual, type(expected)) and len(expected) == len(actual) and all(
            _rng_item_equal(left, right) for left, right in zip(expected, actual))
    else:
        equal = expected == actual
    if not equal:
        raise ValueError(f"RESTORED_RNG_STATE_MISMATCH:{name}")


def _rng_item_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_rng_item_equal(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def _saved_xml_patch(xml: str):
    from robosuite.environments.base import MujocoEnv
    original = MujocoEnv._initialize_sim
    def initialize(instance: Any, xml_string: str | None = None) -> Any:
        return original(instance, xml_string=xml)
    MujocoEnv._initialize_sim = initialize
    try:
        yield
    finally:
        MujocoEnv._initialize_sim = original


def build_production_binding(*, snapshot: str | Path, task: int, physical_group_key: str,
                             route_seed: int, checkpoint: str | Path,
                             horizon: int = 520, device: str = "cuda",
                             visual_capture: Any = None,
                             restore_evidence: RestoreEvidence | None = None) -> ProductionBindingAdapter:
    """Construct A10F processors/VLA, inject R5 XML, restore, then bind obs."""
    snapshot = Path(snapshot).resolve()
    meta = json.loads((snapshot / "meta.json").read_text())
    validate_snapshot_contract(meta, task=task, physical_group_key=physical_group_key,
                               route_seed=route_seed)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401

    env_cfg = LiberoEnvConfig(task="libero_10", task_ids=[task], obs_type="pixels_agent_pos",
                              observation_height=256, observation_width=256, episode_length=horizon)
    vla_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    runtime_path = Path(os.environ["HIMEM_A_BASELINE_RUNTIME"]) if os.environ.get("HIMEM_A_BASELINE_RUNTIME") else None
    if runtime_path is not None and runtime_path.is_file():
        for key, value in json.loads(runtime_path.read_text()).items():
            setattr(vla_cfg, key, value)
        vla_cfg.__post_init__()
    vla_cfg.device = device
    vla_cfg.pretrained_path = str(checkpoint)
    with _saved_xml_patch((snapshot / "model.xml").read_text()):
        envs = make_env(env_cfg, n_envs=1)
    env = envs["libero_10"][task]
    env_pre, env_post = make_env_pre_post_processors(env_cfg, vla_cfg)
    vla = make_policy(cfg=vla_cfg, env_cfg=env_cfg,
                      rename_map={"observation.images.image2": "observation.images.wrist_image"})
    if device == "cuda":
        vla = vla.cuda()
    vla.eval()
    for parameter in vla.parameters():
        parameter.requires_grad_(False)
    overrides = {"rename_observations_processor": {"rename_map": {"observation.images.image2": "observation.images.wrist_image"}},
                 "device_processor": {"device": device}}
    vla_pre, vla_post = make_pre_post_processors(vla_cfg, pretrained_path=str(checkpoint),
                                                  preprocessor_overrides=overrides)
    restored = _restore_snapshot_into_libero(_physical_env(env), snapshot,
                                             restore_evidence=restore_evidence,
                                             physical_group_key=physical_group_key)
    return ProductionBindingAdapter(envs=envs, env=env, env_pre=env_pre, env_post=env_post,
                                    vla=vla, vla_pre=vla_pre, vla_post=vla_post,
                                    task=task, observation=restored["observation"],
                                    visual_capture=visual_capture,
                                    restore_validation=restored.get("restore_validation"))


def compose_route_action(route: str, base_action7: Any, proposal6: Any,
                         effective_arm6: Any) -> np.ndarray:
    """Apply the frozen A21 route composition without touching the gripper."""
    base = np.asarray(base_action7, dtype=np.float32)
    proposal = np.asarray(proposal6, dtype=np.float32)
    effective = np.asarray(effective_arm6, dtype=np.float32)
    if base.shape != (7,) or proposal.shape != (6,) or effective.shape != (6,):
        raise ValueError("ROUTE_ACTION_SHAPE_MISMATCH")
    if route == "absolute_refresh_control":
        arm = effective + np.float32(0.1) * (proposal - effective)
    elif route == "base_relative_refresh_treatment":
        arm = effective + np.float32(0.1) * proposal
    else:
        raise ValueError("UNKNOWN_ROUTE")
    return np.concatenate((arm.astype(np.float32), base[6:7])).astype(np.float32)


MEMORY_VISUAL_SIDE_CALL = "MEMORY_VISUAL_REPEAT4_SLOT0_SIDE_CALL"
ACTOR_ACTION_VISUAL = "ACTOR_ACTION_FORWARD_BATCH1_VISUAL"


def memory_visual_side_call(binding: ProductionBindingAdapter, *, device: str = "cuda",
                            include_numeric: bool = False) -> dict[str, Any]:
    """Read-only repeat4/slot0 memory visual call from the current production batch."""
    import torch

    if binding.last_batch is None:
        raise RuntimeError("MEMORY_VISUAL_REQUIRES_CURRENT_VLA_BATCH")
    expert = getattr(getattr(binding.vla, "model", None), "vlm_with_expert", None)
    if expert is None:
        raise RuntimeError("MEMORY_VISUAL_MODEL_ENDPOINT_MISSING")
    cameras = [key for key in binding.last_batch if str(key).startswith("observation.images.")]
    if len(cameras) != 2:
        raise RuntimeError(f"MEMORY_VISUAL_REQUIRES_TWO_CAMERAS:{len(cameras)}")
    before_rng = _rng_digest()
    queue = ProductionBindingAdapter._action_queue(binding.vla)
    queue_before = len(queue)
    slots: dict[str, Any] = {}
    with torch.inference_mode():
        for key in cameras:
            image = binding.last_batch[key]
            if image.ndim != 4 or image.shape[0] != 1:
                raise RuntimeError("MEMORY_VISUAL_SOURCE_MUST_BATCH1")
            repeated = image[:1].expand(4, -1, -1, -1).contiguous().to(device)
            tokens = expert.embed_image(repeated)
            if tokens.ndim != 3 or tokens.shape[0] != 4 or not torch.isfinite(tokens).all():
                raise RuntimeError(f"MEMORY_VISUAL_TOKEN_SHAPE_INVALID:{tuple(tokens.shape)}")
            values = tokens.detach().float().cpu().numpy()
            pooled = values.mean(axis=1, dtype=np.float32)
            slots[key] = {
                "tokens_shape": list(values.shape),
                "tokens_sha256": _sha_bytes(values.tobytes()),
                "mask": [True, True, True, True],
                "pooled_slot0": pooled[0].tolist(),
                "slot_max_abs_diff": float(np.max(np.abs(values[1:] - values[0:1]))),
                "pooled_sha256": _sha_bytes(pooled[0].tobytes()),
            }
            if include_numeric:
                slots[key]["tokens_slot0"] = values[0].copy()
                slots[key]["mask_slot0"] = np.ones(values.shape[1], dtype=np.bool_)
    if len(queue) != queue_before:
        raise RuntimeError("MEMORY_VISUAL_CHANGED_ACTION_QUEUE")
    after_rng = _rng_digest()
    if before_rng != after_rng:
        raise RuntimeError("MEMORY_VISUAL_CHANGED_RNG")
    return {
        "kind": MEMORY_VISUAL_SIDE_CALL,
        "is_action_forward": False,
        "queue_before": queue_before,
        "queue_after": len(queue),
        "cameras": slots,
        "rng_unchanged": True,
    }


def _rng_digest() -> str:
    import random

    import torch

    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    digest.update(repr(np.random.get_state()).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.numpy().tobytes())
    return digest.hexdigest()
