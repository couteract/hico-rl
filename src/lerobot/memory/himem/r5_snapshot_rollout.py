"""One-shot Q20R5 snapshot restoration for the local Q21R rollout path."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

os.environ.update(
    MUJOCO_GL="osmesa",
    PYOPENGL_PLATFORM="osmesa",
    LIBGL_ALWAYS_SOFTWARE="1",
    GALLIUM_DRIVER="llvmpipe",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode(value: Any) -> Any:
    kind = value["kind"]
    if kind == "ndarray":
        return np.frombuffer(
            base64.b64decode(value["bytes"]), dtype=np.dtype(value["dtype"])
        ).reshape(value["shape"]).copy()
    if kind == "bytes":
        return base64.b64decode(value["bytes"])
    if kind == "tuple":
        return tuple(_decode(item) for item in value["items"])
    if kind == "list":
        return [_decode(item) for item in value["items"]]
    if kind == "dict":
        return {key: _decode(item) for key, item in value["items"]}
    return value.get("value")


def _same(expected: Any, actual: Any) -> bool:
    left = np.asarray(expected)
    right = np.asarray(actual)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def validate_snapshot_contract(
    meta: dict[str, Any],
    *,
    task: int,
    physical_group_key: str,
    route_seed: int | None = None,
    expected_route_seed: int | None = None,
    slot: int | None = None,
    expected_slot: int | None = None,
    initial_frame: int = 0,
    initial_queue_length: int = 0,
    confirmation: bool = False,
    reserve: bool = False,
    restore_sequence: tuple[str, ...] | None = None,
) -> None:
    """Validate immutable identity and pre-step state before construction."""
    if int(meta["canonical_task_id"]) != int(task):
        raise ValueError("SNAPSHOT_TASK_ID_MISMATCH")
    if meta["physical_group_key"] != str(physical_group_key):
        raise ValueError("SNAPSHOT_PHYSICAL_GROUP_MISMATCH")
    if int(meta["env_step"]) != 0 or int(meta["policy_action"]) != 0 or int(meta["reward_outcome"]) != 0:
        raise ValueError("SNAPSHOT_NONZERO_COUNTER")
    if route_seed is not None and expected_route_seed is not None and int(route_seed) != int(expected_route_seed):
        raise ValueError("ROUTE_SEED_MISMATCH")
    if slot is not None and expected_slot is not None and int(slot) != int(expected_slot):
        raise ValueError("SNAPSHOT_SLOT_MISMATCH")
    if int(initial_frame) != 0:
        raise ValueError("INITIAL_FRAME_MUST_BE_ZERO")
    if int(initial_queue_length) != 0:
        raise ValueError("INITIAL_QUEUE_MUST_BE_EMPTY")
    if confirmation:
        raise ValueError("CONFIRMATION_SLOT_FORBIDDEN")
    if reserve:
        raise ValueError("RESERVE_SLOT_FORBIDDEN")
    if restore_sequence is not None and tuple(restore_sequence) != (
        "construct_xml",
        "restore_runtime_arrays",
        "restore_integration",
        "mj_forward",
        "restore_rng",
        "read_observation",
    ):
        raise ValueError("RESTORE_SEQUENCE_INVALID")


def validate_snapshot_payload(
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    *,
    restore_results: dict[str, Any] | None = None,
    rng_payload_bytes: bytes | None = None,
    expected_rng_payload_sha256: str | None = None,
) -> None:
    """Validate immutable payload fields before handing them to MuJoCo."""
    allowlist = list(meta["model_state_allowlist"])
    if sorted(arrays) != sorted(allowlist):
        raise ValueError("SNAPSHOT_MODEL_ALLOWLIST_MISMATCH")
    if "site_rgba" not in arrays:
        raise ValueError("SNAPSHOT_SITE_RGBA_MISSING")
    if "integration" not in state:
        raise ValueError("SNAPSHOT_INTEGRATION_MISSING")
    if restore_results is not None:
        # Restore evidence may be supplied as an external, already byte-bound
        # attestation whose fields are at the document top level. Legacy local
        # files retain their entries[0].fields shape.
        fields = restore_results.get("fields")
        if fields is None:
            fields = restore_results.get("entries", [{}])[0].get("fields", {})
        expected = fields.get("integration")
        integration = np.asarray(state["integration"])
        if expected is not None:
            if list(integration.shape) != expected["shape"] or np.dtype(integration.dtype).str != expected["dtype"]:
                raise ValueError("SNAPSHOT_INTEGRATION_SHAPE_OR_DTYPE")
            if _sha_bytes(integration.tobytes()) != expected["expected_sha256"]:
                raise ValueError("SNAPSHOT_INTEGRATION_HASH_MISMATCH")
        for name, value in arrays.items():
            expected = fields.get(f"model_array:{name}")
            if expected is None:
                raise ValueError(f"SNAPSHOT_MODEL_FIELD_UNEXPECTED:{name}")
            if list(value.shape) != expected["shape"] or np.dtype(value.dtype).str != expected["dtype"]:
                raise ValueError(f"MODEL_FIELD_SHAPE_OR_DTYPE:{name}")
            if _sha_bytes(value.tobytes()) != expected["expected_sha256"]:
                raise ValueError(f"MODEL_FIELD_HASH_MISMATCH:{name}")
    if expected_rng_payload_sha256 is not None:
        if rng_payload_bytes is None or _sha_bytes(rng_payload_bytes) != expected_rng_payload_sha256:
            raise ValueError("SNAPSHOT_RNG_PAYLOAD_HASH_MISMATCH")


class R5SnapshotRolloutAdapter:
    """A one-shot restored frame driver with A21-compatible methods."""

    def __init__(
        self,
        snapshot: str | Path,
        *,
        task: int,
        physical_group_key: str,
        route_seed: int,
    ):
        self.snapshot = Path(snapshot).resolve()
        self.task = int(task)
        self.physical_group_key = str(physical_group_key)
        self.route_seed = int(route_seed)
        self.env: Any | None = None
        self.model: Any | None = None
        self.data: Any | None = None
        self._restored = False
        self._closed = False
        self.frame = 0
        self.decision = 0
        self.queue: list[np.ndarray] = []
        self.pending: Any | None = None
        self.counters = {
            "env_construct": 0,
            "reset": 0,
            "env_step": 0,
            "policy": 0,
            "reward_outcome": 0,
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: str | Path,
        *,
        task: int,
        physical_group_key: str,
        route_seed: int,
    ) -> "R5SnapshotRolloutAdapter":
        instance = cls(
            snapshot,
            task=task,
            physical_group_key=physical_group_key,
            route_seed=route_seed,
        )
        instance._restore_once()
        return instance

    def _restore_once(self) -> None:
        if self._restored:
            raise RuntimeError("SNAPSHOT_RESTORE_ALREADY_PERFORMED")
        import mujoco
        from libero.libero.envs.env_wrapper import ControlEnv
        from robosuite.environments.base import MujocoEnv

        meta = json.loads((self.snapshot / "meta.json").read_text())
        state = dict(np.load(self.snapshot / "state.npz", allow_pickle=False))
        arrays = dict(np.load(self.snapshot / "runtime_model_arrays.npz", allow_pickle=False))
        payload = json.loads((self.snapshot / "rng_payload.json").read_text())
        xml = (self.snapshot / "model.xml").read_text()
        restore_results = json.loads((self.snapshot / "restore_results.json").read_text())
        validate_snapshot_contract(
            meta,
            task=self.task,
            physical_group_key=self.physical_group_key,
            initial_frame=self.frame,
            initial_queue_length=len(self.queue),
        )
        if _sha_bytes(xml.encode()) != meta["model_sha256"]:
            raise ValueError("SNAPSHOT_XML_HASH_MISMATCH")
        validate_snapshot_payload(
            meta,
            arrays,
            state,
            restore_results=restore_results,
        )

        original_initialize = MujocoEnv._initialize_sim

        def initialize(instance: Any, xml_string: str | None = None) -> Any:
            return original_initialize(instance, xml_string=xml)

        MujocoEnv._initialize_sim = initialize
        try:
            self.env = ControlEnv(
                meta["bddl_path"],
                use_camera_obs=True,
                has_renderer=False,
                has_offscreen_renderer=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=32,
                camera_widths=32,
                initialization_noise=None,
            )
        finally:
            MujocoEnv._initialize_sim = original_initialize
        self.counters["env_construct"] += 1
        self.model, self.data = self.env.sim.model._model, self.env.sim.data._data
        for name, value in arrays.items():
            target = np.asarray(getattr(self.model, name))
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"MODEL_FIELD_SHAPE_OR_DTYPE:{name}")
            target[...] = value
        mujoco.mj_setState(self.model, self.data, state["integration"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        random.setstate(_decode(payload["python_random"]))
        np.random.set_state(_decode(payload["numpy_legacy"]))
        self.observation = self._read_observation()
        self._restored = True

    def _read_observation(self) -> dict[str, np.ndarray]:
        if self.env is None:
            raise RuntimeError("SNAPSHOT_NOT_RESTORED")
        raw = self.env.env._get_observations(force_update=True)
        return {
            "agentview": np.asarray(raw["agentview_image"]).copy(),
            "eye": np.asarray(raw["robot0_eye_in_hand_image"]).copy(),
            "state8": np.asarray(raw["robot0_proprio-state"])[:8].copy(),
        }

    def observe(self) -> dict[str, np.ndarray]:
        if not self._restored or self._closed:
            raise RuntimeError("ADAPTER_NOT_ACTIVE")
        return {key: value.copy() for key, value in self.observation.items()}

    def pop_vla_action(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        if not self._restored or self._closed:
            raise RuntimeError("ADAPTER_NOT_ACTIVE")
        if not self.queue:
            raise RuntimeError("VLA_ACTION_QUEUE_EMPTY")
        action = self.queue.pop(0).copy()
        queue_before = len(self.queue) + 1
        queue_after = len(self.queue)
        self.counters["policy"] += 1
        return action[:8], action[:7], queue_before, queue_after

    def enqueue_vla_action(self, action8: Any) -> None:
        value = np.asarray(action8, dtype=np.float32)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise ValueError("VLA_ACTION_MUST_BE_FINITE_SHAPE8")
        self.queue.append(value.copy())

    def step(self, action7: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if not self._restored or self._closed:
            raise RuntimeError("ADAPTER_NOT_ACTIVE")
        value = np.asarray(action7, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("ENV_ACTION_MUST_BE_FINITE_SHAPE7")
        result = self.env.step(value)
        self.counters["env_step"] += 1
        self.frame += 1
        self.observation = self._read_observation()
        return result

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.counters["reset"] += 1
        raise RuntimeError("ORDINARY_RESET_FORBIDDEN_AFTER_R5_RESTORE")

    def restore(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SECONDARY_RESTORE_FORBIDDEN")

    def close(self) -> None:
        if self.env is not None and not self._closed:
            self.env.close()
        self._closed = True

    def evidence(self) -> dict[str, Any]:
        return {
            "snapshot": str(self.snapshot),
            "task": self.task,
            "physical_group_key": self.physical_group_key,
            "route_seed": self.route_seed,
            "restored": self._restored,
            "frame": self.frame,
            "decision": self.decision,
            "queue_length": len(self.queue),
            "pending": self.pending is not None,
            "counters": dict(self.counters),
        }


def compare_snapshot(adapter: R5SnapshotRolloutAdapter) -> dict[str, Any]:
    raw = dict(np.load(adapter.snapshot / "state.npz", allow_pickle=False))
    meta = json.loads((adapter.snapshot / "meta.json").read_text())
    fields = {
        "qpos": _same(raw["qpos"], adapter.data.qpos),
        "qvel": _same(raw["qvel"], adapter.data.qvel),
        "time": _same(raw["time"], np.asarray([adapter.data.time])),
        "agentview": _same(raw["agentview"], adapter.observation["agentview"]),
        "eye": _same(raw["eye"], adapter.observation["eye"]),
        "state8": _same(raw["state8"], adapter.observation["state8"]),
    }
    return {
        "status": "PASS" if all(fields.values()) else "FAIL",
        "fields": fields,
        "model_allowlist": meta["model_state_allowlist"],
        "counters": adapter.evidence()["counters"],
    }


__all__ = [
    "R5SnapshotRolloutAdapter",
    "compare_snapshot",
    "validate_snapshot_contract",
    "validate_snapshot_payload",
]
