"""B-local single-step submitted-action instrumentation; collection stays disabled."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class InterventionError(ValueError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def technical_specs() -> list[dict]:
    return [{"name": name, "authorization": "TECHNICAL_ONLY_NOT_COLLECTION_APPROVED",
             "minimum_difference": 1e-5,
             "candidates": [{"type": "BASE"}, {"type": "GEOMETRIC", "delta6": [amplitude, 0, 0, 0, 0, 0]},
                            {"type": "GEOMETRIC", "delta6": [-amplitude, 0, 0, 0, 0, 0]}]}
            for name, amplitude in (("small", .01), ("medium", .03), ("large", .05))]


def generate_candidates(raw_base7: Any, bounds: Any, specification: dict, identity_seed: str) -> list[dict]:
    base = np.asarray(raw_base7, dtype=np.float32)
    limits = np.asarray(bounds, dtype=np.float32)
    if base.shape != (7,) or limits.shape != (2, 6) or not np.isfinite(base).all() or not np.isfinite(limits).all():
        raise InterventionError("E_NONFINITE_OR_SHAPE")
    if np.any(limits[0] >= limits[1]):
        raise InterventionError("E_BOUNDS")
    if any(key in specification for key in ("reward", "outcome", "Q", "q_prediction")):
        raise InterventionError("E_OUTCOME_SELECTION")
    minimum = float(specification["minimum_difference"])
    if not np.isfinite(minimum) or minimum < 1e-6:
        raise InterventionError("E_MINIMUM_DIFFERENCE")
    effective = np.clip(base[:6], *limits)
    result = []
    for index, item in enumerate(specification["candidates"]):
        kind = item["type"]
        if kind == "BASE":
            requested = np.zeros(6, dtype=np.float32)
        elif kind == "GEOMETRIC":
            requested = np.asarray(item["delta6"], dtype=np.float32)
        elif kind == "FROZEN_PROPOSAL_DIRECTION" and item.get("proposal_sha256"):
            requested = np.asarray(item["direction6"], dtype=np.float32) * float(item["amplitude"])
        else:
            raise InterventionError("E_CANDIDATE_TYPE")
        if requested.shape != (6,) or not np.isfinite(requested).all():
            raise InterventionError("E_NONFINITE_OR_SHAPE")
        proposed = effective + requested
        arm = np.clip(proposed, *limits)
        submitted = np.concatenate((arm, base[6:]))
        if kind != "BASE" and np.max(np.abs(arm - effective)) < minimum:
            raise InterventionError("E_TOO_CLOSE")
        for previous in result:
            distance = np.max(np.abs(submitted - np.asarray(previous["submitted7"], np.float32)))
            if distance == 0:
                raise InterventionError("E_DUPLICATE_CANDIDATE")
            if distance < minimum:
                raise InterventionError("E_TOO_CLOSE")
        result.append({"candidate_id": digest([identity_seed, specification, index, submitted.tolist()]),
                       "type": kind, "delta6": (arm - effective).tolist(), "requested_delta6": requested.tolist(),
                       "submitted7": submitted.tolist(), "clipped": (arm != proposed).tolist(),
                       "saturated": ((arm == limits[0]) | (arm == limits[1])).tolist(),
                       "base_clipped": (effective != base[:6]).tolist()})
    if not 2 <= len(result) <= 4 or sum(item["type"] == "BASE" for item in result) != 1:
        raise InterventionError("E_CANDIDATE_COUNT")
    return result


def numeric_payload(directory: Path, value: Any) -> dict:
    array = np.ascontiguousarray(value)
    if not np.isfinite(array).all():
        raise InterventionError("E_NONFINITE_OR_SHAPE")
    directory.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()).hexdigest()
    path = directory / (identity + ".npy")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return {"path": str(path.resolve()), "shape": list(array.shape), "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_record(*, preaction: dict, candidates: list[dict], runtimes: list[dict], bounds: Any,
                 specification: dict, identity_seed: str, source: str) -> dict:
    return {"schema": "B02Q24_PRESTEP_V1", "source": source, "usage": "TECHNICAL_PRESTEP_ONLY",
            "preaction": copy.deepcopy(preaction), "binding_sha256": digest(preaction),
            "bounds": np.asarray(bounds).tolist(), "specification": copy.deepcopy(specification),
            "identity_seed": identity_seed, "candidates": [dict(copy.deepcopy(candidate),
            preaction=copy.deepcopy(preaction), runtime=copy.deepcopy(runtime), lifecycle="CANDIDATE_READY")
            for candidate, runtime in zip(candidates, runtimes, strict=True)],
            "selection_rule": "PREACTION_FROZEN_ONLY", "env_step": 0, "reset_after_restore": 0,
            "outcome_reads": [], "intervention_steps": 1, "replay_after_advance": False,
            "terminal": False, "timeout": False, "step_after_terminal": False,
            "accepted": False, "technical_incomplete": False, "return_claim": "ONE_SAMPLE_NOT_EXPECTED_Q",
            "estimand": "Q_hard^pi_replan_SUBMITTED_ACTION", "controller_effective": {"status": "NOT_OBSERVED"},
            "nonempty_history_validated": False}


class BranchStateMachine:
    states = ("PLANNED", "RESTORED", "PREACTION_BOUND", "CANDIDATE_READY", "SUBMITTED_PENDING",
              "STEPPED", "EARLY_REPLAN_BOUNDARY", "CONTINUATION_RUNNING", "ACCEPTED")

    def __init__(self, path: Path, identity: dict, *, mock: bool = False):
        self.path, self.mock = Path(path), bool(mock)
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if self.record["identity"] != identity or self.record["mock"] != self.mock:
                raise InterventionError("E_BRANCH_IDENTITY")
        else:
            self.record = {"identity": copy.deepcopy(identity), "state": "PLANNED", "mock": self.mock,
                           "advanced": False, "terminal": False, "timeout": False, "journal": []}
            atomic_json(self.path, self.record)

    def reset(self) -> None:
        raise InterventionError("E_RESET_AFTER_RESTORE")

    def transition(self, target: str, **evidence: Any) -> None:
        current = self.record["state"]
        if current in ("ACCEPTED", "TECHNICAL_INCOMPLETE"):
            raise InterventionError("E_REPLAY_AFTER_ADVANCE")
        if not self.mock and target not in self.states[:4] + ("TECHNICAL_INCOMPLETE",):
            raise InterventionError("E_PRESTEP_ONLY")
        if target == "SUBMITTED_PENDING" and (self.record["terminal"] or self.record["timeout"]):
            raise InterventionError("E_STEP_AFTER_TERMINAL")
        if target != "TECHNICAL_INCOMPLETE" and self.states.index(target) != self.states.index(current) + 1:
            raise InterventionError("E_STATE_TRANSITION")
        updated = copy.deepcopy(self.record)
        updated.update(state=target)
        if target == "STEPPED":
            updated["advanced"] = True
        updated.update({key: evidence[key] for key in ("terminal", "timeout") if key in evidence})
        if target == "ACCEPTED" and not (updated["terminal"] or updated["timeout"]):
            raise InterventionError("E_OUTCOME_LIFECYCLE")
        if target == "TECHNICAL_INCOMPLETE" and not evidence.get("cause"):
            raise InterventionError("E_FAILURE_EVIDENCE")
        updated["journal"].append({"from": current, "to": target, "evidence": evidence})
        atomic_json(self.path, updated)
        self.record = updated


def early_replan(runtime: dict, action7: Any, next_observation: Any, policy: Any) -> dict:
    action = np.asarray(action7, np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise InterventionError("E_NONFINITE_OR_SHAPE")
    runtime["queue"].clear()
    interval = {"executed_count": 1, "mean7": action.tolist(), "last7": action.tolist(), "end_reason": "early_replan"}
    runtime["history"].append(interval)
    runtime["queue"].extend(policy(next_observation))
    return interval
