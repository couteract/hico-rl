"""Independent CPU validation of B02Q24 durable pre-step records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class ValidationError(ValueError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValidationError(code)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_numeric(reference: dict) -> np.ndarray:
    require(isinstance(reference, dict) and "path" in reference, "E_NUMERIC_REQUIRED")
    path = Path(reference["path"])
    require(path.is_file(), "E_PAYLOAD_HASH")
    require(path.stat().st_size == reference["size_bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"], "E_PAYLOAD_HASH")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == reference["shape"] and str(value.dtype) == reference["dtype"], "E_PAYLOAD_SHAPE")
    require(np.isfinite(value).all(), "E_PAYLOAD_FINITE")
    return value


def validate_record(record: dict, authority: dict) -> dict:
    try:
        return _validate(record, authority)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValidationError("E_SCHEMA") from error


def _validate(record: dict, authority: dict) -> dict:
    require(record["schema"] == "B02Q24_PRESTEP_V1", "E_SCHEMA")
    require(record["usage"] == "TECHNICAL_PRESTEP_ONLY", "E_USAGE")
    require(record["env_step"] == 0, "E_ENV_STEP")
    require(record["reset_after_restore"] == 0, "E_RESET_AFTER_RESTORE")
    require(not record["outcome_reads"], "E_OUTCOME_READ")
    require(record["selection_rule"] == "PREACTION_FROZEN_ONLY", "E_OUTCOME_SELECTION")
    require(record["estimand"] == "Q_hard^pi_replan_SUBMITTED_ACTION", "E_ESTIMAND")
    require(record["return_claim"] == "ONE_SAMPLE_NOT_EXPECTED_Q", "E_EXPECTED_Q")
    require(record["intervention_steps"] == 1, "E_TWO_STEP")
    require(not record["step_after_terminal"], "E_STEP_AFTER_TERMINAL")
    require(not record["replay_after_advance"], "E_REPLAY_AFTER_ADVANCE")
    require(not record["accepted"], "E_INCOMPLETE_ACCEPTED")
    require(record["controller_effective"]["status"] == "NOT_OBSERVED", "E_EFFECTIVE_CLAIM")
    preaction = record["preaction"]
    require(preaction["physical_identity"] == authority["physical_identity"], "E_PHYSICAL_IDENTITY")
    for key, code in (("snapshot", "E_SNAPSHOT"), ("state8", "E_STATE"), ("visual", "E_VISUAL"),
                      ("task", "E_TASK"), ("raw_base7", "E_BASE"), ("queue", "E_QUEUE"),
                      ("time", "E_TIME"), ("history", "E_HISTORY"), ("memory", "E_MEMORY"),
                      ("model", "E_MODEL"), ("rng", "E_RNG"), ("integration", "E_INTEGRATION")):
        require(preaction[key] == authority[key], code)
    require(preaction["continuation"] == authority["continuation"], "E_CONTINUATION")
    require(preaction["continuation"]["queue_rule"] == "CLEAR_AT_NEXT_FRAME_THEN_REPLAN", "E_OLD_QUEUE")
    require(preaction["boundary"]["recorded"] and preaction["boundary"]["queue_before"] == 0, "E_BOUNDARY")
    require(preaction["remaining_horizon"] > 0, "E_TIME")
    require(preaction["memory"]["fold"] == authority["memory"]["fold"], "E_FOLD")
    require(preaction["memory"]["current_version"] == preaction["boundary"]["decision"], "E_VERSION")
    require(not record["nonempty_history_validated"] or bool(preaction["history"]["events"]), "E_COLD_HISTORY")
    require(canonical_hash(preaction) == record["binding_sha256"], "E_BINDING_HASH")
    visual = preaction["visual"]
    require(visual["stage"] == "MEMORY_VISUAL_REPEAT4_SLOT0_SIDE_CALL" and visual["actor_stage"] == "ACTOR_ACTION_FORWARD_BATCH1_VISUAL", "E_VISUAL_STAGE")
    require(len(visual["cameras"]) == 2, "E_VISUAL")
    pooled = []
    for camera in visual["cameras"]:
        tokens = read_numeric(camera["tokens"])
        mask = read_numeric(camera["mask"])
        mean = read_numeric(camera["pooled"])
        raw = read_numeric(camera["raw"])
        processed = read_numeric(camera["preprocessed"])
        require(tokens.ndim == 2 and mask.shape == tokens.shape[:1] and mask.dtype == np.bool_ and mask.any(), "E_PAYLOAD_SHAPE")
        require(raw.ndim == 4 and processed.ndim == 4 and raw.shape[0] == processed.shape[0] == 1, "E_PAYLOAD_SHAPE")
        require(np.any(tokens) and not visual.get("synthetic_zero_fill", False), "E_ZERO_FILL")
        require(np.allclose(tokens[mask].mean(0), mean, atol=2e-6, rtol=0), "E_POOLING")
        pooled.append(mean)
    require(np.concatenate(pooled).shape == (1920,), "E_PAYLOAD_SHAPE")
    memory = preaction["memory"]
    require(len(memory["event_inputs"]) == len(preaction["history"]["events"]) + 1, "E_HISTORY")
    current = memory["event_inputs"][-1]
    require(current["state8"] == preaction["state8"] and current["visual1920"] == np.concatenate(pooled).tolist(), "E_MEMORY_EVENT")
    require(current["previous_interval"] == preaction["history"]["previous_interval"], "E_HISTORY")
    for key in ("W1", "W8"):
        values = np.asarray(memory[key])
        require(values.shape == (256,) and np.isfinite(values).all() and np.any(values), "E_MEMORY_NUMERIC")
    base = np.asarray(preaction["raw_base7"], np.float32)
    bounds = np.asarray(record["bounds"], np.float32)
    require(base.shape == (7,) and bounds.shape == (2, 6) and np.isfinite(base).all() and np.isfinite(bounds).all(), "E_BASE")
    require(np.all(bounds[0] < bounds[1]), "E_BOUNDS")
    clipped = np.clip(base[:6], *bounds)
    require(np.array_equal(clipped, np.asarray(preaction["clipped_base6"], np.float32)), "E_BASE")
    queue = preaction["queue"]
    require(len(queue["content7"]) == queue["length"] and 0 <= queue["cursor"] <= queue["length"], "E_QUEUE")
    queue_array = np.asarray(queue["content7"])
    require(queue_array.shape == (queue["length"], 7) and np.isfinite(queue_array).all(), "E_QUEUE")
    candidates = record["candidates"]
    require(2 <= len(candidates) <= 4, "E_CANDIDATE_COUNT")
    require(len({item["candidate_id"] for item in candidates}) == len(candidates), "E_DUPLICATE_CANDIDATE")
    minimum = record["specification"]["minimum_difference"]
    require(np.isfinite(minimum) and minimum >= 1e-6, "E_MINIMUM_DIFFERENCE")
    runtime_ids, submitted_values = [], []
    for index, candidate in enumerate(candidates):
        require(candidate["preaction"] == preaction, "E_BRANCH_PREACTION")
        require(candidate["lifecycle"] == "CANDIDATE_READY", "E_PENDING")
        submitted = np.asarray(candidate["submitted7"], np.float32)
        require(submitted.shape == (7,) and np.isfinite(submitted).all(), "E_CANDIDATE_ACTION")
        require(np.all(submitted[:6] >= bounds[0]) and np.all(submitted[:6] <= bounds[1]), "E_OUT_OF_BOUNDS")
        require(submitted[6] == base[6], "E_GRIPPER")
        for previous in submitted_values:
            difference = np.max(np.abs(previous - submitted))
            require(difference != 0, "E_DUPLICATE_CANDIDATE")
            require(difference >= minimum, "E_TOO_CLOSE")
        submitted_values.append(submitted)
        specification = record["specification"]["candidates"][index]
        require(candidate["type"] == specification["type"], "E_CANDIDATE_TYPE")
        requested = np.asarray(candidate["requested_delta6"], np.float32)
        if specification["type"] == "BASE":
            expected_delta = np.zeros(6, np.float32)
        elif specification["type"] == "GEOMETRIC":
            expected_delta = np.asarray(specification["delta6"], np.float32)
        else:
            require(bool(specification["proposal_sha256"]), "E_CANDIDATE_TYPE")
            expected_delta = np.asarray(specification["direction6"], np.float32) * float(specification["amplitude"])
        require(np.array_equal(requested, expected_delta), "E_CANDIDATE_ACTION")
        expected_arm = np.clip(clipped + requested, *bounds)
        require(np.array_equal(submitted[:6], expected_arm) and np.array_equal(np.asarray(candidate["delta6"], np.float32), expected_arm - clipped), "E_CANDIDATE_ACTION")
        require(candidate["candidate_id"] == canonical_hash([record["identity_seed"], record["specification"], index, submitted.tolist()]), "E_CANDIDATE_ID")
        runtime = candidate["runtime"]
        require(runtime["queue_rule"] == "CLEAR_AT_NEXT_FRAME_THEN_REPLAN", "E_OLD_QUEUE")
        require(runtime["binding_sha256"] == record["binding_sha256"] and runtime["isolated"], "E_SHARED_RUNTIME")
        require(runtime["queue_content_sha256"] == canonical_hash(queue["content7"]), "E_QUEUE")
        runtime_ids.append(runtime["mutable_runtime_id"])
    require(len(set(runtime_ids)) == len(runtime_ids), "E_SHARED_RUNTIME")
    return {"status": "PASS", "candidates": len(candidates), "scientific_evidence": False}
