"""Isolated CPU-only Memory–VLA–SAC Q-only interface contract.

This module deliberately does not modify the native SAC policy.  It validates the
approved B02Q0R interface: a 6D stochastic arm residual, a frozen VLA gripper,
and a complete 7D submitted action consumed by Q/replay/environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn as nn
from lerobot.policies.sac.modeling_sac import CriticHead, MLP, Policy, TanhMultivariateNormalDiag


ARM_DIM = 6
ACTION_DIM = 7
BETA = 0.1
TARGET_ENTROPY = -3.0
B02Q14P0_POLICY_VERSION = "B02Q14P0_LOG_STD_CLAMP_BEFORE_EXP_V1"
NO_BOOTSTRAP_ON_TERMINAL_OR_TIMEOUT = "NO_BOOTSTRAP_ON_TERMINAL_OR_TIMEOUT"


@dataclass(frozen=True)
class QMemoryInput:
    """Forward-only Q input contract (critic consumes complete action + memory)."""

    actor_output6: np.ndarray
    action7: np.ndarray
    memory: np.ndarray
    memory_version: str

    def validate(self) -> None:
        _vector(self.actor_output6, ARM_DIM, "actor_output6")
        _vector(self.action7, ACTION_DIM, "action7")
        m = np.asarray(self.memory, dtype=np.float32)
        if m.ndim != 1 or not np.isfinite(m).all():
            raise ValueError("memory must be a finite 1D vector")
        if not self.memory_version:
            raise ValueError("memory_version is required")


def actor6_to_critic7(actor_output6: Any, action7: Any, memory: Any, memory_version: str) -> QMemoryInput:
    """Isolated adapter proving actor stochastic 6D and critic action 7D separation.

    The adapter does not concatenate memory into actor features: Q-only conditions
    Q/target-Q on memory while the SAC actor remains memory-independent.
    """
    a = _vector(action7, ACTION_DIM, "action7")
    return QMemoryInput(_vector(actor_output6, ARM_DIM, "actor_output6"), a, np.asarray(memory, dtype=np.float32), memory_version)


def _vector(x: Any, dim: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def compose_action(raw_vla7: Any, u_arm6: Any | None = None, *, beta: float = BETA, proposer: Any | None = None) -> dict[str, Any]:
    """Compose one submitted action without post-composition clipping."""
    raw = _vector(raw_vla7, ACTION_DIM, "raw_vla7")
    if beta == 0.0:
        # Preserve the original VLA path exactly, including out-of-wrapper values.
        return {
            "raw_vla7": raw.copy(),
            "bounded_base_arm6": None,
            "u_arm6": None,
            "submitted_action7": raw.copy(),
            "raw_arm_oob_mask6": (raw[:ARM_DIM] < -1.0) | (raw[:ARM_DIM] > 1.0),
            "submitted_arm_controller_saturation_mask6": (raw[:ARM_DIM] < -1.0) | (raw[:ARM_DIM] > 1.0),
            "gripper_sign": int(np.sign(raw[6])),
            "beta": 0.0,
            "jacobian_dim": 0,
            "log_pi_a_offset": 0.0,
            "residual_proposer_advanced": False,
        }
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must satisfy 0 < beta <= 1")
    if proposer is not None:
        if u_arm6 is not None:
            raise ValueError("provide either u_arm6 or proposer, not both")
        u_arm6 = proposer()
    if u_arm6 is None:
        raise ValueError("u_arm6 is required when beta > 0")
    u = _vector(u_arm6, ARM_DIM, "u_arm6")
    if np.any(u < -1.0) or np.any(u > 1.0):
        raise ValueError("u_arm6 must be bounded in [-1, 1]")
    raw_arm = raw[:ARM_DIM]
    base = np.clip(raw_arm, -1.0, 1.0)
    submitted_arm = (1.0 - beta) * base + beta * u
    submitted = np.concatenate([submitted_arm, raw[6:7]]).astype(np.float32, copy=False)
    # This mask is telemetry for the runtime controller, not an extra clip.
    sat = (submitted_arm < -1.0) | (submitted_arm > 1.0)
    return {
        "raw_vla7": raw.copy(),
        "bounded_base_arm6": base,
        "u_arm6": u.copy(),
        "submitted_action7": submitted.copy(),
        "raw_arm_oob_mask6": (raw_arm < -1.0) | (raw_arm > 1.0),
        "submitted_arm_controller_saturation_mask6": sat,
        "gripper_sign": int(np.sign(raw[6])),
        "beta": float(beta),
        "jacobian_dim": ARM_DIM,
        "log_pi_a_offset": float(-ARM_DIM * np.log(beta)),
        "residual_proposer_advanced": True,
    }


def validate_action_record(record: Mapping[str, Any], *, beta: float = BETA) -> None:
    required = {"raw_vla7", "bounded_base_arm6", "u_arm6", "submitted_action7", "gripper_sign", "beta", "jacobian_dim"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"action record missing fields: {missing}")
    if float(record["beta"]) != float(beta):
        raise ValueError("action record beta mismatch")
    raw = _vector(record["raw_vla7"], ACTION_DIM, "raw_vla7")
    submitted = _vector(record["submitted_action7"], ACTION_DIM, "submitted_action7")
    if beta == 0.0:
        if not np.array_equal(raw, submitted):
            raise ValueError("beta=0 must preserve byte-exact raw VLA action")
        if record.get("residual_proposer_advanced", False) or float(record.get("log_pi_a_offset", 1.0)) != 0.0:
            raise ValueError("beta=0 must not advance residual proposer or compute Jacobian")
        expected_oob = (raw[:ARM_DIM] < -1.0) | (raw[:ARM_DIM] > 1.0)
        if not np.array_equal(record.get("raw_arm_oob_mask6"), expected_oob) or not np.array_equal(record.get("submitted_arm_controller_saturation_mask6"), expected_oob):
            raise ValueError("beta=0 telemetry mask mismatch")
        if int(record.get("gripper_sign")) != int(np.sign(raw[6])):
            raise ValueError("gripper_sign mismatch")
        return
    base = _vector(record["bounded_base_arm6"], ARM_DIM, "bounded_base_arm6")
    u = _vector(record["u_arm6"], ARM_DIM, "u_arm6")
    if not np.array_equal(base, np.clip(raw[:ARM_DIM], -1.0, 1.0)):
        raise ValueError("bounded_base_arm6 is not the approved clip(raw arm)")
    expected = np.concatenate([(1.0 - beta) * base + beta * u, raw[6:7]]).astype(np.float32)
    if not np.array_equal(expected, submitted):
        raise ValueError("submitted action is not the approved 6D composition plus VLA gripper")
    if record.get("jacobian_dim") != ARM_DIM:
        raise ValueError("continuous Jacobian dimension must be 6")
    if not record.get("residual_proposer_advanced", False):
        raise ValueError("beta>0 residual proposer must advance exactly once")
    if np.any(submitted[:ARM_DIM] < -1.0) or np.any(submitted[:ARM_DIM] > 1.0):
        raise ValueError("submitted arm must be bounded without a post-composition clip")
    expected_masks = (raw[:ARM_DIM] < -1.0) | (raw[:ARM_DIM] > 1.0)
    if not np.array_equal(record.get("raw_arm_oob_mask6"), expected_masks):
        raise ValueError("raw_arm_oob_mask6 mismatch")
    submitted_oob = (submitted[:ARM_DIM] < -1.0) | (submitted[:ARM_DIM] > 1.0)
    if not np.array_equal(record.get("submitted_arm_controller_saturation_mask6"), submitted_oob):
        raise ValueError("submitted saturation mask mismatch")
    if int(record.get("gripper_sign")) != int(np.sign(raw[6])):
        raise ValueError("gripper_sign mismatch")
    if not np.isclose(float(record.get("log_pi_a_offset", 0.0)), -ARM_DIM * np.log(beta)):
        raise ValueError("log_pi_a_offset mismatch")


@dataclass(frozen=True)
class Transition:
    """Schema-only transition; no tensors or optimizer state are held."""

    action: dict[str, Any]
    decision_index: int
    queue_offset: int
    replan_marker: str | None
    memory_version_current: str
    memory_version_next: str
    terminated: bool
    truncated: bool
    reward: float
    previous_summary_action: np.ndarray | None

    def validate(self) -> None:
        validate_action_record(self.action, beta=float(self.action["beta"]))
        if self.decision_index < 0 or self.queue_offset < 0:
            raise ValueError("decision_index and queue_offset must be nonnegative")
        if not self.memory_version_current or not self.memory_version_next:
            raise ValueError("current/next memory versions are required")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are distinct flags")
        if self.decision_index == 0:
            if self.previous_summary_action is not None:
                raise ValueError("previous_summary: decision 0 cannot have previous interval summary")
        else:
            summary = np.asarray(self.previous_summary_action, dtype=np.float32)
            if summary.shape != (2 * ACTION_DIM,) or not np.isfinite(summary).all():
                raise ValueError("previous_summary_action must be canonical mean7+last7")


def canonical_action_summary(actions: Any) -> np.ndarray:
    """Return the causal 14D mean7+last7 summary of an executed interval."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM or a.shape[0] == 0 or not np.isfinite(a).all():
        raise ValueError("actions must be a non-empty [frames,7] finite array")
    return np.concatenate([a.mean(axis=0), a[-1]], dtype=np.float32)


def validate_causal_sequence(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate decision/frame ordering and causal previous-interval summaries."""
    if not rows:
        raise ValueError("sequence cannot be empty")
    required = {"episode_id", "chunk_start_frame", "chunk_end_frame", "frame_index", "raw_done", "terminated", "truncated", "memory_reset", "memory_zero"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"sequence missing fields: {sorted(missing)}")
    episode_ids = {r["episode_id"] for r in rows}
    if len(episode_ids) != 1:
        raise ValueError("sequence cannot mix episode identities")
    episode_id = str(rows[0]["episode_id"])
    memory_prefix = f"{episode_id}:"
    for r in rows:
        for field in ("memory_version_current", "memory_version_next"):
            version = r.get(field)
            if not isinstance(version, str) or not version.startswith(memory_prefix):
                raise ValueError(f"{field} must be bound to episode {episode_id}")
    if any(bool(r["terminated"]) and bool(r["truncated"]) for r in rows):
        raise ValueError("terminated and truncated are mutually exclusive")
    for r in rows:
        expected_term = bool(r["raw_done"]) and not bool(r["truncated"])
        if bool(r["terminated"]) != expected_term:
            raise ValueError("terminated must be derived from raw_done and truncated")
        if bool(r["terminated"]) or bool(r["truncated"]):
            if int(r["frame_index"]) != int(rows[-1]["frame_index"]):
                raise ValueError("terminal/timeout must occur on episode final frame")
    frame_indices = [int(r["frame_index"]) for r in rows]
    if frame_indices != sorted(frame_indices) or len(set(frame_indices)) != len(frame_indices):
        raise ValueError("global frame order must be strictly increasing")
    by_decision: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_decision.setdefault(int(row["decision_index"]), []).append(row)
    decisions = list(by_decision)
    if decisions[0] != 0 or decisions != list(range(decisions[-1] + 1)):
        raise ValueError("decision indices must be contiguous from zero")
    checks = []
    prev_summary = None
    previous_boundary_next = None
    for d in decisions:
        group = by_decision[d]
        starts = {int(r["chunk_start_frame"]) for r in group}
        ends = {int(r["chunk_end_frame"]) for r in group}
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("chunk bounds must be constant within decision")
        start, end = starts.pop(), ends.pop()
        if end <= start or end - start != len(group):
            raise ValueError("chunk end-exclusive interval does not match rows")
        allowed_markers = {None, "queue_depleted", "terminal", "timeout", "explicit_replan"}
        if any(r.get("replan_marker") not in allowed_markers for r in group):
            raise ValueError("illegal queue/replan marker")
        frames = [int(r["frame_index"]) for r in group]
        if frames != list(range(start, end)):
            raise ValueError("queue offset/frame sequence is not contiguous")
        actions = np.stack([_vector(r["submitted_action7"], ACTION_DIM, "submitted_action7") for r in group])
        versions = [r.get("memory_version_current") for r in group]
        if len(set(versions)) != 1:
            raise ValueError("memory version advanced within one decision")
        for offset, r in enumerate(group):
            if int(r.get("queue_offset", offset)) != offset:
                raise ValueError("queue offset does not reset/increment canonically")
            expected_next = versions[0] if offset < len(group) - 1 else r.get("memory_version_next")
            if offset < len(group) - 1 and r.get("memory_version_next", versions[0]) != expected_next:
                raise ValueError("next memory advanced before decision boundary")
            if offset < len(group) - 1 and r.get("replan_marker") is not None:
                raise ValueError("boundary marker is only allowed on decision final row")
        final_markers = [r.get("replan_marker") for r in group]
        if any(m in {"terminal", "timeout"} for m in final_markers[:-1]):
            raise ValueError("terminal/timeout marker must be on episode final row")
        if final_markers[-1] == "terminal" and not group[-1]["terminated"]:
            raise ValueError("terminal marker disagrees with terminated flag")
        if final_markers[-1] == "timeout" and not group[-1]["truncated"]:
            raise ValueError("timeout marker disagrees with truncated flag")
        if bool(group[-1]["terminated"]) and final_markers[-1] != "terminal":
            raise ValueError("final terminated row requires terminal marker")
        if bool(group[-1]["truncated"]) and final_markers[-1] != "timeout":
            raise ValueError("final truncated row requires timeout marker")
        if d > 0 and previous_boundary_next != versions[0]:
            raise ValueError("boundary next-memory does not match next decision current-memory")
        supplied_values = [r.get("previous_summary_action") for r in group]
        if d == 0 and any(v is not None for v in supplied_values):
            raise ValueError("decision 0 must reset previous summary on every row")
        if d > 0 and any(not np.array_equal(np.asarray(v, dtype=np.float32), np.asarray(supplied_values[0], dtype=np.float32)) for v in supplied_values):
            raise ValueError("previous summary differs within decision")
        supplied = supplied_values[0]
        if d == 0:
            if supplied is not None:
                raise ValueError("decision 0 must reset previous summary")
        elif not np.array_equal(np.asarray(supplied, dtype=np.float32), prev_summary):
            raise ValueError("previous interval summary is not k-1 executed interval")
        if d == 0:
            if not bool(group[0]["memory_reset"]):
                raise ValueError("decision 0 must carry explicit memory reset")
            for r in group:
                mz = np.asarray(r["memory_zero"], dtype=np.float32)
                if mz.ndim != 1 or mz.size == 0 or not np.all(np.isfinite(mz)) or not np.array_equal(mz, np.zeros_like(mz)):
                    raise ValueError("decision 0 memory zero/reset tensor is invalid")
        elif any(bool(r.get("memory_reset")) for r in group):
            raise ValueError("memory reset may only occur at episode start")
        prev_summary = canonical_action_summary(actions)
        previous_boundary_next = group[-1].get("memory_version_next", versions[0])
        checks.append({"decision_index": d, "frames": len(group), "memory_version": group[0].get("memory_version_current"), "summary_bound": d > 0})
    return {"decisions": len(decisions), "checks": checks, "episode_reset": True, "causal": True}


def bellman_mask(terminated: bool, truncated: bool, *, contract: str = NO_BOOTSTRAP_ON_TERMINAL_OR_TIMEOUT) -> float:
    if contract != NO_BOOTSTRAP_ON_TERMINAL_OR_TIMEOUT:
        raise ValueError(f"unsupported termination contract: {contract}")
    if terminated or truncated:
        return 0.0
    return 1.0


def latent_entropy_contract(log_pi_u: float, *, beta: float = BETA) -> dict[str, float]:
    if not np.isfinite(log_pi_u):
        raise ValueError("log_pi_u must be finite")
    if not (0.0 < beta <= 1.0):
        raise ValueError("beta must satisfy 0 < beta <= 1 for executed-action telemetry")
    return {
        "log_pi_u": float(log_pi_u),
        "target_entropy": TARGET_ENTROPY,
        "log_pi_a_telemetry": float(log_pi_u - ARM_DIM * np.log(beta)),
        "log_pi_a_used_in_loss": 0.0,
    }


def validate_identity_hashes(actual: Mapping[str, str], expected: Mapping[str, str]) -> None:
    for key, digest in expected.items():
        if actual.get(key) != digest:
            raise ValueError(f"identity hash mismatch: {key}")


def validate_path_manifest(manifest: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """Rehash every manifest path and fail closed on path or byte drift."""
    actual = {}
    for key, entry in manifest.items():
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"identity path missing: {key}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"identity file hash mismatch: {key}")
        actual[key] = digest
    return actual


class TorchQOnlyCritic(nn.Module):
    """Isolated twin-Q adapter using the native CriticHead implementation."""
    def __init__(self, observation_dim: int, memory_dim: int, hidden_dims: list[int] | None = None, num_critics: int = 2):
        super().__init__()
        hidden_dims = hidden_dims or [32, 32]
        input_dim = observation_dim + memory_dim + ACTION_DIM
        self.critics = nn.ModuleList([CriticHead(input_dim=input_dim, hidden_dims=hidden_dims, activate_final=True) for _ in range(num_critics)])

    def forward(self, observation_features: torch.Tensor, memory: torch.Tensor, action7: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observation_features, memory, action7], dim=-1)
        return torch.stack([head(x).squeeze(-1) for head in self.critics], dim=0)


def build_torch_policy6(observation_dim: int, *, seed: int = 0) -> Policy:
    """Build the native SAC Policy with a six-dimensional stochastic output."""
    torch.manual_seed(seed)
    # Policy requires an encoder attribute, but this isolated path uses a tiny
    # identity encoder and exercises the native tanh distribution path.
    class IdentityEncoder(nn.Module):
        def forward(self, observations, cache=None, detach=False):
            return observations
    enc = IdentityEncoder()
    return Policy(enc, MLP(observation_dim, [32, 32], activate_final=True), ARM_DIM,
                   std_min=-5, std_max=2, init_final=0.05, use_tanh_squash=True)


class BLogStdPolicy6(Policy):
    """Future B-only SAC actor; state_dict keys match the historical Policy6.

    The historical builder above remains unchanged for Q12R interpretation.
    """

    implementation_version = B02Q14P0_POLICY_VERSION

    def __init__(self, encoder: nn.Module, network: nn.Module):
        super().__init__(encoder, network, ARM_DIM, std_min=-5, std_max=2,
                         init_final=0.05, use_tanh_squash=True)
        if not self.use_tanh_squash or self.fixed_std is not None:
            raise ValueError("B policy requires trainable log_std and tanh squash")

    def distribution_parameters(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations, cache=None, detach=self.encoder_is_shared)
        outputs = self.network(features)
        mean = self.mean_layer(outputs)
        log_std = self.std_layer(outputs).clamp(self.std_min, self.std_max)
        std = log_std.exp()
        if mean.ndim != 2 or mean.shape[-1] != ARM_DIM or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("B Policy6 requires finite [batch,6] mean and std")
        return mean, std

    def forward(self, observations: torch.Tensor, observation_features: torch.Tensor | None = None):
        if not self.use_tanh_squash or self.fixed_std is not None or observation_features is not None:
            raise ValueError("B Policy6 requires tanh squash, learned std, and actor-only observations")
        mean, std = self.distribution_parameters(observations)
        distribution = TanhMultivariateNormalDiag(loc=mean, scale_diag=std)
        action = distribution.rsample()
        log_pi_u = distribution.log_prob(action)
        if not torch.isfinite(action).all() or not torch.isfinite(log_pi_u).all():
            raise ValueError("nonfinite B latent action or log probability")
        return action, log_pi_u, mean

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluation action with no sampling or RNG consumption."""
        mean, _ = self.distribution_parameters(observations)
        return mean.tanh()


def build_b02q14p0_policy6(observation_dim: int, *, seed: int = 0) -> BLogStdPolicy6:
    """Build the versioned future policy; no historical Q12R entry uses this."""
    torch.manual_seed(seed)

    class IdentityEncoder(nn.Module):
        def forward(self, observations, cache=None, detach=False):
            return observations

    return BLogStdPolicy6(IdentityEncoder(), MLP(observation_dim, [32, 32], activate_final=True))
