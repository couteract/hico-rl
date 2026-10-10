"""Causal memory providers for B02M0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from lerobot.memory.himem import HiMemEncoder, RollingWindowStateManager
from .dataset import REASONS


@dataclass
class MemoryOutput:
    memory_features: torch.Tensor
    memory_valid: torch.Tensor
    window_lengths: torch.Tensor
    audit: list[dict[str, Any]]


class VisualAdapter(nn.Module):
    """Trainable pooled-actor visual adapter; deltas live in adapter space."""

    def __init__(self, input_dim: int = 1920, output_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.shape[-2:] != (2, 960):
            raise ValueError(f"pooled_visual must end in [2,960], got {tuple(pooled.shape)}")
        return self.norm(self.linear(pooled.reshape(*pooled.shape[:-2], 1920)))


def expected_event_observable(reason: str | None, count: int | None, horizon: int, device: torch.device) -> torch.Tensor:
    event = torch.zeros(32, dtype=torch.float32, device=device)
    if reason is None:
        if count is not None:
            raise ValueError("decision 0 previous executed count must be None")
        return event
    if reason not in REASONS:
        raise ValueError(f"unexpected previous reason: {reason}")
    if count is None or count <= 0:
        raise ValueError("non-first decision requires positive previous executed count")
    event[REASONS.index(reason)] = 1.0
    event[5] = float(count) / float(horizon)
    return event


def validate_causal_row(row: dict[str, Any], action_horizon: int, device: torch.device) -> torch.Tensor:
    """Validate a public causal row and return its canonical event vector."""
    allowed = {
        "member", "episode_id", "env_id", "decision_index", "decision_start_frame",
        "prev_executed_start", "prev_executed_end", "prev_executed_count",
        "previous_reason", "previous_action", "previous_visual_feature", "previous_state",
        "previous_decision_index", "visual_feature", "state", "event_observable",
    }
    extra = set(row) - allowed
    if extra:
        raise ValueError(f"forbidden model input fields: {sorted(extra)}")
    idx = int(row["decision_index"])
    if row.get("previous_decision_index") != (None if idx == 0 else idx - 1):
        raise ValueError(f"previous decision source mismatch at decision {idx}")
    previous = ("previous_visual_feature", "previous_state", "previous_action")
    if idx == 0:
        if any(row.get(k) is not None for k in previous):
            raise ValueError("decision 0 previous fields must be None")
        if row.get("previous_reason") is not None or row.get("prev_executed_start") is not None or row.get("prev_executed_end") is not None or row.get("prev_executed_count") is not None:
            raise ValueError("decision 0 previous metadata must be None")
    elif any(row.get(k) is None for k in previous):
        raise ValueError(f"decision {idx} missing strict k-1 previous fields")
    expected = expected_event_observable(row.get("previous_reason"), row.get("prev_executed_count"), action_horizon, device)
    actual = row["event_observable"].to(device=device, dtype=torch.float32)
    if actual.shape != (32,) or not torch.isfinite(actual).all() or not torch.equal(actual, expected):
        raise ValueError(f"event_observable does not match frozen schema at decision {idx}")
    return expected


class Pistar06MemoryProvider(nn.Module):
    """Memory provider using the canonical HiMem W8 zero-recompute path."""

    def __init__(self, encoder: HiMemEncoder, action_horizon: int = 5) -> None:
        super().__init__()
        if encoder.config.memory_dim != 256:
            raise ValueError("B02M0 requires a 256D HiMem output")
        self.encoder = encoder
        self.adapter = VisualAdapter()
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._online: dict[str, RollingWindowStateManager] = {}

    @property
    def schema(self):
        return self.encoder.config.token_schema

    def _encode_rows(self, rows: list[dict[str, Any]], device: torch.device) -> list:
        from lerobot.memory.himem import EventFeatureEncoder

        enc = EventFeatureEncoder(self.schema)
        out = []
        previous_visual = None
        previous_state = None
        for row in rows:
            event = validate_causal_row(row, self.action_horizon, device)
            idx = int(row["decision_index"])
            visual_raw = row["visual_feature"].to(device=device, dtype=torch.float32)
            visual = self.adapter(visual_raw)
            state = row["state"].to(device=device, dtype=torch.float32)
            previous_action = row["previous_action"].to(device=device, dtype=torch.float32) if row["previous_action"] is not None else None
            # A window may begin at k>0. Use the cache's explicit k-1 fields
            # for its first row; subsequent rows use the preceding row here.
            if previous_visual is None and row.get("previous_visual_feature") is not None:
                previous_visual = self.adapter(row["previous_visual_feature"].to(device=device, dtype=torch.float32))
            if previous_state is None and row.get("previous_state") is not None:
                previous_state = row["previous_state"].to(device=device, dtype=torch.float32)
            token = enc.encode_token(
                visual_feature=visual,
                state=state,
                prev_action_summary=previous_action,
                prev_visual_feature=previous_visual,
                prev_state=previous_state,
                event_observable=event,
                episode_id=row["episode_id"],
                env_id=row["env_id"],
                decision_index=int(row["decision_index"]),
                decision_start_frame=int(row["decision_start_frame"]),
                prev_executed_start=row["prev_executed_start"],
                prev_executed_end=row["prev_executed_end"],
                prev_executed_count=row["prev_executed_count"],
                end_reason=row["previous_reason"],
                schema_hash=self.schema.compute_hash(),
            )
            out.append(token)
            previous_visual = visual
            previous_state = state
        return out

    def offline(self, rows: list[dict[str, Any]], device: Optional[torch.device] = None) -> MemoryOutput:
        if not rows:
            raise ValueError("rows must be non-empty")
        device = device or next(self.parameters()).device
        features = []
        lengths = []
        audits = []
        for target in range(len(rows)):
            start = max(0, target - int(self.encoder.config.window_size) + 1)
            window_rows = rows[start : target + 1]
            tokens = self._encode_rows(window_rows, device)
            memory = self.encoder.forward_from_tokens(tokens)
            features.append(memory.float())
            lengths.append(len(tokens))
            audits.append({"episode_id": rows[target]["episode_id"], "member": rows[target]["member"], "decision_index": rows[target]["decision_index"], "window_start_decision": rows[start]["decision_index"], "window_end_decision": rows[target]["decision_index"]})
        return MemoryOutput(torch.stack(features), torch.ones(len(features), dtype=torch.bool, device=device), torch.tensor(lengths, dtype=torch.int64, device=device), audits)

    def reset_env(self, env_id: str) -> None:
        self._online.pop(env_id, None)

    def observe_decision(self, row: dict[str, Any], env_id: Optional[str] = None) -> MemoryOutput:
        env_id = env_id or row["env_id"]
        manager = self._online.get(env_id)
        if manager is None:
            manager = self.encoder.create_state_manager()
            self._online[env_id] = manager
        tokens = self._encode_rows([row], next(self.parameters()).device)
        manager.add_token(tokens[0])
        window = manager.get_recent_tokens(env_id)
        memory = self.encoder.forward_from_tokens(window)
        idx = int(row["decision_index"])
        return MemoryOutput(memory[None].float(), torch.ones(1, dtype=torch.bool, device=memory.device), torch.tensor([len(window)], dtype=torch.int64, device=memory.device), [{"episode_id": row["episode_id"], "member": row["member"], "decision_index": idx, "window_start_decision": window[0].audit_metadata["decision_index"], "window_end_decision": idx}])

    def forward(self, batch_rows: list[list[dict[str, Any]]], device: Optional[torch.device] = None) -> MemoryOutput:
        outputs = [self.offline(rows, device=device) for rows in batch_rows]
        return MemoryOutput(torch.stack([x.memory_features[-1] for x in outputs]), torch.ones(len(outputs), dtype=torch.bool, device=outputs[0].memory_features.device), torch.tensor([int(x.window_lengths[-1]) for x in outputs], dtype=torch.int64, device=outputs[0].memory_features.device), [x.audit[-1] for x in outputs])


class NoMemoryProvider(nn.Module):
    """Shape-compatible deterministic memoryless provider."""

    def __init__(self, memory_dim: int = 256, action_horizon: int = 5) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._online: dict[str, tuple[str, int]] = {}

    def reset_env(self, env_id: str) -> None:
        self._online.pop(env_id, None)

    def observe_decision(self, row: dict[str, Any], env_id: Optional[str] = None) -> MemoryOutput:
        env_id = env_id or row["env_id"]
        validate_causal_row(row, action_horizon=self.action_horizon, device=torch.device("cpu"))
        episode_id, decision_index = row["episode_id"], int(row["decision_index"])
        current = self._online.get(env_id)
        if current is None:
            if decision_index != 0:
                raise ValueError(f"first decision for {env_id} must be 0")
        else:
            if current[0] != episode_id:
                raise ValueError(f"env {env_id} requires reset before episode change")
            if decision_index != current[1] + 1:
                raise ValueError(f"non-chronological decision for {env_id}: {decision_index}")
        self._online[env_id] = (episode_id, decision_index)
        device = row["state"].device
        return MemoryOutput(
            torch.zeros((1, self.memory_dim), dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.bool, device=device),
            torch.tensor([min(8, decision_index + 1)], dtype=torch.int64, device=device),
            [{"episode_id": episode_id, "member": row["member"], "decision_index": decision_index,
              "window_start_decision": max(0, decision_index - 7), "window_end_decision": decision_index}],
        )

    def forward(self, batch_rows: list[list[dict[str, Any]]], device: Optional[torch.device] = None) -> MemoryOutput:
        if not batch_rows:
            raise ValueError("batch_rows must be non-empty")
        device = device or torch.device("cpu")
        audits = []
        lengths = []
        for rows in batch_rows:
            if not rows:
                raise ValueError("empty episode rows")
            row = rows[-1]
            validate_causal_row(row, action_horizon=self.action_horizon, device=device)
            audits.append({"episode_id": row["episode_id"], "member": row["member"], "decision_index": row["decision_index"], "window_start_decision": max(0, int(row["decision_index"]) - 7), "window_end_decision": row["decision_index"]})
            lengths.append(min(8, len(rows)))
        return MemoryOutput(torch.zeros((len(batch_rows), self.memory_dim), dtype=torch.float32, device=device), torch.zeros(len(batch_rows), dtype=torch.bool, device=device), torch.tensor(lengths, dtype=torch.int64, device=device), audits)
