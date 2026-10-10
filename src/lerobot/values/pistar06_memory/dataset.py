"""Read-only B02C1 cache loader and causal transition construction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


CAMERAS = ["observation.images.image", "observation.images.wrist_image"]
REASONS = ["queue_depleted", "early_replan", "terminal", "truncated", "timeout"]
EXPECTED_DTYPES = {
    "decision_index": torch.int64,
    "chunk_start_frame": torch.int64,
    "visual_tokens": torch.float32,
    "visual_valid_mask": torch.bool,
    "pooled_visual": torch.float32,
    "visual_delta": torch.float32,
    "observation_state": torch.float32,
    "state_delta": torch.float32,
    "has_previous": torch.bool,
    "previous_action_mean7": torch.float32,
    "previous_action_last7": torch.float32,
    "previous_executed_action_count": torch.int64,
}


@dataclass
class CacheEpisode:
    member: str
    decision_count: int
    sidecar: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    overlay: dict[int, dict[str, Any]]
    shard_hash: str
    sidecar_hash: str


@dataclass
class TransitionSample:
    member: str
    decision_index: int
    rows: list[dict[str, Any]]
    target_state_delta: torch.Tensor
    target_executed_action: torch.Tensor


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _event(reason: str | None, count: int | None, horizon: int) -> torch.Tensor:
    out = torch.zeros(32, dtype=torch.float32)
    if reason is not None:
        if reason not in REASONS:
            raise ValueError(f"unexpected previous reason: {reason}")
        out[REASONS.index(reason)] = 1.0
        out[5] = float(count) / float(horizon)
    return out


def _runtime_horizon(recorded_root: Path, members: list[str]) -> tuple[int, dict[str, dict[str, Any]]]:
    """Bind the horizon to the recorded runtime overrides, never actor config."""
    bindings: dict[str, dict[str, Any]] = {}
    horizons = set()
    for member in members:
        manifest = recorded_root / member / "ONLINE_COLLECTION_MANIFEST.json"
        data = json.loads(manifest.read_text())
        runtime = data.get("runtime_overrides", {})
        chunk_size = runtime.get("chunk_size")
        n_action_steps = runtime.get("n_action_steps")
        if not isinstance(chunk_size, int) or not isinstance(n_action_steps, int) or chunk_size <= 0 or n_action_steps <= 0:
            raise ValueError(f"invalid runtime action horizon for {member}")
        if chunk_size != n_action_steps:
            raise ValueError(f"runtime chunk_size/n_action_steps mismatch for {member}: {chunk_size}/{n_action_steps}")
        horizons.add(chunk_size)
        bindings[member] = {"path": str(manifest), "sha256": sha(manifest), "chunk_size": chunk_size, "n_action_steps": n_action_steps}
    if len(horizons) != 1:
        raise ValueError(f"runtime action horizons are inconsistent: {sorted(horizons)}")
    return horizons.pop(), bindings


def load_cache_dataset(old_run: Path, overlay_path: Path, expected_inventory_sha: str | None = None, expected_r_run_sha: str | None = None, action_horizon: int | None = None, recorded_root: Path | None = None) -> "CacheDataset":
    old_run = Path(old_run)
    index_path = old_run / "CACHE_INDEX.json"
    index = json.loads(index_path.read_text())
    if expected_inventory_sha is not None and sha(old_run / "ARTIFACT_INVENTORY.json") != expected_inventory_sha:
        raise ValueError("B02C1 inventory hash mismatch")
    overlay_doc = json.loads(Path(overlay_path).read_text())
    if expected_r_run_sha is not None and sha(Path(overlay_path).parent / "ARTIFACT_INVENTORY.json") != expected_r_run_sha:
        raise ValueError("B02C1R inventory hash mismatch")
    overlay = {(x["member"], int(x["decision_index"])): x for x in overlay_doc["entries"]}
    if len(overlay) != len(overlay_doc["entries"]):
        raise ValueError("duplicate overlay member/decision")
    entries = index["entries"]
    members = [entry["member"] for entry in entries]
    if len(set(members)) != len(members):
        raise ValueError("duplicate member in cache index")
    if recorded_root is None:
        recorded_root = old_run.parents[1] / "B02C0" / "run_20260907_110716_b02c0_final" / "recorded_dataset"
    actual_horizon, horizon_bindings = _runtime_horizon(Path(recorded_root), members)
    if action_horizon is not None and int(action_horizon) != actual_horizon:
        raise ValueError(f"requested action horizon {action_horizon} != recorded runtime horizon {actual_horizon}")
    episodes = []
    for entry in entries:
        member = entry["member"]
        if member == "test_ep0":
            raise ValueError("test_ep0 is forbidden")
        shard = Path(entry["shard_path"])
        sidecar = Path(entry["sidecar_path"])
        if sha(shard) != entry["shard_sha256"] or sha(sidecar) != entry["sidecar_sha256"]:
            raise ValueError(f"cache hash mismatch for {member}")
        meta = json.loads(sidecar.read_text())
        if meta["camera_order"] != CAMERAS:
            raise ValueError(f"camera order mismatch for {member}")
        tensors = load_file(str(shard), device="cpu")
        n = int(entry["decision_count"])
        expected_shapes = {"decision_index": (n,), "chunk_start_frame": (n,), "visual_tokens": (n, 2, 64, 960), "visual_valid_mask": (n, 2, 64), "pooled_visual": (n, 2, 960), "visual_delta": (n, 2, 960), "observation_state": (n, 8), "state_delta": (n, 8), "has_previous": (n,), "previous_action_mean7": (n, 7), "previous_action_last7": (n, 7), "previous_executed_action_count": (n,)}
        if set(tensors) != set(expected_shapes):
            raise ValueError(f"tensor key whitelist mismatch for {member}")
        for key, shape in expected_shapes.items():
            if tensors[key].dtype != EXPECTED_DTYPES[key]:
                raise ValueError(f"tensor dtype mismatch for {member}/{key}: {tensors[key].dtype}")
            if tuple(tensors[key].shape) != shape or not torch.isfinite(tensors[key].float()).all():
                raise ValueError(f"tensor contract mismatch for {member}/{key}")
        if not bool(tensors["visual_valid_mask"].all()):
            raise ValueError(f"visual valid mask must be all true for {member}")
        for i in range(n):
            if int(tensors["decision_index"][i]) != i:
                raise ValueError(f"nonconsecutive decision for {member}/{i}")
            if bool(tensors["has_previous"][i]) != (i > 0):
                raise ValueError(f"has_previous mismatch for {member}/{i}")
            if (member, i) not in overlay:
                raise ValueError(f"missing causal overlay for {member}/{i}")
            if i == 0 and int(tensors["previous_executed_action_count"][i]) != 0:
                raise ValueError(f"decision 0 previous count must be zero for {member}")
            if i == 0:
                for key in ("previous_action_mean7", "previous_action_last7", "visual_delta", "state_delta"):
                    if not torch.equal(tensors[key][i], torch.zeros_like(tensors[key][i])):
                        raise ValueError(f"decision 0 {key} must be exactly zero for {member}")
                overlay0 = overlay[(member, i)]
                if overlay0.get("previous_decision_index") is not None or overlay0.get("previous_canonical_end_reason") is not None:
                    raise ValueError(f"decision 0 overlay previous fields must be null for {member}")
            else:
                if int(tensors["previous_executed_action_count"][i]) != int(tensors["chunk_start_frame"][i] - tensors["chunk_start_frame"][i - 1]):
                    raise ValueError(f"previous executed count interval mismatch for {member}/{i}")
                if not torch.equal(tensors["visual_delta"][i], tensors["pooled_visual"][i] - tensors["pooled_visual"][i - 1]):
                    raise ValueError(f"visual delta mismatch for {member}/{i}")
                if not torch.equal(tensors["state_delta"][i], tensors["observation_state"][i] - tensors["observation_state"][i - 1]):
                    raise ValueError(f"state delta mismatch for {member}/{i}")
                overlay_i = overlay[(member, i)]
                if overlay_i.get("previous_decision_index") != i - 1 or overlay_i.get("previous_canonical_end_reason") not in REASONS:
                    raise ValueError(f"overlay strict k-1 mismatch for {member}/{i}")
        episodes.append(CacheEpisode(member, n, meta, tensors, {k[1]: v for k, v in overlay.items() if k[0] == member}, entry["shard_sha256"], entry["sidecar_sha256"]))
    if sum(x.decision_count for x in episodes) != 714:
        raise ValueError("expected 714 decisions")
    return CacheDataset(episodes, actual_horizon, horizon_bindings)


class CacheDataset:
    def __init__(self, episodes: list[CacheEpisode], action_horizon: int = 5, horizon_bindings: dict[str, dict[str, Any]] | None = None) -> None:
        self.episodes = episodes
        self.action_horizon = int(action_horizon)
        self.horizon_bindings = horizon_bindings or {}
        members = [episode.member for episode in episodes]
        if len(set(members)) != len(members):
            raise ValueError("duplicate member")
        self.samples: list[TransitionSample] = []
        for episode in episodes:
            t = episode.tensors
            for k in range(episode.decision_count - 1):
                rows = [self.row(episode, j) for j in range(max(0, k - 7), k + 1)]
                self.samples.append(TransitionSample(episode.member, k, rows, t["observation_state"][k + 1] - t["observation_state"][k], torch.cat([t["previous_action_mean7"][k + 1], t["previous_action_last7"][k + 1]])))

    def row(self, episode: CacheEpisode, k: int) -> dict[str, Any]:
        t = episode.tensors
        overlay = episode.overlay[k]
        prev = None if k == 0 else torch.cat([t["previous_action_mean7"][k], t["previous_action_last7"][k]])
        prev_overlay = None if k == 0 else overlay["previous_canonical_end_reason"]
        return {
            "member": episode.member,
            "episode_id": episode.member,
            "env_id": episode.member,
            "decision_index": k,
            "decision_start_frame": int(t["chunk_start_frame"][k]),
            "prev_executed_start": None if k == 0 else int(t["chunk_start_frame"][k - 1]),
            "prev_executed_end": None if k == 0 else int(t["chunk_start_frame"][k]),
            "prev_executed_count": None if k == 0 else int(t["previous_executed_action_count"][k]),
            "previous_reason": prev_overlay,
            "previous_action": prev,
            "previous_decision_index": None if k == 0 else k - 1,
            "visual_feature": t["pooled_visual"][k],
            "previous_visual_feature": None if k == 0 else t["pooled_visual"][k - 1],
            "state": t["observation_state"][k],
            "previous_state": None if k == 0 else t["observation_state"][k - 1],
            "event_observable": _event(prev_overlay, None if k == 0 else int(t["previous_executed_action_count"][k]), self.action_horizon),
        }

    def split(self, seed: int = 20260907) -> tuple[list[CacheEpisode], list[CacheEpisode]]:
        ordered = sorted(self.episodes, key=lambda x: hashlib.sha256(f"{seed}{x.member}".encode()).hexdigest())
        return ordered[:8], ordered[8:]
