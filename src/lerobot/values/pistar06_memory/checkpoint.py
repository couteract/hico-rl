"""Strict outer checkpoint bundle for B02M0 smoke models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_bundle(model: torch.nn.Module, path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {"state_dict": state, "state_dict_hash": tensor_hash(state), "manifest": manifest}
    torch.save(payload, path)
    out = dict(manifest)
    out.update({"checkpoint_path": str(path), "checkpoint_tensor_state_hash": payload["state_dict_hash"], "checkpoint_size": path.stat().st_size})
    path.with_suffix(".manifest.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def load_bundle(path: Path, model: torch.nn.Module, expected: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = payload["manifest"]
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"checkpoint provenance mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    if payload.get("state_dict_hash") != tensor_hash(payload["state_dict"]):
        raise ValueError("checkpoint tensor-byte hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    return manifest
