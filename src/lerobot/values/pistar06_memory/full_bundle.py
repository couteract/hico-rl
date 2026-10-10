"""Auditable complete Pistar06 + memory residual checkpoint bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from lerobot.values.pistar06.modeling_pistar06 import Pistar06Policy
from lerobot.values.pistar06_memory.value import MemoryAwarePistar06Value


REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "pistar06_save_info.json",
    "memory_value.pt",
    "memory_value_config.json",
)
MANIFEST_NAME = "full_value_bundle.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy().tobytes()


def tensor_mapping_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(_tensor_bytes(state[name]))
    return h.hexdigest()


def _actual_contract(directory: Path, provenance: dict[str, Any], external_files: list[dict[str, Any]]) -> dict[str, Any]:
    files = {
        name: {"size": (directory / name).stat().st_size, "sha256": _sha256(directory / name)}
        for name in REQUIRED_FILES
    }
    base_state = load_file(str(directory / "model.safetensors"), device="cpu")
    payload = torch.load(directory / "memory_value.pt", map_location="cpu", weights_only=False)
    residual_state = payload.get("state_dict", {})
    if set(residual_state) != {"residual.weight"}:
        raise ValueError("bundle residual state must contain only residual.weight")
    residual_hash = tensor_mapping_hash(residual_state)
    if payload.get("state_dict_hash") != hashlib.sha256(_tensor_bytes(residual_state["residual.weight"])).hexdigest():
        raise ValueError("bundle residual internal tensor hash mismatch")
    external = []
    for item in external_files:
        path = Path(item["path"])
        external.append({"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)})
    core = {
        "schema_version": 1,
        "required_files": files,
        "base_tensor_hash": tensor_mapping_hash(base_state),
        "residual_tensor_hash": residual_hash,
        "provenance": provenance,
        "provenance_sha256": hashlib.sha256(_canonical(provenance)).hexdigest(),
        "external_files": sorted(external, key=lambda x: x["path"]),
    }
    core["final_value_bundle_hash"] = hashlib.sha256(_canonical(core)).hexdigest()
    return core


def write_full_value_bundle_manifest(
    directory: str | Path,
    provenance: dict[str, Any],
    external_files: list[dict[str, Any]],
) -> dict[str, Any]:
    directory = Path(directory)
    contract = _actual_contract(directory, provenance, external_files)
    (directory / MANIFEST_NAME).write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return contract


def save_full_value_bundle(
    model: MemoryAwarePistar06Value,
    directory: str | Path,
    provenance: dict[str, Any],
    external_files: list[dict[str, Any]],
) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.base.save_pretrained(directory)
    model.save_pretrained(directory, provenance)
    return write_full_value_bundle_manifest(directory, provenance, external_files)


def validate_full_value_bundle(
    directory: str | Path,
    expected_provenance: dict[str, Any],
    *,
    expected_bundle_hash: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    directory = Path(directory)
    if not strict:
        raise ValueError("full VALUE bundle validation is strict-only")
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("full VALUE bundle manifest missing")
    manifest = json.loads(manifest_path.read_text())
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"full VALUE bundle files missing: {missing}")
    actual = _actual_contract(directory, manifest.get("provenance", {}), manifest.get("external_files", []))
    if actual != manifest:
        raise ValueError("full VALUE bundle file/tensor/hash contract mismatch")
    if expected_bundle_hash is not None and actual["final_value_bundle_hash"] != expected_bundle_hash:
        raise ValueError("full VALUE bundle hash mismatch")
    if set(actual["provenance"]) != set(expected_provenance):
        raise ValueError("full VALUE bundle provenance fields mismatch")
    mismatches = {k: {"actual": actual["provenance"].get(k), "expected": v}
                  for k, v in expected_provenance.items() if actual["provenance"].get(k) != v}
    if mismatches:
        raise ValueError(f"full VALUE bundle provenance mismatch: {json.dumps(mismatches, sort_keys=True)}")
    payload = torch.load(directory / "memory_value.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((directory / "memory_value_config.json").read_text())
    if payload.get("provenance") != actual["provenance"] or cfg.get("provenance") != actual["provenance"]:
        raise ValueError("full VALUE bundle embedded provenance mismatch")
    return actual


def load_full_value_bundle(
    directory: str | Path,
    expected_provenance: dict[str, Any],
    *,
    expected_bundle_hash: str | None = None,
    strict: bool = True,
    device: str | torch.device | None = None,
) -> tuple[MemoryAwarePistar06Value, dict[str, Any]]:
    directory = Path(directory)
    contract = validate_full_value_bundle(
        directory, expected_provenance, expected_bundle_hash=expected_bundle_hash, strict=strict
    )
    base = Pistar06Policy.from_pretrained(directory, local_files_only=True).to(device or "cpu")
    model = MemoryAwarePistar06Value.from_pretrained(
        directory, base, expected_provenance=expected_provenance, strict=True
    ).to(device or "cpu")
    return model, contract
