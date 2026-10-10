import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.values.pistar06_memory.full_bundle import (
    validate_full_value_bundle,
    write_full_value_bundle_manifest,
)


def _fake_bundle(tmp_path: Path):
    save_file({"base.weight": torch.arange(4, dtype=torch.float32)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "pistar06_save_info.json").write_text("{}\n")
    residual = torch.zeros(2, 2)
    import hashlib
    rh = hashlib.sha256(residual.numpy().tobytes()).hexdigest()
    provenance = {"fold_id": 0, "route": "memory_w8", "seed": 20261020}
    torch.save({"state_dict": {"residual.weight": residual}, "state_dict_hash": rh, "provenance": provenance}, tmp_path / "memory_value.pt")
    (tmp_path / "memory_value_config.json").write_text(json.dumps({"memory_dim": 2, "num_bins": 2, "provenance": provenance}))
    external = tmp_path / "processor.json"
    external.write_text("{}\n")
    manifest = write_full_value_bundle_manifest(tmp_path, provenance, [{"path": str(external)}])
    return provenance, manifest


def test_full_bundle_validates_and_rejects_base_tamper(tmp_path):
    provenance, manifest = _fake_bundle(tmp_path)
    assert validate_full_value_bundle(tmp_path, provenance, expected_bundle_hash=manifest["final_value_bundle_hash"])
    save_file({"base.weight": torch.ones(4)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_full_value_bundle(tmp_path, provenance, expected_bundle_hash=manifest["final_value_bundle_hash"])


def test_full_bundle_rejects_provenance_mismatch(tmp_path):
    provenance, manifest = _fake_bundle(tmp_path)
    bad = dict(provenance, seed=20261021)
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_full_value_bundle(tmp_path, bad, expected_bundle_hash=manifest["final_value_bundle_hash"])
