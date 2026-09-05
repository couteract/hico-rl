"""Local SmolVLM2 artifact discovery without downloading or loading weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SMOLVLM2_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
)
SMOLVLM2_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


@dataclass(frozen=True)
class SmolVLM2PathReport:
    reference: str
    local_path: str | None
    is_huggingface_id: bool
    exists: bool
    required_files: dict[str, bool]
    weights: tuple[str, ...]
    valid: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "local_path": self.local_path,
            "is_huggingface_id": self.is_huggingface_id,
            "exists": self.exists,
            "required_files": dict(self.required_files),
            "weights": list(self.weights),
            "valid": self.valid,
            "note": self.note,
        }


def inspect_smolvlm2_path(reference: str | Path) -> SmolVLM2PathReport:
    """Inspect a local model directory or describe a Hub repo id.

    This function performs filesystem checks only. It deliberately does not
    invoke ``transformers`` or access the network, so it is safe in minimal
    benchmark environments.
    """

    ref = str(reference)
    candidate = Path(reference).expanduser()
    looks_like_hub_id = not candidate.exists() and "/" in ref and not ref.startswith((".", "/"))
    if not candidate.is_dir():
        return SmolVLM2PathReport(
            reference=ref,
            local_path=None,
            is_huggingface_id=looks_like_hub_id,
            exists=False,
            required_files={name: False for name in SMOLVLM2_REQUIRED_FILES},
            weights=(),
            valid=False,
            note=(
                "Hub repo id; Transformers will resolve it when explicitly loaded."
                if looks_like_hub_id
                else "Local SmolVLM2 directory does not exist."
            ),
        )

    required = {name: (candidate / name).is_file() for name in SMOLVLM2_REQUIRED_FILES}
    weights = tuple(name for name in SMOLVLM2_WEIGHT_NAMES if (candidate / name).is_file())
    if (candidate / "model.safetensors").is_file():
        weights = weights + tuple(sorted(p.name for p in candidate.glob("model-*.safetensors")))
    valid = all(required.values()) and bool(weights)
    return SmolVLM2PathReport(
        reference=ref,
        local_path=str(candidate.resolve()),
        is_huggingface_id=False,
        exists=True,
        required_files=required,
        weights=tuple(dict.fromkeys(weights)),
        valid=valid,
        note="Local artifact is complete." if valid else "Local artifact is missing required files or weights.",
    )


def resolve_smolvlm2_reference(reference: str | Path) -> str:
    """Return an absolute local path when present, otherwise preserve a Hub id."""

    candidate = Path(reference).expanduser()
    return str(candidate.resolve()) if candidate.is_dir() else str(reference)


__all__ = [
    "SMOLVLM2_REQUIRED_FILES",
    "SMOLVLM2_WEIGHT_NAMES",
    "SmolVLM2PathReport",
    "inspect_smolvlm2_path",
    "resolve_smolvlm2_reference",
]
