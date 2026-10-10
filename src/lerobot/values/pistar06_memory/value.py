"""Independent memory-aware VALUE adapter for the B02M1 engineering stage."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.values.pistar06.modeling_pistar06 import expected_value_from_logits, project_values_to_bins


class MemoryAwarePistar06Value(nn.Module):
    """Wrap unchanged Pistar06 VALUE logits with a causal residual branch."""

    def __init__(self, base_policy: nn.Module, memory_dim: int = 256, num_bins: int = 201) -> None:
        super().__init__()
        self.base = base_policy
        self.memory_dim = int(memory_dim)
        self.num_bins = int(num_bins)
        if not hasattr(base_policy, "bin_centers") or int(base_policy.bin_centers.numel()) != self.num_bins:
            raise ValueError("base policy bin contract does not match the memory adapter")
        self.residual = nn.Linear(self.memory_dim, self.num_bins, bias=False)
        nn.init.zeros_(self.residual.weight)

    @property
    def config(self):
        return self.base.config

    def base_logits(self, batch: dict[str, Tensor]) -> Tensor:
        return self.base.model(
            input_ids=batch["observation.language.tokens"],
            attention_mask=batch["observation.language.attention_mask"],
            images=batch["observation.pistar06.images"],
            image_attention_mask=batch["observation.pistar06.image_attention_mask"],
        )

    def forward_logits(self, batch: dict[str, Tensor], memory_features: Tensor | None = None) -> Tensor:
        logits = self.base_logits(batch)
        if memory_features is None:
            return logits
        memory_features = memory_features.to(device=logits.device, dtype=logits.dtype)
        expected = (logits.shape[0], self.memory_dim)
        if memory_features.ndim != 2 or tuple(memory_features.shape) != expected:
            raise ValueError(f"memory_features must have shape {expected}, got {tuple(memory_features.shape)}")
        return logits + self.residual(memory_features)

    def predict_value(self, batch: dict[str, Tensor], memory_features: Tensor | None = None) -> Tensor:
        logits = self.forward_logits(batch, memory_features)
        return expected_value_from_logits(logits, self.base.bin_centers.to(logits.device))

    def forward(self, batch: dict[str, Tensor], memory_features: Tensor | None = None, reduction: str = "mean"):
        target_key = self.config.target_key
        if target_key not in batch:
            raise KeyError(f"Missing target key '{target_key}'")
        target = batch[target_key]
        if not isinstance(target, Tensor):
            target = torch.as_tensor(target)
        target = target.to(device=next(self.parameters()).device, dtype=torch.float32).reshape(-1)
        logits = self.forward_logits(batch, memory_features)
        soft_target = project_values_to_bins(target, self.base.bin_centers.to(logits.device))
        per_sample = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if self.config.loss_weight_key in batch:
            weight = batch[self.config.loss_weight_key]
            if not isinstance(weight, Tensor):
                weight = torch.as_tensor(weight)
            per_sample = per_sample * weight.to(device=logits.device, dtype=torch.float32).reshape(-1)
        loss = per_sample if reduction == "none" else per_sample.mean()
        predicted = expected_value_from_logits(logits, self.base.bin_centers.to(logits.device))
        metrics = {"loss": float(loss.mean().detach()), "value_mae": float((predicted - target).abs().mean().detach())}
        return loss, metrics

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def save_pretrained(self, directory: str | Path, provenance: dict[str, Any] | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # The Pistar06 base checkpoint is saved separately by its public
        # save_pretrained path. Keep this adapter artifact small and explicit:
        # frozen SigLIP/Gemma tensors must not be duplicated here.
        adapter_state = {"residual.weight": self.residual.weight.detach().cpu()}
        tensor_bytes = adapter_state["residual.weight"].contiguous().numpy().tobytes()
        tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()
        torch.save({"state_dict": adapter_state, "state_dict_hash": tensor_hash, "provenance": provenance or {}}, directory / "memory_value.pt")
        (directory / "memory_value_config.json").write_text(json.dumps({"memory_dim": self.memory_dim, "num_bins": self.num_bins, "provenance": provenance or {}}, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        base_policy: nn.Module,
        expected_provenance: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> "MemoryAwarePistar06Value":
        directory = Path(directory)
        payload = torch.load(directory / "memory_value.pt", map_location="cpu", weights_only=False)
        cfg = json.loads((directory / "memory_value_config.json").read_text())
        if strict and expected_provenance is None:
            raise ValueError("strict adapter load requires expected_provenance")
        if expected_provenance is not None:
            actual = payload.get("provenance")
            if not isinstance(actual, dict):
                raise ValueError("adapter checkpoint is missing provenance")
            missing = sorted(set(expected_provenance) - set(actual))
            if missing:
                raise ValueError(f"adapter provenance missing fields: {missing}")
            mismatches = {
                key: {"actual": actual.get(key), "expected": value}
                for key, value in expected_provenance.items()
                if actual.get(key) != value
            }
            if mismatches:
                raise ValueError(f"adapter provenance mismatch: {json.dumps(mismatches, sort_keys=True, default=str)}")
        model = cls(base_policy, memory_dim=int(cfg["memory_dim"]), num_bins=int(cfg["num_bins"]))
        if set(payload.get("state_dict", {})) != {"residual.weight"}:
            raise ValueError("memory adapter checkpoint must contain only residual.weight")
        weight = payload["state_dict"].get("residual.weight")
        if not isinstance(weight, Tensor) or tuple(weight.shape) != tuple(model.residual.weight.shape):
            raise ValueError("memory adapter residual.weight shape mismatch")
        actual_hash = hashlib.sha256(weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if payload.get("state_dict_hash") is not None and payload.get("state_dict_hash") != actual_hash:
            raise ValueError("memory adapter tensor-byte hash mismatch")
        model.residual.weight.data.copy_(weight.to(dtype=model.residual.weight.dtype))
        return model
