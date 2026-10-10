"""Pretraining smoke model built around the independent memory provider."""
from __future__ import annotations

import torch
from torch import nn

from lerobot.memory.himem import HiMemConfig, HiMemEncoder
from .provider import Pistar06MemoryProvider


class PredictionHeads(nn.Module):
    def __init__(self, memory_dim: int = 256) -> None:
        super().__init__()
        self.state_delta = nn.Linear(memory_dim, 8)
        self.action_summary = nn.Linear(memory_dim, 14)

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state_delta(memory), self.action_summary(memory)


class Pistar06MemorySmokeModel(nn.Module):
    def __init__(self, config: HiMemConfig) -> None:
        super().__init__()
        self.memory = Pistar06MemoryProvider(HiMemEncoder(config))
        self.heads = PredictionHeads(config.memory_dim)

    def forward(self, batch_rows: list[list[dict]], device: torch.device) -> dict[str, torch.Tensor]:
        output = self.memory(batch_rows, device=device)
        state, action = self.heads(output.memory_features)
        return {"memory_features": output.memory_features, "memory_valid": output.memory_valid, "window_lengths": output.window_lengths, "state_delta": state, "action_summary": action, "audit": output.audit}
