#!/usr/bin/env python3
"""Fast import/configuration checks for the Evo-RL custom SmolVLA integration."""

from __future__ import annotations

import importlib
import sys


def check_module(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"FAIL module {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"OK module {name}: {getattr(module, '__file__', '<builtin>')}")
    return True


def main() -> int:
    import torch

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")

    module_names = (
        "lerobot",
        "lerobot.policies.factory",
        "lerobot.policies.smolvla.configuration_smolvla",
        "lerobot.policies.smolvla.modeling_smolvla",
        "lerobot.policies.smolvla.processor_smolvla",
        "lerobot.values",
    )
    module_failures = [name for name in module_names if not check_module(name)]

    try:
        from lerobot.policies.factory import get_policy_class, make_policy_config
    except Exception as exc:
        print(f"FAIL factory import: {type(exc).__name__}: {exc}")
        print("Install the Evo-RL base dependencies, then rerun this check.")
        return 1

    try:
        policy_class = get_policy_class("smolvla")
        config = make_policy_config("smolvla", device="cpu", use_mamba_temporal=False)
        print(f"OK factory smolvla: {policy_class.__module__}.{policy_class.__name__}")
        print(f"OK config: type={config.type}, device={config.device}")
        print(f"OK config serialization fields: lora_rank={config.lora_rank}, use_mamba_temporal={config.use_mamba_temporal}")
    except Exception as exc:
        print(f"FAIL factory/config: {type(exc).__name__}: {exc}")
        return 1

    mamba_ok = check_module("mamba_ssm")
    if mamba_ok and torch.cuda.is_available():
        try:
            from lerobot.policies.smolvla.temporal_adapter_mamba import TemporalMambaAdapter

            adapter = TemporalMambaAdapter(
                hidden_size=32,
                history_length=50,
                d_state=16,
                d_conv=4,
                expand=2,
                time_embed_dim=16,
            ).cuda()
            inputs = torch.randn(1, 8, 32, device="cuda", requires_grad=True)
            output = adapter(inputs, timestep=torch.tensor([0.5], device="cuda"))
            output.square().mean().backward()
            if output.shape != inputs.shape or inputs.grad is None or not torch.isfinite(inputs.grad).all():
                raise RuntimeError("unexpected output shape or non-finite input gradients")
            print(f"OK Mamba CUDA forward/backward: shape={tuple(output.shape)}")
        except Exception as exc:
            print(f"FAIL Mamba CUDA forward/backward: {type(exc).__name__}: {exc}")
            return 1

    return 1 if module_failures or not mamba_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
