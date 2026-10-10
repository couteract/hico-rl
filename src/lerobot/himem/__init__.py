"""
Compatibility shim for src/lerobot/himem.

This module re-exports the canonical implementation from src/lerobot/memory/himem.
Kept for backward compatibility with B01 tests.
"""

import warnings

warnings.warn(
    "lerobot.himem is deprecated. Use lerobot.memory.himem instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from canonical location
from lerobot.memory.himem import *  # noqa: F401, F403

__all__ = [
    "HiMemConfig",
    "TokenSchema",
    "CausalToken",
    "HIMEM_PROTOCOL_VERSION",
    "HIMEM_SCHEMA_VERSION",
    "get_debug_config",
    "get_production_config",
    "EventFeatureEncoder",
    "summarize_action_chunk",
    "GRUMemoryBackend",
    "Mamba2Backend",
    "SimplifiedSSMExperimental",
    "create_backend",
    "RollingWindowStateManager",
    "HiMemEncoder",
    "create_himem_encoder",
    "save_checkpoint",
    "load_checkpoint",
    "verify_checkpoint_determinism",
    "compute_state_dict_hash",
]
