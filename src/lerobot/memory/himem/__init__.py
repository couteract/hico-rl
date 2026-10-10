"""
HiMem: Hierarchical Memory for Vision-Language-Action models.

B01R implementation with rolling window semantics and causal token protocol.
"""

from .config import (
    HiMemConfig,
    TokenSchema,
    CausalToken,
    HIMEM_PROTOCOL_VERSION,
    HIMEM_SCHEMA_VERSION,
    get_debug_config,
    get_production_config,
)

from .event_features import (
    EventFeatureEncoder,
    summarize_action_chunk,
)

from .backends import (
    GRUMemoryBackend,
    Mamba2Backend,
    SimplifiedSSMExperimental,
    create_backend,
)

from .state_manager import RollingWindowStateManager

from .encoder import (
    HiMemEncoder,
    create_himem_encoder,
)

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint_determinism,
    compute_state_dict_hash,
)
from .r5_snapshot_rollout import (
    R5SnapshotRolloutAdapter,
    compare_snapshot,
    validate_snapshot_contract,
    validate_snapshot_payload,
)
from .sac_q_interface import (
    compose_action, validate_action_record, Transition, QMemoryInput,
    actor6_to_critic7, canonical_action_summary, validate_causal_sequence,
    TorchQOnlyCritic, build_torch_policy6, bellman_mask, latent_entropy_contract,
    validate_path_manifest,
)
from .q_only_learner import (
    GAMMA, POLYAK_TAU, SYNTHETIC_EXACT_CONTRACT, LATENT_ENTROPY_DOMAIN,
    QOnlyBatch, QOnlySACLearner, compose_submitted_action_torch,
    beta_zero_vla_baseline, validate_optimizer_partition,
)

__all__ = [
    # Config
    "HiMemConfig",
    "TokenSchema",
    "CausalToken",
    "HIMEM_PROTOCOL_VERSION",
    "HIMEM_SCHEMA_VERSION",
    "get_debug_config",
    "get_production_config",
    # Feature encoding
    "EventFeatureEncoder",
    "summarize_action_chunk",
    # Backends
    "GRUMemoryBackend",
    "Mamba2Backend",
    "SimplifiedSSMExperimental",
    "create_backend",
    # State management
    "RollingWindowStateManager",
    # Main encoder
    "HiMemEncoder",
    "create_himem_encoder",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "verify_checkpoint_determinism",
    "compute_state_dict_hash",
    "R5SnapshotRolloutAdapter",
    "compare_snapshot",
    "validate_snapshot_contract",
    "validate_snapshot_payload",
    "compose_action", "validate_action_record", "Transition", "QMemoryInput",
    "actor6_to_critic7", "canonical_action_summary", "validate_causal_sequence",
    "TorchQOnlyCritic", "build_torch_policy6", "bellman_mask", "latent_entropy_contract",
    "validate_path_manifest",
    "GAMMA", "POLYAK_TAU", "SYNTHETIC_EXACT_CONTRACT", "LATENT_ENTROPY_DOMAIN",
    "QOnlyBatch", "QOnlySACLearner", "compose_submitted_action_torch",
    "beta_zero_vla_baseline", "validate_optimizer_partition",
]

__version__ = "0.2.0-b01r"
