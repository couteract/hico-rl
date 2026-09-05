# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50  # Overlapping inference: re-observe before consuming the full chunk.

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Gripper dimension for weighted loss calculation
    # -1 means auto-detect (use last dimension as gripper)
    gripper_action_dim: int = -1

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = -1  # ✅ 修复：-1 表示禁用自注意力交替，永远走 cross-attention
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # 🔥 SSM2 (Mamba-2) Temporal Adapter配置
    # ✅ P0 FIX: 对齐chunk_size，配合严格滑动窗口修复
    # Disabled by default so importing/using base SmolVLA does not require the
    # optional CUDA extension. Enable explicitly for the custom temporal model.
    use_mamba_temporal: bool = False
    mamba_history_length: int = 50  # ⭐ 扩展历史缓冲：覆盖3次推理上下文（50→100）
    # 原理：
    #   - 训练时Mamba只见过seq_len=50帧（Position 0-49）
    #   - 推理时必须保证Mamba输入≤50帧（通过滑动窗口截断）
    #   - history_length=50足够支持滑动窗口，同时避免过大buffer浪费内存
    #   - 配合temporal_adapter_mamba.py中的滑动窗口约束消除OOD噪声
    mamba_d_state: int = 64  # Mamba state dimension
    mamba_d_conv: int = 4  # Conv kernel size
    mamba_expand: int = 2  # Expansion factor

    # 🆕 Timestep Conditioning（时间步条件化）
    # 允许Mamba根据扩散去噪进度调整平滑强度
    # - 早期去噪(t≈1，高噪声): 重度平滑以抑制噪声
    # - 后期去噪(t≈0，低噪声): 轻度平滑以保留细节
    mamba_time_embed_dim: int = 64  # 时间嵌入特征维度（默认64）

    # 🆕 Mamba预热配置（用户请求：让预热仅在需要时启用）
    # 用途：解决Episode 0冷启动问题（特别是在ACG模式下）
    # 建议：基准模式关闭（保持训练-测试一致），ACG模式开启（利用预热提升性能）
    enable_mamba_warmup: bool = False  # 是否启用Mamba历史预热（默认关闭，对齐基线）
    mamba_warmup_frames: int = 3  # 预热帧数（零向量填充）

    # 🔥 MSE Loss 权重配置（专家建议：避免夹爪被淹没）
    gripper_loss_weight: float = 1  # 夹爪loss权重（相对于关节）
    # 原理：在未加权的losses.mean()中，夹爪只占1/7（14.3%）维度，
    # 梯度会被关节（6/7=85.7%）淹没。5倍权重让优化器重视夹爪精度。

    # 🔥 ACG (Action Coherence Guidance)配置 - Shared History Architecture
    # ACG is a training-free, test-time guidance method (arXiv:2510.22201)
    # It improves action coherence by steering away from incoherent actions
    # ⚠️ ACG doubles inference time (18ms→38ms) but improves success rates on fine manipulation
    #
    # 🔥 Architecture: Dual-Scale Coherence Theory (Expert-Recommended)
    #   - Mamba (Micro-Coherence): Physical smoothness (kinematic consistency)
    #   - ACG (Macro-Coherence): Semantic alignment (task-goal consistency)
    #   - Shared History: Both normal and incoherent paths use the same Mamba history H_{t-1}
    #                     This ensures pure semantic gradient (no history state mismatch noise)
    #
    use_acg: bool = False  # Whether to enable ACG during inference (test-time only, no training needed)
    acg_guidance_scale: float = 1.0  # ✅ Guidance strength (can increase to 1.2-1.5 due to purer signal)
    acg_target_layers: tuple | None = None  # ✅ Target layers for ACG (set to [9,10,11,12] in __post_init__)

    # ❌ DEPRECATED (Shared History Architecture no longer uses these):
    acg_incoherent_noise_scale: float = 0.15  # [UNUSED] Previously for noise injection (now removed)
    acg_history_blend_ratio: float = 0.7      # [UNUSED] Previously for history blending (now removed)
    acg_magnitude_normalization: bool = False # [UNUSED] Previously for magnitude normalization
    acg_scale_clip_range: tuple = (0.5, 2.0)  # [UNUSED] Previously for scale clipping
    log_acg_metrics: bool = False             # [UNUSED] Debug logging

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

        # ✅ Phase 1 Fix: 设置ACG目标层级（如果未指定）
        # 固化为已验证的最优层级[9,10,11,12]，避免auto-calc的不确定性
        if self.acg_target_layers is None:
            self.acg_target_layers = (9, 10, 11, 12)

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
