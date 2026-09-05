#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
# ✅ FIX Bug #3: 移除硬导入，改为延迟导入（避免use_mamba_temporal=False时仍需要mamba_ssm依赖）
# from lerobot.policies.smolvla.temporal_adapter_mamba import TemporalMambaAdapter
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """
        This should be called whenever the environment is reset.

        🔥 条件性Mamba预热（用户可配置）
        - 用途：解决Episode 0冷启动问题（特别是在ACG模式下）
        - 权衡：预热可能提升ACG性能，但会导致基准模式train-test不一致
        - 建议：基准模式关闭（enable_mamba_warmup=False），ACG模式开启（=True）
        """
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

        # 🆕 VLM特征缓存（推理时复用）
        self.cached_image_hidden_states = None
        self.prev_image_hash = None

        # ✅ 条件性Mamba历史预热（可通过配置控制）
        if hasattr(self.model, "temporal_adapter") and self.model.temporal_adapter is not None:
            # 总是先清空历史
            self.model.temporal_adapter.reset_history()

            # 🆕 根据配置决定是否预热
            if self.config.enable_mamba_warmup:
                # 预热逻辑：填充N帧零向量到历史缓冲区
                # 注意：这会改变Episode开始时的初始状态
                # - 如果模型训练时从空历史开始，预热会导致train-test不一致
                # - 但在ACG模式下，预热可能帮助稳定初始推理
                warm_frames = self.config.mamba_warmup_frames
                dummy_feature = torch.zeros(
                    1, 1, self.model.temporal_adapter.hidden_size,
                    device=self.model.temporal_adapter._dummy.device,
                    dtype=self.model.temporal_adapter._dummy.dtype
                )
                for _ in range(warm_frames):
                    self.model.temporal_adapter.history_buffer.append(dummy_feature)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues[ACTION]) == 0:
            # 🔥 Soft Reset: 只在queue耗尽时执行（不是每次调用！）
            # 此时机器人已执行完上一批所有动作，需要生成新动作
            # 数学原理：Mamba Conv1d kernel = 4，计算加速度需要4帧历史
            # - Hard Reset (全清空): 虚假加速度 → 机械臂猛冲
            # - No Reset (全保留): 累积误差导致漂移
            # - Soft Reset (保留K帧): 速度连续性 ✓ + 漂移消除 ✓
            if hasattr(self.model, "temporal_adapter") and self.model.temporal_adapter is not None:
                adapter = self.model.temporal_adapter
                # 🔧 自适应Soft Reset帧数：保持20%比例跨不同n_action_steps值
                # n_action_steps=20 → WARMUP_FRAMES=4 (20%)
                # n_action_steps=25 → WARMUP_FRAMES=5 (20%)
                # 最小值4对应Mamba Conv1d kernel size，保证时序连续性
                WARMUP_FRAMES = max(4, int(self.config.n_action_steps * 0.2))

                if len(adapter.history_buffer) > 0:
                    # 保留最后4帧（对应最近4个已执行动作的context）
                    recent_history = list(adapter.history_buffer)[-WARMUP_FRAMES:]
                    adapter.reset_history()
                    for frame in recent_history:
                        adapter.history_buffer.append(frame)

            actions = self._get_action_chunk(batch, noise, **kwargs)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()


    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        # ✅ FIX Bug #2: 传递actions_is_pad用于检测episode边界并重置Mamba历史
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, actions_is_pad)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # ✅ 专家修复：分离关节和夹爪loss，避免夹爪被淹没
        # 🔥 修复：改用加权平均而非错误的加法（避免loss爆炸）
        # 原理：在未加权的losses.mean()中，夹爪只占1/7（14.3%）维度
        # 梯度会被关节（6/7=85.7%）淹没，导致机器人学会移动但学不会抓取
        gripper_dim = self.config.gripper_action_dim
        if gripper_dim == -1:
            gripper_dim = self.config.max_action_dim - 1

        joint_losses = losses[:, :, :gripper_dim]  # [B, L, joint_dim]
        gripper_losses = losses[:, :, gripper_dim:]  # [B, L, 1 or 2]

        # 🔥 加权平均公式（保持loss量级，避免爆炸）
        # 之前错误：mse_loss = joint + 5*gripper（如果gripper=0.5，总loss=2.6爆炸）
        # 现在正确：mse_loss = (joint + 5*gripper) / 6（保持loss在合理范围）
        weight_sum = 1.0 + self.config.gripper_loss_weight
        mse_loss = (joint_losses.mean() + self.config.gripper_loss_weight * gripper_losses.mean()) / weight_sum

        # 记录详细监控指标
        loss_dict["mse_loss"] = mse_loss.item()
        loss_dict["joint_loss"] = joint_losses.mean().item()  # 新增：关节loss
        loss_dict["gripper_loss"] = gripper_losses.mean().item()  # 新增：夹爪loss

        # ✅ 详细监控每个时间步和维度的误差
        # 这有助于诊断哪个时间步或哪个动作维度（关节/夹爪）学习困难
        loss_dict["mse_per_timestep"] = losses.mean(dim=(0, 2)).tolist()  # [L] 每个时间步的平均loss
        loss_dict["mse_per_dim"] = losses.mean(dim=(0, 1)).tolist()  # [D] 每个维度的平均loss

        loss = mse_loss
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device,
            acg_target_layers=self.config.acg_target_layers,  # ✅ FIX: Pass ACG target layers
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # 🔥 SSM2 Temporal Adapter（动作空间平滑）
        # ✅ 修复：从隐空间（768维）移到动作空间（32维）
        # 原因：避免抑制隐空间的突变信号，让Mamba平滑真实动作值
        self.temporal_adapter = None
        if self.config.use_mamba_temporal:
            # ✅ FIX Bug #3: 延迟导入TemporalMambaAdapter（只在实际使用时导入）
            from lerobot.policies.smolvla.temporal_adapter_mamba import TemporalMambaAdapter
            self.temporal_adapter = TemporalMambaAdapter(
                hidden_size=self.config.max_action_dim,  # 🆕 改为动作维度（32），而非expert_hidden_size（768）
                history_length=self.config.mamba_history_length,
                d_state=self.config.mamba_d_state,
                d_conv=self.config.mamba_d_conv,
                expand=self.config.mamba_expand,
                time_embed_dim=self.config.mamba_time_embed_dim,  # 🆕 NEW: Timestep conditioning
            )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

        # Train temporal adapter if enabled
        if self.temporal_adapter is not None:
            for params in self.temporal_adapter.parameters():
                params.requires_grad = True

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, actions_is_pad=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)

        # ✅ 修复：先投影到动作空间，再应用Mamba平滑
        # 原顺序：Mamba(隐特征) -> action_proj  【问题：抑制隐空间突变信号】
        # 新顺序：action_proj -> Mamba(动作)    【修复：平滑真实动作值】
        v_t = self.action_out_proj(suffix_out)  # [B, L, action_dim]

        # 🔥 SSM2 Temporal Smoothing（在动作空间）
        if self.temporal_adapter is not None:
            # ✅ FIX CRITICAL BUG: 训练时不应重置历史，让Mamba学习chunk内的时序关系
            #
            # 🔴 FIX P0 (Critical): 训练时必须reset_history=True
            # 问题根源（训练-推理分布偏移）：
            #   - 训练时：Batch内32个sample来自连续idx，高度重叠（每对相邻sample重叠49/50帧）
            #   - 每个sample代表独立的预测起点（不同的预测任务）
            #   - reset=False会累积混合32个任务的历史 → P_train(h) = mix(h0, h1, ..., h31)
            #   - 推理时：单一连续轨迹 → P_test(h) = sequential(h_t-1, h_t-2, ...)
            #   - 分布不一致 → Mamba学不到有效的时序依赖
            #
            # 修复方案：
            #   - reset_history=True：每个sample独立处理（消除跨sample污染）
            #   - update_history=False：不需要存储训练batch的中间状态
            #   - Mamba学习chunk内（50帧）的时序平滑，与推理语义一致
            #
            # 推理时：在denoise_step中通过_apply_temporal_adapter_with_history_control管理
            v_t = self.temporal_adapter(
                v_t,
                reset_history=True,
                update_history=False,
                timestep=time  # NEW: Pass training timestep [B] for adaptive smoothing
            )

        losses = F.mse_loss(u_t, v_t, reduction="none")  # [B, L, D]

        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        # ✅ FIX 1: 捕获 outputs (包含 image hidden states)
        outputs, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        # outputs[0] 是 VLM 输出 (Image Hidden States)，我们需要缓存它
        cached_vlm_features = outputs[0]
        # ✅ FIX Bug #1: 保存VLM特征到实例变量供后续推理复用
        self.cached_image_hidden_states = cached_vlm_features

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        step_idx = 0
        total_steps = self.config.num_steps

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            is_final_step = (step_idx == total_steps - 1)  # ✅ FIX: 标记最后一步

            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                cached_vlm_features=cached_vlm_features,
                is_final_step=is_final_step,  # ✅ FIX: 传递是否最后一步的标志
                step_idx=step_idx,  # ✅ FIX: 传递step_idx用于调试日志
            )
            # Euler step
            x_t += dt * v_t
            time += dt
            step_idx += 1
        return x_t

    def _apply_temporal_adapter_with_history_control(self, suffix_out, is_final_step=False, timestep=None, return_residual=False):
        """
        Apply temporal adapter with manual history control during inference.

        During diffusion denoising, we must prevent noise frames from polluting
        the history buffer. Only the final denoised action should be stored.

        Args:
            suffix_out: Expert output features [B, L, D]
            is_final_step: Whether this is the final diffusion step
            timestep: Optional [B] timestep for adaptive smoothing
            return_residual: If True, return (output, residual) tuple for ACG compatibility

        Returns:
            smoothed output [B, L, D]
            OR (smoothed, residual) if return_residual=True

        Note:
            This function is only called during inference (in denoise_step).
            During training, temporal adapter is called directly in forward() method.
        """
        if self.temporal_adapter is None:
            if return_residual:
                # No Mamba: residual is zero
                return suffix_out, torch.zeros_like(suffix_out)
            return suffix_out

        # 推理时：手动管理历史缓冲区，防止扩散去噪中间步骤的噪声帧污染历史
        # 注意：此函数只在推理时调用（denoise_step中），训练时在forward()中直接调用
        from collections import deque

        # 1. 保存真实历史缓冲区（跨物理时间帧的真实动作）
        real_history_backup = list(self.temporal_adapter.history_buffer)
        original_buffer = self.temporal_adapter.history_buffer

        # 2. 创建临时缓冲区（使用真实历史的副本）用于本次平滑
        temp_buffer = deque(real_history_backup, maxlen=self.temporal_adapter.history_length)
        self.temporal_adapter.history_buffer = temp_buffer

        # 3. 调用 adapter（它会基于真实历史平滑，但不更新临时缓冲区中的噪声帧）
        # 🔥 FIX: 必须传 update_history=False，防止扩散去噪的噪声帧污染历史
        # ✅ ACG Support: 传递return_residual参数
        result = self.temporal_adapter(
            suffix_out,
            reset_history=False,
            update_history=False,
            timestep=timestep,  # NEW: Pass timestep for adaptive smoothing
            return_residual=return_residual  # ✅ 新增：支持返回residual
        )

        # ✅ 处理返回值（根据return_residual参数）
        if return_residual:
            smoothed_out, residual = result
        else:
            smoothed_out = result

        # 4. 恢复原始缓冲区（真实历史保持不变，临时缓冲区中的噪声帧被丢弃）
        self.temporal_adapter.history_buffer = original_buffer

        # 5. 只有在最后一步，才把真正去噪完成的动作存入真实历史
        # ✅ CRITICAL: 存储原始输出（训练-推理分布一致性）
        # 核心原理（基于o7模型实验验证）：
        #   实验证据：存储smoothed_out导致residual_scale从-0.5降到-1.5
        #   → Mamba在训练中主动降低自己的影响（输出有害）
        #   → 证明"递归平滑陷阱"真实存在
        #
        # 为什么存储suffix_out（Raw）？
        #   1. 避免IIR递归平滑：防止Smooth(Smooth(...))的无限嵌套
        #   2. 训练-推理一致性：Mamba应该处理Raw序列（网络原始预测）
        #   3. Mamba角色：平滑器，不是轨迹预测器
        #
        # 预期效果：存储Raw后，下次训练residual_scale应该增长而非减小
        if is_final_step:
            # 只存储实际执行的帧（避免未来预测污染历史）
            # 🔥 关键修复：避免Mamba状态断裂
            # 数学原理：h_t应该只包含实际执行的动作历史
            # 否则会产生h_belief ≠ h_actual的状态误差
            steps_to_store = self.config.n_action_steps
            for t in range(steps_to_store):
                frame = suffix_out[:, t:t+1, :].detach()
                self.temporal_adapter.history_buffer.append(frame)

        # ✅ 根据参数返回
        if return_residual:
            return smoothed_out, residual
        return smoothed_out

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        cached_vlm_features=None,  # ✅ FIX 3: 接收缓存的VLM特征
        is_final_step: bool = False,  # ✅ FIX: 标记是否为扩散去噪的最后一步
        step_idx: int = 0,  # ✅ FIX: 添加step_idx参数用于调试日志
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # 🔥 ACG: Post-Smoothing + Manual History Correction
        # Based on ACG paper (Equation 8, Algorithm 1) + Expert analysis
        # ✅ FIX: 复用Mamba残差以消除Timestep Conditioning引入的配对失效
        if self.config.use_acg and not self.training:
            # 1. Normal path: Normal Transformer attention + Mamba smoothing (with real history, but NO update)
            outputs_normal, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
                use_incoherent_attention=False,  # ✅ Normal attention
                cached_image_hidden_states=cached_vlm_features,
            )
            suffix_out_normal = outputs_normal[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_raw_normal = self.action_out_proj(suffix_out_normal)

            mamba_residual = None  # 初始化Mamba残差
            if self.temporal_adapter is not None:
                # ✅ Key Fix 1: 获取Normal路径的Mamba残差（用于Incoherent路径复用）
                # Rationale: 确保v_diff = v_raw_normal - v_raw_incoherent（Mamba残差抵消）
                v_t_normal, mamba_residual = self._apply_temporal_adapter_with_history_control(
                    v_raw_normal,
                    is_final_step=False,  # Force no history update
                    timestep=timestep,  # NEW: Pass timestep for adaptive smoothing
                    return_residual=True  # 🔥 关键：获取Mamba残差
                )
            else:
                v_t_normal = v_raw_normal

            # 2. Incoherent path: Identity Attention + 复用Mamba残差（不重新计算）
            outputs_incoherent, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
                use_incoherent_attention=True,  # ✅ Identity Attention (Equation 11 in paper)
                cached_image_hidden_states=cached_vlm_features,
            )
            suffix_out_incoherent = outputs_incoherent[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_raw_incoherent = self.action_out_proj(suffix_out_incoherent)

            if self.temporal_adapter is not None and mamba_residual is not None:
                # ✅ 🔥 CRITICAL FIX: 复用Normal路径的Mamba残差（不重新计算Mamba）
                #
                # 问题根源：
                #   Timestep Conditioning使Mamba的gate状态依赖于输入：
                #   Gate_normal = σ(W·[v_raw_normal, t])
                #   Gate_incoh  = σ(W·[v_raw_incoherent, t])
                #   由于v_raw_normal ≠ v_raw_incoherent（Incoherent被破坏），gate不同，
                #   导致v_diff = 语义差异 + Mamba状态差异（噪声）
                #
                # 修复方案：
                #   直接加上Normal路径的Mamba残差，强制物理平滑项抵消：
                #   v_t_normal     = v_raw_normal + Δ
                #   v_t_incoherent = v_raw_incoherent + Δ  ← 同样的Δ！
                #   v_diff = (v_raw_normal + Δ) - (v_raw_incoherent + Δ)
                #          = v_raw_normal - v_raw_incoherent  ← 纯净语义梯度
                #
                # 数学正确性：
                #   ACG论文假设共享历史时Mamba输出应该相同，但Timestep Conditioning破坏了这个假设。
                #   通过显式复用残差，我们强制恢复这个假设。
                v_t_incoherent = v_raw_incoherent + mamba_residual
            else:
                v_t_incoherent = v_raw_incoherent

            # 3. ACG formula (Equation 8 in paper)
            lambda_acg = self.config.acg_guidance_scale
            v_t = (1 + lambda_acg) * v_t_normal - lambda_acg * v_t_incoherent

            # 4. ✅ CRITICAL FIX: 存储v_raw_normal而非v_t（避免ACG污染历史）
            #
            # 问题：
            #   如果存储v_t = (1+λ)*v_normal - λ*v_incoherent，
            #   下一个denoise step的Mamba会基于这个"虚构"的历史进行平滑。
            #   但这个v_t可能不是机器人实际执行的动作（还有后续去噪步骤）。
            #
            # 修复：
            #   存储v_raw_normal（Normal路径的Transformer原始输出，未被ACG修改）
            #   这样历史只包含"真实"的网络预测，而非ACG的假设性修正。
            #
            # 数学原理：
            #   Mamba应该平滑"实际发生"的动作序列，而不是"可能发生"的动作。
            #   v_raw_normal是Normal路径的原始预测，更接近训练时的分布。
            #   这确保历史存储与Normal路径一致（都存储Mamba前的原始输入）。
            if is_final_step and self.temporal_adapter is not None:
                # 🔥 核心修复：只存储实际执行的视界 (n_action_steps)
                # 必须与 train_mambavision.sh 中的 N_ACTION_STEPS 一致
                # 否则 Soft Reset 取出的就是未来的幻觉
                # Bug原因：T=0预测50步存入buffer，T=20时机器人只执行了20步
                #         Soft Reset取buffer[-4:]=[a_46,a_47,a_48,a_49]是未来的幻觉
                steps_to_store = self.config.n_action_steps

                # 防御性截断，防止越界
                steps_to_store = min(steps_to_store, v_raw_normal.shape[1])

                for t in range(steps_to_store):
                    frame = v_raw_normal[:, t:t+1, :].detach()  # ✅ 存储原始Normal输出
                    self.temporal_adapter.history_buffer.append(frame)

            return v_t
        else:
            # Normal path (training or ACG disabled)
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
                cached_image_hidden_states=cached_vlm_features,  # ✅ FIX 6: 传入缓存的VLM特征
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)

            # ✅ 修复：先投影到动作空间，再应用Mamba平滑
            v_t = self.action_out_proj(suffix_out)  # [B, L, action_dim]

            # 🔥 SSM2 Temporal Smoothing（在动作空间）
            # ✅ FIX: 使用统一的辅助函数管理历史
            if self.temporal_adapter is not None:
                v_t = self._apply_temporal_adapter_with_history_control(
                    v_t,  # 传入动作而非隐特征
                    is_final_step=is_final_step,
                    timestep=timestep  # NEW: Pass inference timestep [B] for adaptive smoothing
                )

            return v_t
