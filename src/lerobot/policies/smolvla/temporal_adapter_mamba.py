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

"""
Temporal Adapter using Mamba-2 for action sequence smoothing.

This module provides temporal coherence by maintaining an internal history buffer
and using Mamba-2 selective state space models to smooth action predictions.
"""

from collections import deque

import torch
import torch.nn as nn

from lerobot.policies.mambavision.models.mamba_vision import MambaVisionMixer


class TemporalMambaAdapter(nn.Module):
    """
    Sidecar Temporal Adapter using Mamba-2 for action sequence smoothing.

    Features:
    - Maintains internal history buffer (self.history_buffer)
    - Automatically accumulates history for single-frame input
    - Configurable buffer length (default: 10 frames)
    - Uses Mamba-2 selective scan for O(n) temporal modeling

    Args:
        hidden_size: Expert hidden size (e.g., 768)
        history_length: Number of historical frames to maintain (default: 10)
        d_state: Mamba state dimension (default: 64)
        d_conv: Conv kernel size (default: 4)
        expand: Expansion factor (default: 2)
    """

    def __init__(
        self,
        hidden_size: int,
        history_length: int = 20,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        time_embed_dim: int = 64,  # NEW: Timestep feature dimension
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.time_embed_dim = time_embed_dim  # NEW

        # History buffer using deque for efficient FIFO operations
        self.register_buffer("_dummy", torch.zeros(1))  # For device tracking
        self.history_buffer = deque(maxlen=history_length)

        # NEW: Timestep processing MLP
        # Converts sinusoidal timestep embedding → learned time features
        # Input: [B, hidden_size] (sinusoidal embedding)
        # Output: [B, time_embed_dim] (learned features)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # MODIFIED: Expand Mamba input dimension to include time features
        # Allows Mamba to learn adaptive smoothing based on diffusion timestep
        mamba_input_dim = hidden_size + time_embed_dim
        self.mamba_block = MambaVisionMixer(
            d_model=mamba_input_dim,  # Was: hidden_size, Now: hidden_size + time_embed_dim
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # NEW: Project Mamba output back to original dimension
        # Mamba outputs [B, L, mamba_input_dim], need to restore to [B, L, hidden_size]
        self.output_proj = nn.Linear(mamba_input_dim, hidden_size)

        # ✅ P0 FIX: 输入归一化层（防止长时间推理中的数值漂移）
        # 稳定Mamba的输入分布，避免SSM状态空间出现数值不稳定
        self.input_norm = nn.LayerNorm(hidden_size)

        # ✅ 修复历史Bug后，使用更温和的初始化
        # -2.0 → sigmoid(-2.0)*0.5 = 0.06 (6%贡献)
        # 优势：
        #   1. 训练初期不破坏预训练权重（vs -0.5的18.88%）
        #   2. 梯度足以让Mamba学习（vs -10.0的0.0045%）
        #   3. 如果Mamba学到有效特征，训练中会自动增长（→-1.5, →-1.0）
        self.residual_scale = nn.Parameter(torch.ones(1) * -2.0)

    def reset_history(self):
        """Clear the history buffer. Useful when starting a new episode."""
        self.history_buffer.clear()

    def forward(self, x, reset_history=False, update_history=True, timestep=None, return_residual=False):
        """
        Apply temporal smoothing with optional timestep conditioning.

        Args:
            x: Input features [B, L, D] - Expert layer output
            reset_history: If True, clear history buffer before processing
            update_history: If True, add current frame to history buffer (default: True)
                           Set to False to use history for smoothing without storing noise frames
            timestep: Optional [B] or [B, 1] timestep in range [0, 1]
                     If None, uses unconditional smoothing (backward compat)
            return_residual: If True, return (output, residual) tuple for ACG compatibility (default: False)
                            This allows ACG to reuse the Mamba residual across Normal/Incoherent paths

        Returns:
            smoothed: Temporally smoothed features [B, L, D]
            OR (smoothed, residual) if return_residual=True

        Process:
        1. (Optional) Clear history if reset_history=True
        2. Build full sequence from history + current chunk
        3. Apply sliding window truncation
        4. (NEW) Inject timestep conditioning via concatenation
        5. Apply Mamba temporal processing
        6. (NEW) Project back to original dimension
        7. Residual connection: output = x + scale * mamba(...)
        8. (Optional) Update history buffer
        9. (NEW) Optionally return residual for ACG path alignment
        """
        if reset_history:
            self.reset_history()

        device = x.device
        dtype = x.dtype
        batch_size, seq_len, hidden_dim = x.shape

        # ✅ 修复：检测 batch size 变化并重置历史
        # 当 batch size 改变时（例如最后一个 batch 更小），清空历史缓冲
        if len(self.history_buffer) > 0:
            prev_batch_size = self.history_buffer[-1].shape[0]
            if prev_batch_size != batch_size:
                self.reset_history()

        # ✅ 修复：滑动窗口Mamba - 一次处理整个chunk+history序列
        # Build history sequence from buffer
        if len(self.history_buffer) > 0:
            # 拼接历史帧: [B, history_len, D]
            history_seq = torch.cat(list(self.history_buffer), dim=1)

            # 拼接历史+当前chunk: [B, history_len+seq_len, D]
            full_seq = torch.cat([history_seq, x], dim=1)

            # ✅ 🔥 P0 FIX: 严格滑动窗口（Strict Sliding Window）
            # 核心修复：强制截断到训练时的长度，消除OOD噪声
            # 训练时Mamba只见过seq_len帧（通常50），推理时也必须保持一致
            # 否则Position 50+的输出是未定义的，产生OOD噪声破坏预测
            if full_seq.shape[1] > seq_len:
                full_seq = full_seq[:, -seq_len:, :]  # 只保留最后seq_len帧
        else:
            # 如果没有历史,直接处理当前chunk
            full_seq = x  # [B, seq_len, D]

        # ✅ P0 FIX: 应用输入归一化（防止数值漂移）
        full_seq = self.input_norm(full_seq)

        # ✅ NEW: Timestep conditioning for adaptive smoothing
        # Enable Mamba to adjust smoothing strength based on diffusion denoising progress
        if timestep is not None:
            # Ensure timestep shape is [B]
            if timestep.dim() == 2:
                timestep = timestep.squeeze(-1)  # [B, 1] → [B]

            # Create sinusoidal timestep embedding (same as embed_suffix)
            # Note: We use hidden_size to match the time_mlp input dimension
            from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.hidden_size,  # Use hidden_size for embedding dimension
                min_period=2,      # Match config defaults
                max_period=10000,
                device=device,
            ).type(dtype=dtype)  # [B, hidden_size]

            # Process through time_mlp to get learned time features
            time_features = self.time_mlp(time_emb)  # [B, time_embed_dim]

            # Broadcast to sequence length
            time_features = time_features.unsqueeze(1).expand(-1, full_seq.shape[1], -1)  # [B, L, time_embed_dim]

            # Concatenate with input
            full_seq_with_time = torch.cat([full_seq, time_features], dim=-1)  # [B, L, hidden_size + time_embed_dim]
        else:
            # Backward compatibility: If no timestep, pad with zeros
            zero_time_features = torch.zeros(
                batch_size, full_seq.shape[1], self.time_embed_dim,
                device=device, dtype=dtype
            )
            full_seq_with_time = torch.cat([full_seq, zero_time_features], dim=-1)

        # Apply Mamba temporal processing with time-conditioned input
        # Mamba处理完整序列,保留所有帧的时序关系
        smoothed_seq = self.mamba_block(full_seq_with_time)  # [B, L, hidden_size + time_embed_dim]

        # ✅ NEW: Project Mamba output back to original dimension
        smoothed_seq = self.output_proj(smoothed_seq)  # [B, L, hidden_size]

        # Extract smoothed features corresponding to current chunk
        # 提取对应当前chunk的平滑输出 (最后seq_len帧)
        smoothed_feature = smoothed_seq[:, -seq_len:, :]  # [B, seq_len, D]

        # Residual connection with learned scale
        # 🔴 FIX P1+P2 (Critical): 恢复scale约束和additive形式
        # 原理：
        # 1. Hard constraint (× 0.5) 强制Mamba作为"辅助去噪器"而非"主预测器"
        # 2. Additive形式 (x + α·smooth) 确保原始梯度始终完整传递
        # 3. 插值形式 (x·(1-α) + smooth·α) 会导致原始梯度被(1-α)衰减
        scale = torch.sigmoid(self.residual_scale) * 0.5  # 输出范围 [0, 0.5]
        residual = scale * smoothed_feature  # ✅ 提取Mamba残差（用于ACG配对）
        output = x + residual  # Additive residual (ResNet原理)

        # ✅ CRITICAL: 存储原始输入（训练-推理分布一致性）
        # 核心原理（基于o7模型实验验证）：
        #   实验证据：存储Smooth导致residual_scale从-0.5降到-1.5
        #   → Mamba在训练中主动降低自己的影响（输出有害）
        #   → 证明"递归平滑陷阱"（IIR滤波器）真实存在
        #
        # 为什么存储Raw？
        #   1. 训练-推理一致性：训练时Mamba看到Raw序列，推理时也应该是Raw
        #   2. 避免IIR递归平滑：Mamba(Raw历史, Raw当前) = MA滤波器（正常）
        #                      Mamba(Smooth历史, Raw当前) = IIR滤波器（动作迟滞）
        #   3. Mamba角色定位：平滑器（处理Raw序列），而非轨迹预测器
        #
        # 预期效果：存储Raw后，residual_scale应该增长（-2.0→-1.0）而非减小
        if update_history:
            for t in range(seq_len):
                # 逐帧添加原始输入到历史buffer
                frame = x[:, t:t+1, :].detach()  # 🔥 回退：存储原始输入
                self.history_buffer.append(frame)

        # ✅ ACG Support: 可选返回residual供Incoherent路径复用
        # 数学原理：确保ACG的v_diff是纯净的语义梯度
        # v_diff = (v_raw_normal + Δ) - (v_raw_incoherent + Δ) = v_raw_normal - v_raw_incoherent
        # 其中Δ是Normal路径计算出的Mamba残差，在两条路径上完全相同，因此抵消
        if return_residual:
            return output, residual  # 返回(平滑后的输出, Mamba残差)
        return output  # ✅ 向后兼容：默认只返回输出

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, history_length={self.history_length}"
