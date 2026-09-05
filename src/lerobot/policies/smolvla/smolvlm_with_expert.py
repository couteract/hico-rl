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

import copy
import re

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

# LoRA is optional. Base SmolVLA remains importable without PEFT; the
# dependency is required only when ``freeze_vision_encoder=False``.
try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ModuleNotFoundError:  # pragma: no cover - minimal installs
    LoraConfig = get_peft_model = PeftModel = None


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
        acg_target_layers: tuple = (4, 5, 6),  # 🔥 ACG target layers
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers

        # ✅ 修复：为所有层添加真正的 Cross-Attention 投影
        if "cross" in attention_mode:
            vlm_hidden_size = config.text_config.hidden_size  # VLM 完整维度（960 for Gemma-2B）
            expert_hidden_size = lm_expert_config.hidden_size  # Expert hidden 维度（480）

            # 🆕 计算expert的实际head_dim和kv_dim
            num_attention_heads = lm_expert_config.num_attention_heads  # 15
            num_kv_heads = lm_expert_config.num_key_value_heads  # 5
            expert_head_dim = expert_hidden_size // num_attention_heads  # 480/15=32
            expert_kv_dim = num_kv_heads * expert_head_dim  # 5*32=160

            print(f"[Cross-Attention 修复] 初始化 {len(self.lm_expert.layers)} 层的 cross-attention 投影")
            print(f"  VLM 维度: {vlm_hidden_size}")
            print(f"  Expert 维度: hidden={expert_hidden_size}, kv={expert_kv_dim}, head_dim={expert_head_dim}")
            print(f"  Expert heads: Q={num_attention_heads}, KV={num_kv_heads}")

            for layer_idx, layer in enumerate(self.lm_expert.layers):
                # 🆕 致命修复：Cross-Attention 需要独立的 Q/K/V 投影
                # 注意：使用 GQA (Grouped Query Attention)
                # Q 有 num_attention_heads 个头，K/V 有 num_key_value_heads 个头

                # Q 从 expert_hidden_size 投影到 num_attention_heads * head_dim
                # 这样 Q 可以有完整的 15 个头
                layer.cross_attn_q_proj = nn.Linear(expert_hidden_size, expert_hidden_size, bias=False)

                # K/V 从 vlm_hidden_size 投影到 expert_kv_dim (5 heads * 64 head_dim)
                layer.cross_attn_k_proj = nn.Linear(vlm_hidden_size, expert_kv_dim, bias=False)
                layer.cross_attn_v_proj = nn.Linear(vlm_hidden_size, expert_kv_dim, bias=False)

                # O 从 expert_hidden_size 投影回 expert_hidden_size
                layer.cross_attn_o_proj = nn.Linear(expert_hidden_size, expert_hidden_size, bias=False)

                # 添加 LayerNorm 防止分布爆炸
                layer.cross_attn_norm = nn.LayerNorm(vlm_hidden_size, eps=1e-5)

                # ✅ 修复：确保所有层的 dtype 与 expert_layer 其他层一致
                # 获取 expert_layer 的 dtype（通常是 bfloat16）
                expert_dtype = layer.self_attn.q_proj.weight.dtype
                layer.cross_attn_norm = layer.cross_attn_norm.to(dtype=expert_dtype)
                layer.cross_attn_q_proj = layer.cross_attn_q_proj.to(dtype=expert_dtype)
                layer.cross_attn_k_proj = layer.cross_attn_k_proj.to(dtype=expert_dtype)
                layer.cross_attn_v_proj = layer.cross_attn_v_proj.to(dtype=expert_dtype)
                layer.cross_attn_o_proj = layer.cross_attn_o_proj.to(dtype=expert_dtype)

                # 小心初始化（避免早期训练不稳定）
                nn.init.xavier_uniform_(layer.cross_attn_q_proj.weight, gain=0.5)
                nn.init.xavier_uniform_(layer.cross_attn_k_proj.weight, gain=0.5)
                nn.init.xavier_uniform_(layer.cross_attn_v_proj.weight, gain=0.5)
                # 🔧 FIX: o_proj 使用小初始化代替零初始化，打破梯度死锁
                # 零初始化会阻止梯度回传到 LayerNorm，导致 Cross-Attention 无法训练
                nn.init.normal_(layer.cross_attn_o_proj.weight, mean=0.0, std=0.01)
                if layer.cross_attn_o_proj.bias is not None:
                    nn.init.zeros_(layer.cross_attn_o_proj.bias)  # bias 保持零（如果存在）

            print(f"[Cross-Attention 修复] 所有 {len(self.lm_expert.layers)} 层初始化完成（包含 o_proj）")

            # ✅ 可学习的Cross-Attention缩放参数 (每层独立)
            # 🔥 修复：回退到72%基线代码的快速激活策略
            # 初始化为0.5 (sigmoid≈0.62), 训练初期就注入62%视觉信号
            # 这是72%成功代码的策略，适合小数据集微调
            self.cross_attn_scales = nn.Parameter(torch.ones(len(self.lm_expert.layers)) * 0.5)
            print(f"✓ 初始化Cross-Attention scales: {len(self.lm_expert.layers)} 层, 初始值=0.5 (72%基线策略)")
            print(f"  sigmoid(0.5)={torch.sigmoid(torch.tensor(0.5)):.4f} (~62%视觉权重，快速激活)")

        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size

        # 🔥 ACG Target Layers - with auto-calculation for different model depths
        # ✅ FIX: Convert to tuple of ints (handles JSON loading, CLI args, etc.)
        if acg_target_layers is not None:
            # Handle string input (e.g., from CLI: "10,11,12,13" or "(10,11,12,13)")
            if isinstance(acg_target_layers, str):
                cleaned = acg_target_layers.strip()
                cleaned = cleaned.strip("[]()")  # Remove surrounding brackets/parens
                pieces = [p.strip() for p in re.split(r"[,\s]+", cleaned) if p.strip()]
                self.acg_target_layers = tuple(int(x) for x in pieces) if pieces else ()
            # Handle list/tuple that might contain strings or mixed types
            elif hasattr(acg_target_layers, '__iter__'):
                try:
                    # Filter out any empty strings, commas, or whitespace
                    clean_items = []
                    for item in acg_target_layers:
                        if isinstance(item, str):
                            # Split on comma/whitespace in case items like "10,11" exist
                            sub_items = [s.strip() for s in re.split(r"[,\s]+", item) if s.strip()]
                            clean_items.extend(sub_items)
                        else:
                            clean_items.append(item)
                    self.acg_target_layers = tuple(int(x) for x in clean_items)
                except (ValueError, TypeError) as e:
                    print(f"[ACG] ERROR parsing acg_target_layers={acg_target_layers}: {e}")
                    print(f"[ACG] Type: {type(acg_target_layers)}, Contents: {list(acg_target_layers) if hasattr(acg_target_layers, '__iter__') else acg_target_layers}")
                    self.acg_target_layers = None
            else:
                # Single integer value
                self.acg_target_layers = (int(acg_target_layers),)
        else:
            self.acg_target_layers = None

        # ✅ FIX: Auto-select middle-later layers if not specified or config is for wrong depth
        if self.acg_target_layers is None or len(self.acg_target_layers) == 0:
            # Auto-select middle-later 37.5% of layers (matching ACG paper proportion)
            num_layers = self.num_expert_layers
            start_layer = num_layers // 2  # Start at 50%
            end_layer = int(num_layers * 0.8125)  # End at ~81%
            self.acg_target_layers = tuple(range(start_layer, end_layer))
            print(f"[ACG] Auto-selected target layers {self.acg_target_layers} "
                  f"out of {num_layers} total expert layers (middle-later 37.5%)")
        elif max(self.acg_target_layers) >= self.num_expert_layers:
            # Config has invalid layer indices - auto-correct
            print(f"[ACG] WARNING: acg_target_layers {self.acg_target_layers} contains indices "
                  f">= num_expert_layers {self.num_expert_layers}. Auto-correcting...")
            num_layers = self.num_expert_layers
            start_layer = num_layers // 2
            end_layer = int(num_layers * 0.8125)
            self.acg_target_layers = tuple(range(start_layer, end_layer))
            print(f"[ACG] Corrected to {self.acg_target_layers}")

        # ✅ P2: 验证日志 - 打印ACG层配置统计信息
        if self.acg_target_layers is not None and len(self.acg_target_layers) > 0:
            print(f"[ACG P2] Final target layers: {self.acg_target_layers}")
            print(f"[ACG P2] Expert layers: {self.num_expert_layers}")
            coverage_pct = len(self.acg_target_layers) / self.num_expert_layers * 100
            print(f"[ACG P2] Coverage: {len(self.acg_target_layers)}/{self.num_expert_layers} "
                  f"= {coverage_pct:.1f}%")
            min_layer = min(self.acg_target_layers)
            max_layer = max(self.acg_target_layers)
            min_pct = min_layer / self.num_expert_layers * 100
            max_pct = max_layer / self.num_expert_layers * 100
            print(f"[ACG P2] Layer range: {min_layer}-{max_layer} "
                  f"({min_pct:.1f}%-{max_pct:.1f}%)")

        # ========== LoRA Vision Encoder Fine-tuning ==========
        # Inject LoRA adapters when vision encoder is unfrozen
        if not freeze_vision_encoder:
            if LoraConfig is None:
                raise ImportError(
                    "PEFT is required when freeze_vision_encoder=False. "
                    "Install the SmolVLA/PEFT dependencies first."
                )
            print("=" * 70)
            print("🚀 [SmolVLA] Detected vision unfrozen request, injecting LoRA adapters...")
            print("=" * 70)

            # Configure LoRA parameters
            # PEFT requires explicit layer names (doesn't support wildcards)
            # Build target_modules list for all 12 vision encoder layers
            target_modules = []
            for layer_idx in range(12):  # SigLIP has 12 encoder layers
                target_modules.extend([
                    f"vision_model.encoder.layers.{layer_idx}.self_attn.q_proj",
                    f"vision_model.encoder.layers.{layer_idx}.self_attn.k_proj",
                    f"vision_model.encoder.layers.{layer_idx}.self_attn.v_proj",
                    f"vision_model.encoder.layers.{layer_idx}.self_attn.out_proj",
                ])
            # Add connector (critical for modality alignment!)
            target_modules.append("connector.modality_projection.proj")

            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
                modules_to_save=None,
            )

            # Inject LoRA into self.vlm
            self.vlm = get_peft_model(self.vlm, peft_config)

            # ✅ Verification: must see trainable% < 1%
            print("\n✅ LoRA injection successful! Trainable parameters:")
            self.vlm.print_trainable_parameters()

            # Double safety: ensure non-LoRA parameters are frozen
            for name, param in self.vlm.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False

            print("✅ LoRA double verification: non-LoRA parameters frozen")
            print("=" * 70)

        self.set_requires_grad()

    def get_vlm_model(self):
        # Handle both PeftModel and regular model
        if PeftModel is not None and isinstance(self.vlm, PeftModel):
            # PeftModel wraps: peft_model.base_model.model = original model
            # original model.model = SmolVLMModel (has vision_model)
            return self.vlm.base_model.model.model
        else:
            return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            # Traditional freezing: completely freeze vision_model
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        else:
            # LoRA mode or full unfrozen
            if PeftModel is not None and isinstance(self.vlm, PeftModel):
                # ✅ LoRA mode: keep base model in eval mode (BatchNorm, etc.)
                # but LoRA layers remain trainable
                # get_vlm_model() handles PeftModel wrapping
                self.get_vlm_model().vision_model.eval()
                print("✅ LoRA mode: Vision encoder in eval mode, LoRA layers trainable")
            else:
                # ⚠️ Full unfrozen mode (not recommended, will OOM)
                print("⚠️ WARNING: Fully unfreezing vision encoder, may cause OOM")
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        # Always keep vision model in eval mode (LoRA or frozen)
        # This ensures BatchNorm/LayerNorm statistics don't update
        # get_vlm_model() handles PeftModel wrapping
        if self.freeze_vision_encoder or (PeftModel is not None and isinstance(self.vlm, PeftModel)):
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    # ⚠️ 已删除 forward_attn_layer 方法（互斥分支已被串行架构替代）

    def _forward_vlm_once(
        self,
        vlm_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_incoherent_attention: bool = False,
    ) -> torch.Tensor:
        """
        一次性运行完整的 VLM，返回最后一层的 hidden_states。

        Args:
            vlm_inputs: [B, seq_len, vlm_hidden_size] VLM 输入
            attention_mask: [B, total_seq_len, total_seq_len] 注意力 mask
            position_ids: [B, total_seq_len] 位置编码
            use_incoherent_attention: 是否使用 ACG

        Returns:
            image_hidden_states: [B, seq_len, vlm_hidden_size] VLM 最终输出
        """
        vlm_model = self.get_vlm_model().text_model

        # 获取VLM层的目标dtype（应该是bfloat16）
        target_dtype = vlm_model.layers[0].self_attn.q_proj.weight.dtype

        # ✅ FIX: Convert input to target dtype at the start to prevent dtype mismatches
        # vlm_inputs is float32 but VLM expects bfloat16
        vlm_hidden_states = vlm_inputs.to(dtype=target_dtype)

        # 提取 VLM 的 attention mask 和 position_ids
        seq_len = vlm_inputs.shape[1]
        vlm_attention_mask = attention_mask[:, :seq_len, :seq_len]
        vlm_position_ids = position_ids[:, :seq_len]
        batch_size = vlm_inputs.shape[0]
        head_dim = self.vlm.config.text_config.head_dim

        # 运行所有 VLM 层
        for vlm_layer_idx in range(self.num_vlm_layers):
            vlm_layer = vlm_model.layers[vlm_layer_idx]

            # LayerNorm
            normed_hidden_states = vlm_layer.input_layernorm(vlm_hidden_states)

            # Self-Attention
            input_shape = normed_hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, vlm_layer.self_attn.head_dim)

            normed_hidden_states = normed_hidden_states.to(dtype=vlm_layer.self_attn.q_proj.weight.dtype)
            query_states = vlm_layer.self_attn.q_proj(normed_hidden_states).view(hidden_shape)
            key_states = vlm_layer.self_attn.k_proj(normed_hidden_states).view(hidden_shape)
            value_states = vlm_layer.self_attn.v_proj(normed_hidden_states).view(hidden_shape)

            # Apply RoPE
            query_states = apply_rope(query_states, vlm_position_ids)
            key_states = apply_rope(key_states, vlm_position_ids)

            # Self-Attention
            attn_output = self.get_attention_interface()(
                vlm_attention_mask,
                batch_size=batch_size,
                head_dim=head_dim,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                use_incoherent_attention=use_incoherent_attention,
                vlm_seq_len=0,  # VLM processing doesn't need ACG block-diagonal masking
            )

            # Output projection
            if attn_output.dtype != vlm_layer.self_attn.o_proj.weight.dtype:
                attn_output = attn_output.to(vlm_layer.self_attn.o_proj.weight.dtype)
            attn_output = vlm_layer.self_attn.o_proj(attn_output)

            # ✅ FIX: Ensure both operands are same dtype before residual connection
            # Logs show: float32 + bfloat16 → float32 (type promotion causes mismatch)
            if vlm_hidden_states.dtype != attn_output.dtype:
                vlm_hidden_states = vlm_hidden_states.to(dtype=attn_output.dtype)
            vlm_hidden_states = attn_output + vlm_hidden_states

            # FFN
            after_attn_residual = vlm_hidden_states.clone()
            vlm_hidden_states = vlm_layer.post_attention_layernorm(vlm_hidden_states)

            # ✅ FIX: Ensure dtype matches MLP weights before calling MLP
            # Logs show: vlm_hidden_states is float32 but MLP expects bfloat16
            mlp_dtype = vlm_layer.mlp.gate_proj.weight.dtype
            if vlm_hidden_states.dtype != mlp_dtype:
                vlm_hidden_states = vlm_hidden_states.to(dtype=mlp_dtype)

            vlm_hidden_states = vlm_layer.mlp(vlm_hidden_states)
            vlm_hidden_states = vlm_hidden_states + after_attn_residual

        return vlm_hidden_states  # [B, seq_len, vlm_hidden_size]

    # ⚠️ 已删除 forward_cross_attn_layer 方法（互斥分支已被串行架构替代）

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        use_incoherent_attention: bool = False,  # 🔥 ACG support
        cached_image_hidden_states: torch.Tensor | None = None,  # 🆕 VLM特征缓存
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        head_dim = self.vlm.config.text_config.head_dim

        # 判断是否使用 cross-attention
        use_cross_attn = "cross" in self.attention_mode

        # ===============================
        # 步骤 1：获取或计算 VLM 特征
        # ===============================
        image_hidden_states = None
        if use_cross_attn:
            if cached_image_hidden_states is not None:
                # 推理时：使用缓存的VLM特征
                image_hidden_states = cached_image_hidden_states
            elif len(inputs_embeds) == 2 and inputs_embeds[0] is not None:
                # 训练时：一次性计算VLM特征
                image_hidden_states = self._forward_vlm_once(
                    vlm_inputs=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_incoherent_attention=False,  # VLM不应用ACG
                )
        if len(inputs_embeds) == 2 and inputs_embeds[1] is None:
            return [image_hidden_states, None], past_key_values
        # ===============================
        # 步骤 2：提取Expert的attention mask和position_ids
        # ===============================
        if len(inputs_embeds) == 2:
            # 🔥 FIX: 当使用缓存的VLM特征时，从缓存或attention_mask推断vlm_seq_len
            if inputs_embeds[0] is not None:
                vlm_seq_len = inputs_embeds[0].shape[1]
            elif cached_image_hidden_states is not None:
                # 推理时使用缓存，从缓存特征获取序列长度
                vlm_seq_len = cached_image_hidden_states.shape[1]
            else:
                vlm_seq_len = 0
            expert_seq_len = inputs_embeds[1].shape[1]
        else:
            vlm_seq_len = 0
            expert_seq_len = inputs_embeds[0].shape[1]

        # 提取Expert的attention mask（用于Self-Attention）
        # 🔥 FIX: attention_mask 的形状是 [batch, query_len, key_len]
        # 在推理时：query_len = expert_seq_len, key_len = vlm_seq_len + expert_seq_len
        # 我们需要提取 expert 对 expert 的部分：[:, :, vlm_seq_len:]
        if vlm_seq_len > 0:
            # attention_mask 的 query 已经是 expert，只需要切片 key 的 expert 部分
            if attention_mask.shape[1] == expert_seq_len:
                # 推理模式：attention_mask = [batch, expert_seq, vlm_seq + expert_seq]
                expert_attention_mask = attention_mask[:, :, vlm_seq_len:]
            else:
                # 训练模式：attention_mask = [batch, total_seq, total_seq]
                expert_attention_mask = attention_mask[:, vlm_seq_len:, vlm_seq_len:]

            # 🔥 FIX: position_ids 也需要同样的处理
            if position_ids.shape[1] == expert_seq_len:
                # 推理模式：position_ids 已经只包含 expert 部分，直接使用
                expert_position_ids = position_ids
            else:
                # 训练模式：从完整的 position_ids 中提取 expert 部分
                expert_position_ids = position_ids[:, vlm_seq_len:]
                # ✅ 修复：保持position_ids连续性,不重置为0
                # 修复前: 重置为 [0, 1, ..., 9] 导致RoPE相对位置编码断裂
                # 修复后: 保持 [800, 801, ..., 809] 使Cross-Attention能正确建模时序关系
                # 注释掉重置逻辑:
                # if expert_position_ids.shape[1] > 0:
                #     expert_position_ids = expert_position_ids - torch.min(expert_position_ids, dim=1, keepdim=True).values
        else:
            expert_attention_mask = attention_mask
            expert_position_ids = position_ids

        # ===============================
        # 步骤 3：串行处理Expert层（Self-Attn → Cross-Attn → MLP）
        # ===============================
        num_layers = self.num_expert_layers
        attention_interface = self.get_attention_interface()

        # 获取Expert的hidden_states
        expert_hidden_states = inputs_embeds[1] if len(inputs_embeds) == 2 else inputs_embeds[0]

        for layer_idx in range(num_layers):
            expert_layer = self.lm_expert.layers[layer_idx]
            is_acg_layer = use_incoherent_attention and layer_idx in self.acg_target_layers

            # ============ 步骤 3.1: Self-Attention（永远执行） ============
            residual = expert_hidden_states
            expert_hidden_states = expert_layer.input_layernorm(expert_hidden_states)

            # Q, K, V投影
            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)
            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)

            query_states = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)
            key_states = expert_layer.self_attn.k_proj(expert_hidden_states).view(expert_hidden_shape)
            value_states = expert_layer.self_attn.v_proj(expert_hidden_states).view(expert_hidden_shape)

            # 应用RoPE
            query_states = apply_rope(query_states, expert_position_ids)
            key_states = apply_rope(key_states, expert_position_ids)

            # Self-Attention计算（应用ACG在指定层）
            attn_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                query_states,
                key_states,
                value_states,
                use_incoherent_attention=is_acg_layer,  # 🔥 ACG在Self-Attn
                vlm_seq_len=vlm_seq_len,  # 🔥 ACG: Pass VLM sequence length
            )

            # Output projection + residual
            if attn_output.dtype != expert_layer.self_attn.o_proj.weight.dtype:
                attn_output = attn_output.to(expert_layer.self_attn.o_proj.weight.dtype)
            attn_output = expert_layer.self_attn.o_proj(attn_output)
            expert_hidden_states = residual + attn_output

            # ============ 步骤 3.2: Cross-Attention（如果有VLM特征） ============
            if use_cross_attn and image_hidden_states is not None:
                residual = expert_hidden_states
                expert_hidden_states = expert_layer.post_attention_layernorm(expert_hidden_states)

                # K, V来自VLM特征
                if hasattr(expert_layer, 'cross_attn_q_proj') and hasattr(expert_layer, 'cross_attn_k_proj') and hasattr(expert_layer, 'cross_attn_v_proj'):
                    # 🆕 Q来自Self-Attention的输出，使用独立的 cross_attn_q_proj
                    expert_hidden_states_normed = expert_hidden_states.to(dtype=expert_layer.cross_attn_q_proj.weight.dtype)

                    # 🆕 Cross-Attention 使用 GQA (Grouped Query Attention)
                    # Q 有 num_attention_heads 个头，K/V 有 num_key_value_heads 个头
                    num_attention_heads = self.lm_expert.config.num_attention_heads  # 15
                    num_kv_heads = self.lm_expert.config.num_key_value_heads  # 5
                    # 🆕 计算expert的实际head_dim: hidden_size / num_attention_heads
                    expert_head_dim = self.expert_hidden_size // num_attention_heads  # 480/15=32

                    # Q: [batch, expert_seq, 15 heads, 64 head_dim]
                    cross_query_shape = (*expert_hidden_states_normed.shape[:-1], num_attention_heads, expert_head_dim)
                    cross_q = expert_layer.cross_attn_q_proj(expert_hidden_states_normed).view(cross_query_shape)

                    # 应用RoPE到Q
                    cross_q = apply_rope(cross_q, expert_position_ids)

                    # K, V来自VLM特征: [batch, vlm_seq, 5 heads, 64 head_dim]
                    vlm_features = image_hidden_states.to(dtype=expert_layer.cross_attn_k_proj.weight.dtype)
                    vlm_features_normed = expert_layer.cross_attn_norm(vlm_features)

                    cross_k = expert_layer.cross_attn_k_proj(vlm_features_normed).view(
                        *vlm_features_normed.shape[:-1], num_kv_heads, expert_head_dim
                    )
                    cross_v = expert_layer.cross_attn_v_proj(vlm_features_normed).view(
                        *vlm_features_normed.shape[:-1], num_kv_heads, expert_head_dim
                    )

                    # Cross-Attention计算（不应用ACG，保持视觉引导完整）
                    # attention_interface 会自动处理 GQA (Q有15头，K/V有5头)
                    cross_attn_output = attention_interface(
                        None,  # Cross-Attn通常全可见
                        batch_size,
                        expert_head_dim,  # 64
                        cross_q,
                        cross_k,
                        cross_v,
                        use_incoherent_attention=False,  # Cross-Attn保持正常
                        vlm_seq_len=0,  # Cross-Attn doesn't need block-diagonal masking
                    )

                    # 🆕 致命修复：使用独立的 cross_attn_o_proj 而非 self_attn.o_proj
                    # Output projection（将attention输出投影到Expert hidden_size）
                    # expert_kv_dim (320/480) → expert_hidden_size (480/720)
                    if cross_attn_output.dtype != expert_layer.cross_attn_o_proj.weight.dtype:
                        cross_attn_output = cross_attn_output.to(expert_layer.cross_attn_o_proj.weight.dtype)
                    cross_attn_output = expert_layer.cross_attn_o_proj(cross_attn_output)

                    # 残差连接 + 可学习缩放
                    # ✅ 修复：使用可学习的per-layer缩放,替代硬编码的2.0倍
                    # 修复前: residual + 2.0 * cross_attn_output (固定缩放,导致梯度失衡)
                    # 修复后: residual + learnable_scale * cross_attn_output (自适应学习)
                    # Scale范围: sigmoid(0) = 0.5 (Flamingo标准,初始温和注入视觉信号)
                    # 训练过程中scale可学习地调整到(0, 1)范围,避免初始权重休克
                    if hasattr(self, 'cross_attn_scales'):
                        # 使用可学习缩放: sigmoid映射到(0,1),初始0.5 (从Flamingo的0开始)
                        scale = torch.sigmoid(self.cross_attn_scales[layer_idx])
                        expert_hidden_states = residual + scale * cross_attn_output
                    else:
                        # 向后兼容: 如果模型没有cross_attn_scales参数(旧checkpoint)
                        expert_hidden_states = residual + cross_attn_output
                else:
                    # 如果没有cross_attn投影层，保持原样
                    expert_hidden_states = residual + expert_hidden_states

            # ============ 步骤 3.3: MLP（永远执行） ============
            residual = expert_hidden_states
            expert_hidden_states = expert_layer.post_attention_layernorm(expert_hidden_states)

            # Dtype转换（MLP期望bfloat16）
            if expert_hidden_states.dtype != expert_layer.mlp.gate_proj.weight.dtype:
                expert_hidden_states = expert_hidden_states.to(expert_layer.mlp.gate_proj.weight.dtype)
            expert_hidden_states = expert_layer.mlp(expert_hidden_states)

            # Dtype转换residual
            if residual.dtype != expert_hidden_states.dtype:
                residual = residual.to(expert_hidden_states.dtype)
            expert_hidden_states = residual + expert_hidden_states

        # 构建输出
        outputs_embeds = []
        if len(inputs_embeds) == 2:
            # 两个输入：VLM + Expert
            if image_hidden_states is not None:
                outputs_embeds.append(image_hidden_states)
            else:
                outputs_embeds.append(inputs_embeds[0])
            outputs_embeds.append(expert_hidden_states)
        else:
            # 单个输入：只有Expert
            outputs_embeds.append(expert_hidden_states)
        inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
        use_incoherent_attention: bool = False,  # 🔥 ACG support
        vlm_seq_len: int = 0,  # 🔥 ACG: VLM sequence length for block-diagonal mask
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        # 🔥 ACG: Incoherent Attention with Block-Diagonal Mask
        # ✅ 统一Softmax域 + 块对角掩码（保持感知-动作耦合，破坏时序连贯性）
        if use_incoherent_attention and vlm_seq_len > 0:
            query_len = query_states.shape[2]
            key_len = key_states.shape[2]

            # 🔥 ACG: 计算 Expert keys 在 key_states 中的偏移量（Inference Offset）
            # 训练时: query_len = 562 (VLM + Expert), key_len = 562, offset = 0
            # 推理时: query_len = 50 (Expert only), key_len = 562 (VLM cached + Expert), offset = 512
            # offset 表示 Expert keys 在 key_states 中的起始位置
            offset = key_len - query_len

            # === Step 1: 计算完整的attention weights（整个key space）===
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights = attn_weights / (head_dim ** 0.5)
            attn_weights = attn_weights.to(dtype=torch.float32)

            # === Step 2: 创建块对角掩码（Block-Diagonal Mask）===
            # 目标：
            # - 所有 queries: 看所有 VLM keys (0 到 offset-1)
            # - 所有 queries: 只看自己对应的 Expert key (offset + i)
            #
            # 推理时示例:
            # Query 0 → Keys: [0..511] (VLM 全部) + Key 512 (对应的 Expert)
            # Query 1 → Keys: [0..511] (VLM 全部) + Key 513 (对应的 Expert)
            # ...
            # Query 49 → Keys: [0..511] (VLM 全部) + Key 561 (对应的 Expert)

            # 初始化：允许所有attention
            block_mask = torch.ones(
                batch_size, query_states.shape[1], query_len, key_len,
                device=query_states.device, dtype=torch.bool
            )

            # 应用块对角约束
            if offset == 0:
                # 🔥 训练时: query 包含 VLM + Expert, key 包含 VLM + Expert
                # - VLM queries (0..vlm_seq_len-1): 保持全 True (正常 attention)
                # - Expert queries (vlm_seq_len..query_len-1): VLM 部分 True, Expert 部分对角线
                expert_len = query_len - vlm_seq_len
                if expert_len > 0:
                    expert_identity = torch.eye(expert_len, device=query_states.device, dtype=torch.bool)
                    expert_identity = expert_identity.unsqueeze(0).unsqueeze(0).expand(
                        batch_size, query_states.shape[1], -1, -1
                    )
                    # 只修改 Expert queries 看 Expert keys 的部分
                    block_mask[:, :, vlm_seq_len:, vlm_seq_len:] = expert_identity
            else:
                # 🔥 推理时: query 只有 Expert, key 包含 VLM + Expert
                # - 所有 queries: 看所有 VLM keys (0..offset-1), 只看对应的 Expert key (offset+i)
                expert_identity = torch.eye(query_len, device=query_states.device, dtype=torch.bool)
                # 将 Expert keys 区域 (offset:) 替换为单位阵
                # 这确保 Query i 只能看到 Key (offset + i)
                block_mask[:, :, :, offset:] = expert_identity.unsqueeze(0).unsqueeze(0)

            # === Step 3: 应用掩码并计算统一的Softmax ===
            big_neg = torch.finfo(attn_weights.dtype).min

            # 首先应用块对角掩码
            masked_weights = torch.where(block_mask, attn_weights, big_neg)

            # 然后应用原始attention mask（padding等）
            masked_weights = torch.where(attention_mask[:, None, :, :], masked_weights, big_neg)

            # 🔥 ACG权重校准：补偿Softmax域大小差异
            # 问题：Incoherent路径的Softmax域 = 513 (VLM:512 + Action:1)
            #       Normal路径的Softmax域 = 562 (VLM:512 + Action:50)
            # 结果：Incoherent路径中VLM权重虚高（99.8% vs 91%）
            # 解决：在log域缩放VLM权重，使两条路径的VLM权重期望值一致
            if offset > 0:  # Incoherent推理模式
                # 计算有效Softmax域大小
                effective_keys_incoherent = offset + 1  # VLM (offset) + 1 Action per query
                effective_keys_normal = key_len  # 所有keys

                # Log域缩放因子：log(513/562) ≈ -0.091
                # 降低VLM部分的竞争优势，使其在softmax后的权重与normal路径一致
                log_scale = torch.log(
                    torch.tensor(
                        effective_keys_incoherent / effective_keys_normal,
                        dtype=attn_weights.dtype,
                        device=attn_weights.device
                    )
                )

                # 应用到VLM部分（前offset个keys）
                # 注意：我们在log域加上负值（-0.091），相当于在概率域乘以0.913
                masked_weights[:, :, :, :offset] = masked_weights[:, :, :, :offset] + log_scale

            # === Step 4: 统一Softmax（在整个key space上归一化）===
            # 现在两条路径的VLM权重期望值一致，ACG只增强Action连贯性而不削弱视觉
            probs = nn.functional.softmax(masked_weights, dim=-1)
            probs = probs.to(dtype=value_states.dtype)

            # === Step 5: 应用attention到values ===
            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        elif use_incoherent_attention:
            # 🔥 ACG: If we reach here with use_incoherent_attention=True but vlm_seq_len=0,
            # something is wrong with the vlm_seq_len calculation from past_key_values or inputs_embeds
            raise RuntimeError(
                f"ACG requires vlm_seq_len > 0, but got vlm_seq_len={vlm_seq_len}. "
                f"This indicates a bug in vlm_seq_len calculation. "
                f"Check that past_key_values contains cached keys for this layer, "
                f"or that inputs_embeds[0] is not None during training/cache fill."
            )
        else:
            # Normal attention computation
            att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            att_weights *= head_dim**-0.5

            att_weights = att_weights.to(dtype=torch.float32)

            # 处理attention_mask（可能为None，例如在Cross-Attention中）
            if attention_mask is not None:
                big_neg = torch.finfo(att_weights.dtype).min
                masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
                probs = nn.functional.softmax(masked_att_weights, dim=-1)
            else:
                # Cross-Attention通常不需要mask（全可见）
                probs = nn.functional.softmax(att_weights, dim=-1)

            probs = probs.to(dtype=value_states.dtype)

            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
