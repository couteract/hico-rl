#!/usr/bin/env bash
set -euo pipefail

# Current-machine LIBERO-10 training template.
# This reproduces the configuration recorded in the 5090 checkpoint, while
# using paths available on this machine. It does not activate or modify conda.

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhang/xzw/two/Evo-RL-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/home/zhang/xzw/datasets/libero_10_image}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
VLM_MODEL="${VLM_MODEL:-/home/zhang/xzw/two/evo-rl-jax/SmolVLM2-500M-Video-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhang/xzw/hico-rl-runs/train_libero10_mambavision}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-40000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
RESUME="${RESUME:-false}"
SEED="${SEED:-42}"

for path in "$PROJECT_ROOT/src/lerobot/scripts/lerobot_train.py" "$DATASET_ROOT" "$VLM_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" src/lerobot/scripts/lerobot_train.py \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.streaming=false \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name="$VLM_MODEL" \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.use_amp=false \
  --policy.num_vlm_layers=16 \
  --policy.num_expert_layers=-1 \
  --policy.expert_width_multiplier=1.0 \
  --policy.attention_mode=cross_attn \
  --policy.self_attn_every_n_layers=-1 \
  --policy.tokenizer_max_length=96 \
  --policy.pad_language_to=longest \
  --policy.prefix_length=-1 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.num_steps=10 \
  --policy.use_mamba_temporal=true \
  --policy.mamba_history_length=50 \
  --policy.mamba_d_state=64 \
  --policy.mamba_d_conv=4 \
  --policy.mamba_expand=2 \
  --policy.mamba_time_embed_dim=64 \
  --policy.gripper_loss_weight=3.0 \
  --policy.lora_rank=32 \
  --policy.lora_alpha=64 \
  --policy.lora_dropout=0.15 \
  --policy.optimizer_lr=2e-4 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.optimizer_grad_clip_norm=10.0 \
  --policy.scheduler_warmup_steps=3000 \
  --policy.scheduler_decay_steps=35000 \
  --policy.scheduler_decay_lr=5e-6 \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --seed="$SEED" \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --eval_freq="$EVAL_FREQ" \
  --save_checkpoint=true \
  --output_dir="$OUTPUT_DIR" \
  --resume="$RESUME"
