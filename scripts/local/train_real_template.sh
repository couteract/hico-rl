#!/usr/bin/env bash
set -euo pipefail

# Recorded real-robot data fine-tuning template. This trains from an existing
# dataset; it does not connect to or command a robot.

: "${DATASET_ROOT:?Set DATASET_ROOT to a recorded real-robot dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the new checkpoint}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhang/xzw/two/Evo-RL-main}"
DATASET_REPO_ID="${DATASET_REPO_ID:-real/so101_sponge_wipe_place}"
VLM_MODEL="${VLM_MODEL:-/home/zhang/xzw/two/evo-rl-jax/SmolVLM2-500M-Video-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-60000}"

if [[ ! -d "$DATASET_ROOT" || ! -d "$VLM_MODEL" ]]; then
  echo "DATASET_ROOT or VLM_MODEL does not exist" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/src/lerobot/scripts/lerobot_train.py" ]]; then
  echo "Evo-RL project is missing lerobot_train.py: $PROJECT_ROOT" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
exec python src/lerobot/scripts/lerobot_train.py \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.streaming=false \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.vlm_model_name="$VLM_MODEL" \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.use_mamba_temporal=false \
  --policy.tokenizer_max_length=64 \
  --policy.gripper_loss_weight=2.0 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=55000 \
  --policy.scheduler_decay_lr=1e-6 \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_checkpoint=true \
  --output_dir="$OUTPUT_DIR" \
  --resume=false
