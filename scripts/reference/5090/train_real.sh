#!/bin/bash
set -e

export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

HICO_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico"
PROJECT_ROOT="${HICO_ROOT}/mma-smolvla-master"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export LIBERO_DATASET_PATH="${HICO_ROOT}/datasets"
export HF_HOME="${HICO_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.cache/huggingface/datasets"
export TORCH_HOME="${HICO_ROOT}/.cache/torch"
export WANDB_DIR="${HICO_ROOT}/.cache/wandb"
export WANDB_CACHE_DIR="${HICO_ROOT}/.cache/wandb"
export WANDB_MODE="${WANDB_MODE:-disabled}"
unset LEROBOT_HOME
export HF_LEROBOT_HOME="${HICO_ROOT}"
mkdir -p "$HF_DATASETS_CACHE"

source /home/wdy/Software/Anaconda/etc/profile.d/conda.sh
conda activate lerobot-mavi

python - <<'PY'
import sys
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"PyTorch CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    print("ERROR: CUDA is required for the Mamba/Triton real-robot training path.")
    print("Please check nvidia-smi and /dev/nvidia* before launching training.")
    sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# Dataset configuration
DATASET="real/so101_sponge_wipe_place"
DATASET_ROOT="${HICO_ROOT}/real/so101_sponge_wipe_place"
VLM_MODEL="${HICO_ROOT}/datasets/SmolVLM2-500M-Video-Instruct"
OUTPUT_DIR="${HICO_ROOT}/outputs/so101_sponge_wipe_place_2_nothing"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-hicovla-real}"

# ========== LoRA / Micro-finetune Configuration ==========
# Optimized for the 51-episode real-robot wipe dataset
BATCH_SIZE=16
NUM_WORKERS=4
WEIGHT_DECAY=1e-4
LORA_DROPOUT=0.10
LORA_RANK=16
LORA_ALPHA=32

# LR Schedule
STEPS=60000
SAVE_FREQ=5000
LOG_FREQ=50
EVAL_FREQ=2000
SCHEDULER_DECAY_STEPS=55000
WARMUP_STEPS=1000
DECAY_LR=1e-6

# Expert configuration
NUM_VLM_LAYERS=16
EXPERT_WIDTH=0.75
TRAIN_EXPERT_ONLY=false
TOKENIZER_MAX_LENGTH=64
USE_AMP=false

# Action configuration
CHUNK_SIZE=10
N_ACTION_STEPS=10
GRIPPER_LOSS_WEIGHT=2

# Mamba temporal model
USE_MAMBA_TEMPORAL=false
MAMBA_HISTORY_LENGTH=50
MAMBA_D_STATE=64
MAMBA_D_CONV=4
MAMBA_EXPAND=2
MAMBA_TIME_EMBED_DIM=64

if [ -n "${TEST_STEPS:-}" ]; then
    OUTPUT_DIR="${HICO_ROOT}/outputs/test_${TEST_STEPS}_mambavision_lora_$(date +%Y%m%d_%H%M%S)"
    STEPS=$TEST_STEPS
    SAVE_FREQ=$TEST_STEPS
    echo "⚠️  TEST_STEPS=$TEST_STEPS: running a shortened full-config test"
fi

if [ "${SMOKE_TEST:-0}" = "1" ]; then
    OUTPUT_DIR="${HICO_ROOT}/outputs/smoke_mambavision_lora"
    BATCH_SIZE=1
    NUM_WORKERS=0
    STEPS=1
    SAVE_FREQ=1
    LOG_FREQ=1
    EVAL_FREQ=0
    WARMUP_STEPS=0
    SCHEDULER_DECAY_STEPS=1
    echo "⚠️  SMOKE_TEST=1: running a one-step startup check"
fi

echo "=========================================================================="
echo "🚀 Real-robot fine-tune for so101_sponge_wipe_place"
echo "=========================================================================="
echo "  Dataset: wdy/so101_sponge_wipe_place (51 episodes)"
echo "  Output: $OUTPUT_DIR"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  LoRA dropout: $LORA_DROPOUT"
echo "  Batch size: $BATCH_SIZE"
echo "  Total steps: $STEPS"
echo "  Gripper loss weight: $GRIPPER_LOSS_WEIGHT"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  LR Schedule: DECAY_STEPS=$SCHEDULER_DECAY_STEPS"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Final LR: $DECAY_LR"
echo "  ✅ load_vlm_weights=true (Loading pretrained SmolVLM)"
echo "  ✅ Expert Width: $EXPERT_WIDTH"
echo "  ✅ Eval frequency: every $EVAL_FREQ steps"
echo "  ✅ WandB: enable=$WANDB_ENABLE project=$WANDB_PROJECT mode=$WANDB_MODE"
echo "=========================================================================="
echo "  Expected behavior: small-data real-robot adaptation with early overfit watch"
echo "  Training time: depends on GPU, checkpoint every 2k steps"
echo "=========================================================================="

cd "$PROJECT_ROOT"

python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=$DATASET_ROOT \
    --dataset.repo_id=$DATASET \
    --dataset.streaming=false \
    --policy.type=smolvla \
    --policy.device=$POLICY_DEVICE \
    --policy.vlm_model_name=$VLM_MODEL \
    --policy.load_vlm_weights=true \
    --policy.push_to_hub=false \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --log_freq=$LOG_FREQ \
    --eval_freq=$EVAL_FREQ \
    --policy.num_vlm_layers=$NUM_VLM_LAYERS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH \
    --policy.attention_mode=cross_attn \
    --policy.optimizer_grad_clip_norm=10.0 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.scheduler_warmup_steps=$WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_decay_lr=$DECAY_LR \
    --save_checkpoint=true \
    --output_dir=$OUTPUT_DIR \
    --resume=false \
    --policy.train_expert_only=$TRAIN_EXPERT_ONLY \
    --policy.use_amp=$USE_AMP \
    --policy.freeze_vision_encoder=true \
    --policy.train_state_proj=true \
    --policy.lora_rank=$LORA_RANK \
    --policy.lora_alpha=$LORA_ALPHA \
    --policy.lora_dropout=$LORA_DROPOUT \
    --policy.tokenizer_max_length=$TOKENIZER_MAX_LENGTH \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.use_mamba_temporal=$USE_MAMBA_TEMPORAL \
    --policy.mamba_history_length=$MAMBA_HISTORY_LENGTH \
    --policy.mamba_d_state=$MAMBA_D_STATE \
    --policy.mamba_d_conv=$MAMBA_D_CONV \
    --policy.mamba_expand=$MAMBA_EXPAND \
    --policy.mamba_time_embed_dim=$MAMBA_TIME_EMBED_DIM \
    --policy.gripper_loss_weight=$GRIPPER_LOSS_WEIGHT \
    --wandb.enable=$WANDB_ENABLE \
    --wandb.project=$WANDB_PROJECT \
    --wandb.mode=$WANDB_MODE

TRAIN_STATUS=$?

echo ""
echo "======================================================================"
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training encountered issues (status: $TRAIN_STATUS)"
fi
echo "======================================================================"

exit $TRAIN_STATUS
