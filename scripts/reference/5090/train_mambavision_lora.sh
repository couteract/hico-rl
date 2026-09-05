#!/bin/bash

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
unset LEROBOT_HOME
export HF_LEROBOT_HOME="${HICO_ROOT}"
mkdir -p "$HF_DATASETS_CACHE"

source /home/wdy/Software/Anaconda/etc/profile.d/conda.sh
conda activate lerobot-mavi

# Dataset configuration
#RAW_DATASET_ROOT="${HICO_ROOT}/datasets/libero_object_image"
#RAW_DATASET_ROOT="${HICO_ROOT}/datasets/libero_10_image"
RAW_DATASET_ROOT="${HICO_ROOT}/datasets/metaworld_mt50"
DATASET_CROP_TAG="top50_left90_h430_w550"
USE_CROP="${USE_CROP:-0}"
DATASET_ROOT="$RAW_DATASET_ROOT"
DATASET="HuggingFaceVLA/libero"
VLM_MODEL="${HICO_ROOT}/datasets/SmolVLM2-500M-Video-Instruct"
OUTPUT_DIR="${HICO_ROOT}/outputs/metaworld_2000"
CROP_PARAMS_PATH="${PROJECT_ROOT}/configs/libero_object_top_crop.json"

# ========== LoRA Critical Configuration ==========
# 🔥 Optimized for libero_object: Vision-dense task with high gripper complexity
BATCH_SIZE=16                    # 🔥 Small batch for more epochs (454 episodes)
NUM_WORKERS=4                    # Reduce CPU overhead
WEIGHT_DECAY=1e-4                # 🔥 Strong regularization to prevent overfitting
SEED="${SEED:-2000}"
LORA_DROPOUT=0.15                # 🔥 Increased to 0.15 for small dataset regularization
LORA_RANK=32                     # 🔥 Doubled from 16 to handle complex wrist vision
LORA_ALPHA=64                    # 🔥 Maintain scaling=2.0 (alpha/rank)

# LR Schedule (🔥 KEY OPTIMIZATION: Extended for small dataset)
STEPS=50000                   # 🔥 Extended to 150k for ~18 epochs (vs 4.4 for libero_10)
SAVE_FREQ=10000                   # 🔥 Frequent checkpointing for monitoring
LOG_FREQ=50
EVAL_FREQ=5000                   # 🔥 Frequent evaluation to catch issues early
SCHEDULER_DECAY_STEPS=45000     # 🔥 Decay at 90% - extended learning window
WARMUP_STEPS=3000                # 🔥 Longer warmup from 2k to 3k
DECAY_LR=5e-6                    # 🔥 Gentler final LR (was 1e-5)

# Expert configuration (Run 105 baseline)
NUM_VLM_LAYERS=16
EXPERT_WIDTH=1.0                 # Maximize capacity
TRAIN_EXPERT_ONLY=false
TOKENIZER_MAX_LENGTH=96          # Run 105 config
USE_AMP=false                     # Use bf16 mixed precision on CUDA to reduce update time

# Action configuration
CHUNK_SIZE=10
N_ACTION_STEPS=10                # Run 105 config
GRIPPER_LOSS_WEIGHT=3         # 🔥 Increased to 6.0 for object manipulation (high gripper complexity)

# Mamba temporal model
USE_MAMBA_TEMPORAL=true
MAMBA_HISTORY_LENGTH=50
MAMBA_D_STATE=64
MAMBA_D_CONV=4
MAMBA_EXPAND=2
MAMBA_TIME_EMBED_DIM=64

if [ -n "${TEST_STEPS:-}" ]; then
    OUTPUT_DIR="${HICO_ROOT}/outputs/test_${TEST_STEPS}_mambavision_lora"
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
echo "🚀 Run 120: CONSERVATIVE Plan A for libero_object_image"
echo "=========================================================================="
echo "  Dataset: libero_object_image (454 episodes, vision-dense)"
echo "  Output: $OUTPUT_DIR"
echo "  Crop mode: $USE_CROP (0=raw/sim, 1=cropped/real)"
echo "  Seed: $SEED"
echo "  🔥 LoRA rank: $LORA_RANK (doubled for complex vision)"
echo "  🔥 LoRA alpha: $LORA_ALPHA (scaling=2.0)"
echo "  🔥 LoRA dropout: $LORA_DROPOUT (strong regularization)"
echo "  🔥 Batch size: $BATCH_SIZE (small for more epochs)"
echo "  🔥 Total steps: $STEPS (extended to 150k for ~18 epochs)"
echo "  🔥 Gripper loss weight: $GRIPPER_LOSS_WEIGHT (6x increase for object manipulation)"
echo "  🔥 Weight decay: $WEIGHT_DECAY (prevent overfitting)"
echo "  🔥 LR Schedule: DECAY_STEPS=$SCHEDULER_DECAY_STEPS (90% decay point)"
echo "  🔥 Warmup steps: $WARMUP_STEPS (gentler warmup)"
echo "  🔥 Final LR: $DECAY_LR (gentler decay)"
echo "  ✅ load_vlm_weights=true (Loading pretrained SmolVLM)"
echo "  ✅ Expert Width: $EXPERT_WIDTH"
echo "  ✅ Eval frequency: every $EVAL_FREQ steps"
echo "=========================================================================="
echo "  Expected performance (454 eps, ~8373 iter/epoch):"
echo "    30k steps (~3.6 epochs): 10-20% (minimum 5%)"
echo "    60k steps (~7.2 epochs): 30-40% (minimum 20%)"
echo "    90k steps (~10.7 epochs): 45-55% (minimum 40%)"
echo "    120k steps (~14.3 epochs): 55-65% (minimum 55%)"
echo "    150k steps (~18 epochs): 60-75% (TARGET)"
echo "  Training time: ~51 hours (2.1 days)"
echo "=========================================================================="

cd "$PROJECT_ROOT"

if [ "$USE_CROP" = "1" ]; then
    DATASET_ROOT="${HICO_ROOT}/datasets/libero_object_image_${DATASET_CROP_TAG}_cropped_resized"
    if [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
        echo "Creating cropped top-camera dataset: ${DATASET_ROOT}"
        if ! python -m lerobot.rl.crop_dataset_roi \
            --repo-id "$DATASET" \
            --root "$RAW_DATASET_ROOT" \
            --crop-params-path "$CROP_PARAMS_PATH" \
            --new-repo-id "HuggingFaceVLA/libero_object_image_${DATASET_CROP_TAG}_cropped_resized" \
            --new-dataset-root "$DATASET_ROOT" \
            --task "libero object"; then
            echo "❌ Failed to create cropped dataset."
            exit 1
        fi
    fi
fi

python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=$DATASET_ROOT \
    --dataset.repo_id=$DATASET \
    --dataset.streaming=false \
    --policy.type=smolvla \
    --policy.vlm_model_name=$VLM_MODEL \
    --policy.load_vlm_weights=true \
    --policy.push_to_hub=false \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --save_freq=$SAVE_FREQ \
    --log_freq=$LOG_FREQ \
    --eval_freq=$EVAL_FREQ \
    --seed=$SEED \
    --policy.num_vlm_layers=$NUM_VLM_LAYERS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH \
    --policy.attention_mode=cross_attn \
    --policy.optimizer_grad_clip_norm=10.0 \
    --policy.optimizer_lr=2e-4 \
    --policy.optimizer_weight_decay=$WEIGHT_DECAY \
    --policy.scheduler_warmup_steps=$WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.scheduler_decay_lr=$DECAY_LR \
    --save_checkpoint=true \
    --output_dir=$OUTPUT_DIR \
    --resume=false \
    --policy.train_expert_only=$TRAIN_EXPERT_ONLY \
    --policy.use_amp=$USE_AMP \
    --policy.freeze_vision_encoder=false \
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
    --policy.gripper_loss_weight=$GRIPPER_LOSS_WEIGHT

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
