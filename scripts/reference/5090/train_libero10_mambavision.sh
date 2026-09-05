

export MUJOCO_GL=egl
export LIBERO_DATASET_PATH=$HOME/.libero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
source /home/wdy/Software/Anaconda/etc/profile.d/conda.sh
conda activate lerobot-mavi
DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_10_image"
#DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_goal_image_download"
#DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_spatial_image"
#DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/metaworld_mt50"
#DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_object_image"
DATASET="HuggingFaceVLA/libero"
VLM_MODEL="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/SmolVLM2-500M-Video-Instruct"
OUTPUT_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/mma-smolvla-master/src/lerobot/outputs/113_run105_baseline"
#CHECKPOINT_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/mma-smolvla-master/src/lerobot/outputs/110/checkpoints/040000"
#CONFIG_PATH="$CHECKPOINT_DIR/pretrained_model/train_config.json"
BATCH_SIZE=32
STEPS=60000
SAVE_FREQ=10000
LOG_FREQ=50
EVAL_FREQ=10000
NUM_WORKERS=8

# Expert配置
NUM_VLM_LAYERS=16
EXPERT_WIDTH=0.75  # Run 105配置

TRAIN_EXPERT_ONLY=false
TOKENIZER_MAX_LENGTH=48  # Run 105配置

# MPC最优配置（Nash均衡点）
CHUNK_SIZE=50
N_ACTION_STEPS=25

# Mamba时序模型
USE_MAMBA_TEMPORAL=true
MAMBA_HISTORY_LENGTH=50
MAMBA_D_STATE=64
MAMBA_D_CONV=4
MAMBA_EXPAND=2
MAMBA_TIME_EMBED_DIM=64
GRIPPER_LOSS_WEIGHT=1.0

# 检查环境
echo "1. 检查环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "GPU: CPU模式"

echo ""
echo "2. 训练配置:"
echo "  数据集: $DATASET"
echo "  数据集路径: $DATASET_ROOT"
echo "  VLM模型: $VLM_MODEL"
echo "  批次大小: $BATCH_SIZE"
echo "  总步数: $STEPS"
echo "  保存频率: 每 $SAVE_FREQ 步"
echo "  日志频率: 每 $LOG_FREQ 步"
echo "  评估频率: 每 $EVAL_FREQ 步"
echo "  Worker数: $NUM_WORKERS"
echo "  输出目录: $OUTPUT_DIR"
echo "  🔄 恢复训练: 从检查点 $CHECKPOINT_DIR"

echo "3. 检查输出目录..."
if [ -d "$OUTPUT_DIR" ]; then
    echo "  输出目录已存在: $OUTPUT_DIR"
else
    echo "  训练脚本将自动创建输出目录: $OUTPUT_DIR"
fi
echo ""




cd /media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/mma-smolvla-master
python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=$DATASET_ROOT \
    --dataset.repo_id=$DATASET \
    --dataset.streaming=false \
    --policy.type=smolvla \
    --policy.vlm_model_name=$VLM_MODEL \
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
    --policy.optimizer_lr=0.0001 \
    --policy.scheduler_warmup_steps=2_000 \
    --policy.scheduler_decay_steps=80_000 \
    --policy.scheduler_decay_lr=1e-5 \
    --save_checkpoint=true \
    --output_dir=$OUTPUT_DIR \
    --resume=false \
    --policy.train_expert_only=$TRAIN_EXPERT_ONLY \
    --policy.freeze_vision_encoder=true \
    --policy.train_state_proj=true \
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

# 训练完成提示
echo ""

echo "======================================================================"
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "训练成功完成！"
else
    echo "训练遇到问题 (状态码: $TRAIN_STATUS)"
fi
echo "======================================================================"
echo ""
echo "📊 输出信息:"
echo "  模型检查点: $OUTPUT_DIR"
echo "  日志目录: $OUTPUT_DIR"
echo ""
echo "📈 监控训练进度:"
echo "  tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "======================================================================"

exit $TRAIN_STATUS
