#!/bin/bash

# 📝 日志配置
LOG_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/logging"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 输出重定向：同时显示在终端和保存到日志文件
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "======================================================================"
echo "🚀 SmolVLA 训练启动 (测试新功能)"
echo "======================================================================"
echo "📝 日志文件: $LOG_FILE"
echo "⏰ 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
echo ""

# 设置环境变量
export MUJOCO_GL=egl
export LIBERO_DATASET_PATH=$HOME/.libero
# 解决GPU内存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1

# WandB离线模式（所有数据保存在本地，不上传到云端）
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR/wandb"

# 激活conda环境
source /home/wdy/Software/Anaconda/etc/profile.d/conda.sh
conda activate lerobot-mavi

# 训练参数 - 测试新功能配置
DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_goal_image_download"
#DATASET_ROOT="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_spatial_image"
DATASET="HuggingFaceVLA/libero"
VLM_MODEL="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/SmolVLM2-500M-Video-Instruct"
OUTPUT_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/mma-smolvla-master/src/lerobot/outputs/test2"

# 📊 可视化保存配置
SAVE_IMAGE_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/picture/libero-gaol"  # 测试实验图像

# WandB配置（离线模式）
WANDB_MODE="offline"
WANDB_PROJECT="lerobot-libero-goal"
WANDB_NAME="smolvla_test_$(date +%Y%m%d_%H%M%S)"

BATCH_SIZE=32
STEPS=20000  # 快速测试
SAVE_FREQ=10000
LOG_FREQ=50
EVAL_FREQ=20000
NUM_WORKERS=8

NUM_VLM_LAYERS=16
EXPERT_WIDTH=0.75
GRADIENT_ACCUMULATION_STEPS=2

CHUNK_SIZE=50
N_ACTION_STEPS=50

# ✅ 测试所有新功能
USE_MAMBA_TEMPORAL=true  # 启用Mamba时序平滑
MAMBA_HISTORY_LENGTH=10  # 当前推荐配置
MAMBA_D_STATE=64
MAMBA_D_CONV=4
MAMBA_EXPAND=2

# Stage Completion 启用（测试）
USE_STAGE_COMPLETION=false
STAGE_LOSS_WEIGHT=0.05
STAGE_VELOCITY_SMOOTHING="none"
STAGE_SMOOTHING_WINDOW=3
STAGE_SPEED_THRESHOLD=0.3
STAGE_GRIPPER_THRESHOLD=0.7
STAGE_Z_LIFT_THRESHOLD=0.03

# Progress Head 启用（测试）- 注意：当前版本可能不支持
USE_MAMBA_PROGRESS=false  # 暂时禁用直到代码支持
PROGRESS_LOSS_WEIGHT=0.01
PROGRESS_USE_SMOOTHNESS=true
PROGRESS_TARGET_TYPE="sigmoid"
PROGRESS_CURVE_STEEPNESS=6.0

GRIPPER_LOSS_WEIGHT=5.0

# ACG 禁用（训练时）
USE_ACG=false
ACG_GUIDANCE_SCALE=1.0

# 检查环境
echo "1. 检查环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "GPU: CPU模式"

echo ""
echo "2. 训练配置 (测试Mamba+Stage新功能):"
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
echo "  📊 图像保存目录: $SAVE_IMAGE_DIR"
echo "  📝 日志保存目录: $LOG_DIR"
echo "  📄 当前日志文件: $(basename $LOG_FILE)"
echo ""
echo "  Expert 配置:"
echo "    - VLM层数: $NUM_VLM_LAYERS"
echo "    - Expert宽度倍数: $EXPERT_WIDTH"
echo ""
echo "  Action预测配置:"
echo "    - Chunk Size: $CHUNK_SIZE 步"
echo "    - Action Steps: $N_ACTION_STEPS 步"
echo ""
echo "  ✅ 新功能测试配置:"
echo "    - Mamba时序平滑: $USE_MAMBA_TEMPORAL (history=$MAMBA_HISTORY_LENGTH)"
echo "    - Stage Completion: $USE_STAGE_COMPLETION (weight=$STAGE_LOSS_WEIGHT)"
echo "    - Progress Head: $USE_MAMBA_PROGRESS (暂时禁用)"
echo "    - ACG引导: $USE_ACG (scale=$ACG_GUIDANCE_SCALE)"
echo "    - Gripper权重: $GRIPPER_LOSS_WEIGHT"

echo ""
echo "3. 检查输出目录..."
if [ -d "$OUTPUT_DIR" ]; then
    echo "  ✅ 输出目录已存在: $OUTPUT_DIR"
else
    echo "  📁 训练脚本将自动创建输出目录: $OUTPUT_DIR"
fi

echo ""
echo "4. 检查图像保存目录..."
if [ -d "$SAVE_IMAGE_DIR" ]; then
    echo "  ✅ 图像保存目录已存在: $SAVE_IMAGE_DIR"
else
    echo "  📁 创建图像保存目录: $SAVE_IMAGE_DIR"
    mkdir -p "$SAVE_IMAGE_DIR"
    if [ $? -eq 0 ]; then
        echo "  ✅ 图像保存目录创建成功"
    else
        echo "  ❌ 图像保存目录创建失败，请检查权限"
    fi
fi
echo ""

echo ""
echo "5. 开始训练（复现ob2配置）..."
echo ""
echo "注: 基于ob2检查点配置"
echo "  - 参考检查点: /media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/mma-smolvla-master/src/lerobot/outputs/ob2/checkpoints/040000"
echo "  - Mamba启用 (history=20，与ob2一致)"
echo "  - Progress Head禁用 (ob2未使用)"
echo "  - 目标: 验证ob2训练配置的可复现性"
echo ""

# 运行训练
cd /media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/mma-smolvla-master
python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=$DATASET_ROOT \
    --dataset.repo_id=$DATASET \
    --dataset.streaming=false \
    --policy.type=smolvla \
    --policy.vlm_model_name=$VLM_MODEL \
    --policy.push_to_hub=false \
    --batch_size=$BATCH_SIZE \
      46 -SAVE_IMAGE_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/picture/libero-gaol-baseline"  # 基线实验图像
      46 +SAVE_IMAGE_DIR="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/picture/libero-gaol"  # 测试实验图像
      47
      48 +# WandB配置（离线模式）
      49 +WANDB_MODE="offline"
    --policy.num_vlm_layers=$NUM_VLM_LAYERS \
    --policy.expert_width_multiplier=$EXPERT_WIDTH \
    --policy.attention_mode=cross_attn \
    --policy.optimizer_grad_clip_norm=10.0 \
    --policy.optimizer_lr=0.0001 \
    --policy.scheduler_warmup_steps=1_000 \
    --policy.scheduler_decay_steps=50_000 \
    --policy.scheduler_decay_lr=1.0e-05 \
    --save_checkpoint=true \
    --output_dir=$OUTPUT_DIR \
    --resume=false \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.use_mamba_temporal=$USE_MAMBA_TEMPORAL \
    --policy.mamba_history_length=$MAMBA_HISTORY_LENGTH \
    --policy.mamba_d_state=$MAMBA_D_STATE \
    --policy.mamba_d_conv=$MAMBA_D_CONV \
    --policy.mamba_expand=$MAMBA_EXPAND \
    --policy.use_stage_completion=$USE_STAGE_COMPLETION \
    --policy.stage_loss_weight=$STAGE_LOSS_WEIGHT \
    --policy.stage_velocity_smoothing=$STAGE_VELOCITY_SMOOTHING \
    --policy.stage_smoothing_window=$STAGE_SMOOTHING_WINDOW \
    --policy.stage_speed_threshold_ratio=$STAGE_SPEED_THRESHOLD \
    --policy.stage_gripper_close_threshold=$STAGE_GRIPPER_THRESHOLD \
    --policy.stage_z_lift_threshold=$STAGE_Z_LIFT_THRESHOLD \
    --policy.gripper_loss_weight=$GRIPPER_LOSS_WEIGHT \
    --policy.use_acg=$USE_ACG \
    --policy.acg_guidance_scale=$ACG_GUIDANCE_SCALE

TRAIN_STATUS=$?

# 训练完成提示
echo ""

echo "======================================================================"
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "✅ 基线训练成功完成！"
    echo ""

    # 生成训练和评估图表
    echo "📊 生成训练和评估图表..."
    python src/generate_plots.py "$OUTPUT_DIR" "$SAVE_IMAGE_DIR"

    if [ $? -eq 0 ]; then
        echo "  ✅ Loss图和成功率图已生成"
    else
        echo "  ⚠️  图表生成失败，请检查WandB数据"
    fi

    echo ""
    echo "📁 整理输出文件..."

    # 复制WandB离线数据到图像目录
    if [ -d "$OUTPUT_DIR/wandb" ]; then
        echo "  📊 复制WandB离线数据..."
        cp -r "$OUTPUT_DIR/wandb" "$SAVE_IMAGE_DIR/" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✅ WandB数据已保存到: $SAVE_IMAGE_DIR/wandb/"
        fi
    fi

    # 创建符号链接到评估视频
    if [ -d "$OUTPUT_DIR/eval" ]; then
        echo "  🔗 创建评估视频链接..."
        if [ -L "$SAVE_IMAGE_DIR/eval" ]; then
            rm "$SAVE_IMAGE_DIR/eval"
        fi
        ln -s "$OUTPUT_DIR/eval" "$SAVE_IMAGE_DIR/eval" 2>/dev/null
        echo "  ✅ 评估视频链接: $SAVE_IMAGE_DIR/eval -> $OUTPUT_DIR/eval"
    fi

    # 创建输出目录的快捷链接
    if [ -L "$SAVE_IMAGE_DIR/latest_output" ]; then
        rm "$SAVE_IMAGE_DIR/latest_output"
    fi
    ln -s "$OUTPUT_DIR" "$SAVE_IMAGE_DIR/latest_output" 2>/dev/null

    echo "  ✅ 所有数据整理完成"
    echo ""
else
    echo "❌ 训练遇到问题 (状态码: $TRAIN_STATUS)"
fi
echo "======================================================================"
echo ""
echo "📊 输出信息:"
echo "  模型检查点: $OUTPUT_DIR/checkpoints/"
echo "  日志目录: $OUTPUT_DIR"
echo "  WandB离线数据: $OUTPUT_DIR/wandb/"
echo ""
echo "📸 可视化输出 (保存在: $SAVE_IMAGE_DIR)"
echo "  ├─ 📈 Loss曲线图: loss_curves.png"
echo "  ├─ 📊 成功率图: success_rate.png"
echo "  ├─ 📉 学习率图: learning_rate.png"
echo "  ├─ 🎯 训练概览: training_overview.png"
echo "  ├─ 📁 WandB数据: wandb/"
echo "  └─ 🔗 评估视频链接: eval/ -> $OUTPUT_DIR/eval/"
echo ""
echo "📈 监控训练进度:"
echo "  tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "🖼️  查看生成的图表:"
echo "  ls $SAVE_IMAGE_DIR/*.png"
echo "  firefox $SAVE_IMAGE_DIR/training_overview.png  # 打开概览图"
echo ""
echo "🎥 查看评估视频:"
echo "  ls $OUTPUT_DIR/eval/videos_step_*/"
echo ""
echo "======================================================================"
echo ""
echo "⏰ 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📝 完整日志已保存到: $LOG_FILE"
echo ""
echo "======================================================================"

exit $TRAIN_STATUS
