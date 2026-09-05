#!/bin/bash
# ============================================================================
# LeRobot SmolVLA 训练脚本 - 使用本地数据和模型
# ============================================================================

set -e

# 配置
DATASET_PATH="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/libero_spatial_image"
VLM_MODEL="/media/wdy/2caac15b-5621-41ce-9a36-92cadc3ee7f2/home/wdy1/xzw/hico/datasets/SmolVLM2-500M-Video-Instruct"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./outputs/libero_smolvla_local_${TIMESTAMP}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET="HuggingFaceVLA/libero"

# 验证路径
echo "📋 验证路径..."
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 数据集不存在: $DATASET_PATH"
    exit 1
fi

if [ ! -d "$VLM_MODEL" ]; then
    echo "❌ VLM模型不存在: $VLM_MODEL"
    exit 1
fi

echo "✅ 数据集: $DATASET_PATH"
echo "✅ VLM模型: $VLM_MODEL"

# 显示配置信息
echo ""
echo "==========================================================================="
echo "🚀 开始SmolVLA训练 - 本地数据和模型"
echo "==========================================================================="
echo "数据集路径:   $DATASET_PATH"
echo "VLM模型:      $VLM_MODEL"
echo "输出目录:     $OUTPUT_DIR"
echo "==========================================================================="
echo ""


cd "$SCRIPT_DIR"


echo ""



python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --policy.vlm_model_name="$VLM_MODEL" \
    --dataset.root="$DATASET_PATH" \
    --dataset.repo_id="$DATASET"  \
    --batch_size=32 \
    --num_workers=4 \
    --steps=20000 \
    --output_dir="../${OUTPUT_DIR}" \
    --policy.device=cuda \
    --policy.push_to_hub=false

echo ""
echo "==========================================================================="
echo "✅ 训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "==========================================================================="
