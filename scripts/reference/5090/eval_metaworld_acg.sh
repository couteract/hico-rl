#!/bin/bash
set -e

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
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

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HICO_ROOT}/outputs/metaworld_2000/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-${N_EPISODES:-10}}"
PARALLEL_ENVS="${PARALLEL_ENVS:-${BATCH_SIZE:-1}}"
SEED="${SEED:-2000}"
ACG_SCALE="${ACG_SCALE:-2.5}"
ACG_TARGET_LAYERS="${ACG_TARGET_LAYERS:-[9,10,11,12]}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
USE_AMP="${USE_AMP:-false}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
ACTION_STEPS="${ACTION_STEPS:-${N_ACTION_STEPS:-10}}"
INFERENCE_STEPS="${INFERENCE_STEPS:-${POLICY_NUM_STEPS:-10}}"
METAWORLD_TASK_GROUPS="${METAWORLD_TASK_GROUPS:-easy medium hard very_hard}"
RENAME_MAP="${RENAME_MAP:-{}}"

usage() {
    cat <<EOF
Usage:
  bash src/eval_metaworld_acg.sh [options]

Options:
  --checkpoint-dir PATH       Policy checkpoint directory.
  --output-dir PATH           Root evaluation output directory.
  --episodes-per-task N       Episodes to run for each MetaWorld task.
  --parallel-envs N           Eval batch size; parallel env count, not episode count.
  --task-groups "A B C"       Space-separated difficulty groups/tasks to run in order.
                              Defaults to "easy medium hard very_hard".
  --chunk-size N              Action chunk horizon predicted by the policy.
  --action-steps N            Actions executed before replanning; must be <= chunk-size.
  --inference-steps N         Flow/denoising inference steps inside the policy.
  --acg-scale X               ACG guidance scale.
  --acg-target-layers VAL     ACG target layers, e.g. '[9,10,11,12]'.
  --max-parallel-tasks N      Number of task envs evaluated concurrently inside one group.
  --seed N                    Start seed.
  --help                      Show this message.

Examples:
  bash src/eval_metaworld_acg.sh --episodes-per-task 5 --parallel-envs 1
  bash src/eval_metaworld_acg.sh --task-groups "easy medium hard very_hard" --acg-scale 3.0
  CHECKPOINT_DIR=/path/to/pretrained_model bash src/eval_metaworld_acg.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --episodes-per-task)
            EPISODES_PER_TASK="$2"
            shift 2
            ;;
        --parallel-envs)
            PARALLEL_ENVS="$2"
            shift 2
            ;;
        --task-groups)
            METAWORLD_TASK_GROUPS="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --action-steps)
            ACTION_STEPS="$2"
            shift 2
            ;;
        --inference-steps)
            INFERENCE_STEPS="$2"
            shift 2
            ;;
        --acg-scale)
            ACG_SCALE="$2"
            shift 2
            ;;
        --acg-target-layers)
            ACG_TARGET_LAYERS="$2"
            shift 2
            ;;
        --max-parallel-tasks)
            MAX_PARALLEL_TASKS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT_DIR" ]; then
    for CANDIDATE in "${CHECKPOINT_ROOT}"/[0-9][0-9][0-9][0-9][0-9][0-9]/pretrained_model; do
        if [ -f "${CANDIDATE}/config.json" ] && [ -f "${CANDIDATE}/model.safetensors" ]; then
            CHECKPOINT_DIR="$CANDIDATE"
        fi
    done
fi

if (( ACTION_STEPS > CHUNK_SIZE )); then
    echo "ERROR: --action-steps ($ACTION_STEPS) must be <= --chunk-size ($CHUNK_SIZE)."
    exit 1
fi

if (( PARALLEL_ENVS > EPISODES_PER_TASK )); then
    echo "ERROR: --parallel-envs ($PARALLEL_ENVS) must be <= --episodes-per-task ($EPISODES_PER_TASK)."
    exit 1
fi

if [ ! -f "${CHECKPOINT_DIR}/config.json" ] || [ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]; then
    echo "ERROR: checkpoint directory must contain config.json and model.safetensors:"
    echo "  ${CHECKPOINT_DIR}"
    exit 1
fi

CHECKPOINT_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"
OUTPUT_DIR="${OUTPUT_DIR:-${HICO_ROOT}/outputs/eval_metaworld_42_${CHECKPOINT_NAME}_acg}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================================================="
echo "Evaluating MetaWorld with ACG"
echo "=========================================================================="
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Output root: $OUTPUT_DIR"
echo "  Task groups: $METAWORLD_TASK_GROUPS"
echo "  Episodes per task: $EPISODES_PER_TASK"
echo "  Parallel envs: $PARALLEL_ENVS"
echo "  Max parallel tasks: $MAX_PARALLEL_TASKS"
echo "  ACG scale: $ACG_SCALE"
echo "  ACG target layers: $ACG_TARGET_LAYERS"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Action steps before replanning: $ACTION_STEPS"
echo "  Inference steps: $INFERENCE_STEPS"
echo "  EGL device id: $MUJOCO_EGL_DEVICE_ID"
echo "=========================================================================="

cd "$PROJECT_ROOT"

for TASK_GROUP in $METAWORLD_TASK_GROUPS; do
    GROUP_OUTPUT_DIR="${OUTPUT_DIR}/${TASK_GROUP}"
    mkdir -p "$GROUP_OUTPUT_DIR"

    echo ""
    echo "=========================================================================="
    echo "Evaluating MetaWorld task group: ${TASK_GROUP}"
    echo "  Group output: ${GROUP_OUTPUT_DIR}"
    echo "=========================================================================="

    python src/lerobot/scripts/lerobot_eval.py \
        --policy.path="$CHECKPOINT_DIR" \
        --env.type=metaworld \
        --env.task="$TASK_GROUP" \
        --env.max_parallel_tasks="$MAX_PARALLEL_TASKS" \
        --eval.n_episodes="$EPISODES_PER_TASK" \
        --eval.batch_size="$PARALLEL_ENVS" \
        --seed="$SEED" \
        --output_dir="$GROUP_OUTPUT_DIR" \
        --policy.device=cuda \
        --policy.use_amp="$USE_AMP" \
        --policy.use_acg=true \
        --policy.acg_guidance_scale="$ACG_SCALE" \
        --policy.acg_target_layers="$ACG_TARGET_LAYERS" \
        --policy.log_acg_metrics=true \
        --policy.enable_mamba_warmup=true \
        --policy.chunk_size="$CHUNK_SIZE" \
        --policy.n_action_steps="$ACTION_STEPS" \
        --policy.num_steps="$INFERENCE_STEPS" \
        --rename_map="$RENAME_MAP"
done

python - "$OUTPUT_DIR" $METAWORLD_TASK_GROUPS <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
task_groups = sys.argv[2:]
summary = {}

for group in task_groups:
    info_path = output_dir / group / "eval_info.json"
    if not info_path.exists():
        summary[group] = {"error": f"Missing {info_path}"}
        continue

    with info_path.open() as f:
        info = json.load(f)

    summary[group] = {
        "overall": info.get("overall", {}),
        "per_group": info.get("per_group", {}).get(group, {}),
    }

with (output_dir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print(f"Wrote summary: {output_dir / 'summary.json'}")
PY

echo ""
echo "=========================================================================="
echo "MetaWorld evaluation complete."
echo "Results saved under: ${OUTPUT_DIR}"
echo "=========================================================================="
