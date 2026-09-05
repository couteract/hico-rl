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
export TORCH_HOME="${HICO_ROOT}/.cache/torch"
export WANDB_DIR="${HICO_ROOT}/.cache/wandb"
export WANDB_CACHE_DIR="${HICO_ROOT}/.cache/wandb"
export LIBERO_CONFIG_PATH="${HICO_ROOT}/.cache/libero"
unset LEROBOT_HOME
export HF_LEROBOT_HOME="${HICO_ROOT}"

source /home/wdy/Software/Anaconda/etc/profile.d/conda.sh
conda activate lerobot-mavi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${HICO_ROOT}/outputs/120_10_conservative42/checkpoints/040000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${HICO_ROOT}/outputs/eval_120_object_040000_acg1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-${N_EPISODES:-5}}"
PARALLEL_ENVS="${PARALLEL_ENVS:-${BATCH_SIZE:-1}}"
SEED="${SEED:-1000}"
ACG_SCALE="${ACG_SCALE:-2.5}"
ACG_TARGET_LAYERS="${ACG_TARGET_LAYERS:-[9,10]}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
USE_AMP="${USE_AMP:-false}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
ACTION_STEPS="${ACTION_STEPS:-${N_ACTION_STEPS:-5}}"
INFERENCE_STEPS="${INFERENCE_STEPS:-${POLICY_NUM_STEPS:-10}}"

usage() {
    cat <<EOF
Usage:
  bash src/eval_libero_object_acg.sh [options]

Options:
  --checkpoint-dir PATH     Policy checkpoint directory.
  --output-dir PATH         Evaluation output directory.
  --episodes-per-task N     Episodes to run for each LIBERO object task.
  --parallel-envs N         Eval batch size; parallel env count, not episode count.
  --chunk-size N            Action chunk horizon predicted by the policy.
  --action-steps N          Actions executed before replanning; must be <= chunk-size.
  --inference-steps N       Flow/denoising inference steps inside the policy.
  --acg-scale X             ACG guidance scale.
  --acg-target-layers VAL   ACG target layers, e.g. '[9,10,11,12]'.
  --seed N                  Start seed.
  --help                    Show this message.

Examples:
  bash src/eval_libero_object_acg.sh --episodes-per-task 5 --parallel-envs 1
  bash src/eval_libero_object_acg.sh --episodes-per-task 5 --action-steps 10 --inference-steps 10 --acg-scale 3.0
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

if (( ACTION_STEPS > CHUNK_SIZE )); then
    echo "ERROR: --action-steps ($ACTION_STEPS) must be <= --chunk-size ($CHUNK_SIZE)."
    exit 1
fi

if (( PARALLEL_ENVS > EPISODES_PER_TASK )); then
    echo "ERROR: --parallel-envs ($PARALLEL_ENVS) must be <= --episodes-per-task ($EPISODES_PER_TASK)."
    exit 1
fi

LIBERO_PKG_ROOT="$(python - <<'PY'
import pathlib
import site

for base in [*site.getsitepackages(), site.getusersitepackages()]:
    candidate = pathlib.Path(base) / "libero" / "libero"
    if (candidate / "__init__.py").exists():
        print(candidate)
        break
else:
    raise SystemExit("Could not locate installed libero/libero package")
PY
)"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
assets: ${LIBERO_PKG_ROOT}/assets
bddl_files: ${LIBERO_PKG_ROOT}/bddl_files
benchmark_root: ${LIBERO_PKG_ROOT}
datasets: ${HICO_ROOT}/datasets
init_states: ${LIBERO_PKG_ROOT}/init_files
EOF

echo "=========================================================================="
echo "Evaluating LIBERO Object with ACG"
echo "=========================================================================="
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Episodes per task: $EPISODES_PER_TASK"
echo "  Parallel envs: $PARALLEL_ENVS"
echo "  ACG scale: $ACG_SCALE"
echo "  ACG target layers: $ACG_TARGET_LAYERS"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Action steps before replanning: $ACTION_STEPS"
echo "  Inference steps: $INFERENCE_STEPS"
echo "  EGL device id: $MUJOCO_EGL_DEVICE_ID"
echo "=========================================================================="

cd "$PROJECT_ROOT"
python src/lerobot/scripts/lerobot_eval.py \
    --policy.path="$CHECKPOINT_DIR" \
    --env.type=libero \
    --env.task=libero_10 \
    --env.max_parallel_tasks="$MAX_PARALLEL_TASKS" \
    --eval.n_episodes="$EPISODES_PER_TASK" \
    --eval.batch_size="$PARALLEL_ENVS" \
    --seed="$SEED" \
    --output_dir="$OUTPUT_DIR" \
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
    --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'
