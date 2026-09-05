#!/usr/bin/env bash
set -euo pipefail

# Current-machine LIBERO-10 evaluation for the 5090-trained checkpoint.
# Override paths with environment variables; no conda activation is performed.

PROJECT_ROOT="${PROJECT_ROOT:-/home/zhang/xzw/two/Evo-RL-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-/media/zhang/KINGSTON/040000/pretrained_model}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/zhang/xzw/hico-rl-runs/checkpoint_overlays/040000_local}"
VLM_MODEL="${VLM_MODEL:-/home/zhang/xzw/two/evo-rl-jax/SmolVLM2-500M-Video-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhang/xzw/hico-rl-runs/eval_libero10_040000}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/zhang/xzw/hico-rl-runs/libero_config}"
LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-/home/zhang/xzw/datasets}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
PARALLEL_ENVS="${PARALLEL_ENVS:-1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
TASK_IDS="${TASK_IDS:-}"
SEED="${SEED:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
ACTION_STEPS="${ACTION_STEPS:-5}"
INFERENCE_STEPS="${INFERENCE_STEPS:-10}"
USE_ACG="${USE_ACG:-true}"
ACG_SCALE="${ACG_SCALE:-2.5}"
ACG_TARGET_LAYERS="${ACG_TARGET_LAYERS:-[9,10]}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$VLM_MODEL" ]]; then
  echo "Local SmolVLM2 directory does not exist: $VLM_MODEL" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/src/lerobot/scripts/lerobot_eval.py" ]]; then
  echo "Evo-RL project is missing lerobot_eval.py: $PROJECT_ROOT" >&2
  exit 2
fi
if (( ACTION_STEPS > CHUNK_SIZE )); then
  echo "ACTION_STEPS must be <= CHUNK_SIZE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CHECKPOINT_DIR="$SOURCE_CHECKPOINT_DIR" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
VLM_MODEL="$VLM_MODEL" \
  "$SCRIPT_DIR/prepare_checkpoint_overlay.sh" >/dev/null

LIBERO_PKG_ROOT="$("$PYTHON_BIN" -c 'import pathlib, site
for base in [*site.getsitepackages(), site.getusersitepackages()]:
    candidate = pathlib.Path(base) / "libero" / "libero"
    if (candidate / "__init__.py").exists():
        print(candidate)
        break
else:
    raise SystemExit("Could not locate installed libero/libero package")')"
mkdir -p "$LIBERO_CONFIG_PATH"
{
  printf 'assets: %s/assets\n' "$LIBERO_PKG_ROOT"
  printf 'bddl_files: %s/bddl_files\n' "$LIBERO_PKG_ROOT"
  printf 'benchmark_root: %s\n' "$LIBERO_PKG_ROOT"
  printf 'datasets: %s\n' "$LIBERO_DATASET_PATH"
  printf 'init_states: %s/init_files\n' "$LIBERO_PKG_ROOT"
} >"$LIBERO_CONFIG_PATH/config.yaml"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LIBERO_CONFIG_PATH
export LIBERO_DATASET_PATH
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
cmd=("$PYTHON_BIN" src/lerobot/scripts/lerobot_eval.py \
  --policy.path="$CHECKPOINT_DIR" \
  --policy.vlm_model_name="$VLM_MODEL" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.use_acg="$USE_ACG" \
  --policy.acg_guidance_scale="$ACG_SCALE" \
  --policy.acg_target_layers="$ACG_TARGET_LAYERS" \
  --policy.log_acg_metrics=true \
  --policy.enable_mamba_warmup=true \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --policy.num_steps="$INFERENCE_STEPS" \
  --env.type=libero \
  --env.task=libero_10 \
  --env.max_parallel_tasks="$MAX_PARALLEL_TASKS" \
  --eval.n_episodes="$EPISODES_PER_TASK" \
  --eval.batch_size="$PARALLEL_ENVS" \
  --seed="$SEED" \
  --output_dir="$OUTPUT_DIR" \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}')

if [[ -n "$TASK_IDS" ]]; then
  cmd+=(--env.task_ids="$TASK_IDS")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'LIBERO_CONFIG_PATH=%q LIBERO_DATASET_PATH=%q PYTHONPATH=%q ' \
    "$LIBERO_CONFIG_PATH" "$LIBERO_DATASET_PATH" "$PYTHONPATH"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
