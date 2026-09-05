#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${EVO_RL_ENV_NAME:-hico-rl}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${EVO_RL_PYTHON_VERSION:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-false}"
export CONDA_NOTICES="${CONDA_NOTICES:-false}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH" >&2
  exit 1
fi

echo "[1/5] Checking host CUDA/driver (informational)"
if [[ "${EVO_RL_SKIP_CUDA_CHECK:-0}" == "1" ]]; then
  echo "Skipping nvidia-smi check because EVO_RL_SKIP_CUDA_CHECK=1"
else
  nvidia-smi || {
    echo "nvidia-smi failed. Fix/verify the host driver before CUDA model tests." >&2
    exit 2
  }
fi

echo "[2/5] Creating or reusing conda environment: ${ENV_NAME}"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -y --offline -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip || {
    echo "Offline Conda package cache did not contain Python ${PYTHON_VERSION}; retrying online." >&2
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
  }
fi

CONDA_RUN=(conda run --no-capture-output -n "${ENV_NAME}")

echo "[3/5] Installing PyTorch CUDA 12.8 wheels"
"${CONDA_RUN[@]}" python -m pip install --upgrade pip setuptools wheel
"${CONDA_RUN[@]}" python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url "${PYTORCH_INDEX_URL}"

echo "[4/5] Installing Evo-RL and SmolVLA dependencies"
"${CONDA_RUN[@]}" python -m pip install -e "${PROJECT_ROOT}[smolvla]"
"${CONDA_RUN[@]}" python -m pip install \
  "peft>=0.18.0,<1.0.0" \
  "timm>=1.0.0,<1.1.0" \
  "ninja>=1.11.1,<2.0.0"

echo "[5/5] Installing Mamba-SSM CUDA extension"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
DEFAULT_MAMBA_WHEEL="${HOME}/.cache/hico-rl-wheels/mamba_ssm-2.2.4-cp310-cp310-linux_x86_64.whl"
MAMBA_WHEEL="${MAMBA_WHEEL:-${DEFAULT_MAMBA_WHEEL}}"
MAMBA_SOURCE_DIR="${MAMBA_SOURCE_DIR:-}"
if [[ -f "${MAMBA_WHEEL}" ]]; then
  echo "Installing the locally built sm_120 wheel: ${MAMBA_WHEEL}"
  "${CONDA_RUN[@]}" python -m pip install --no-deps --force-reinstall "${MAMBA_WHEEL}"
elif [[ -n "${MAMBA_SOURCE_DIR}" && -d "${MAMBA_SOURCE_DIR}/csrc/selective_scan" ]]; then
  echo "Building Mamba-SSM from complete source: ${MAMBA_SOURCE_DIR}"
  if ! "${CONDA_RUN[@]}" env MAX_JOBS="${MAX_JOBS}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps --force-reinstall "${MAMBA_SOURCE_DIR}"; then
    echo "WARNING: source Mamba-SSM build failed. Mamba mode is not ready." >&2
  fi
else
  echo "WARNING: RTX 50-series requires a source build with sm_120." >&2
  echo "Set MAMBA_WHEEL to a verified sm_120 wheel, or MAMBA_SOURCE_DIR to a patched complete v2.2.4 checkout." >&2
fi

echo "Environment setup finished. Run:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
echo "  python scripts/check_smolvla_compat.py"
