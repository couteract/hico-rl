#!/usr/bin/env bash
set -euo pipefail

# Build a lightweight, writable view of a checkpoint whose saved model paths
# belonged to another machine. Large tensors remain read-only symlinks.

SOURCE_CHECKPOINT_DIR="${SOURCE_CHECKPOINT_DIR:-/media/zhang/KINGSTON/040000/pretrained_model}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/zhang/xzw/hico-rl-runs/checkpoint_overlays/040000_local}"
VLM_MODEL="${VLM_MODEL:-/home/zhang/xzw/two/evo-rl-jax/SmolVLM2-500M-Video-Instruct}"

required_files=(
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_preprocessor_step_5_normalizer_processor.safetensors
  policy_postprocessor.json
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
)

for name in "${required_files[@]}"; do
  if [[ ! -f "$SOURCE_CHECKPOINT_DIR/$name" ]]; then
    echo "Checkpoint file is missing: $SOURCE_CHECKPOINT_DIR/$name" >&2
    exit 2
  fi
done
if [[ ! -d "$VLM_MODEL" ]]; then
  echo "Local SmolVLM2 directory does not exist: $VLM_MODEL" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to patch checkpoint JSON without changing the source" >&2
  exit 2
fi

mkdir -p "$CHECKPOINT_DIR"

config_tmp="$(mktemp "$CHECKPOINT_DIR/.config.json.XXXXXX")"
preprocessor_tmp="$(mktemp "$CHECKPOINT_DIR/.policy_preprocessor.json.XXXXXX")"
trap 'rm -f "$config_tmp" "$preprocessor_tmp"' EXIT

jq --arg vlm_model "$VLM_MODEL" \
  '.vlm_model_name = $vlm_model' \
  "$SOURCE_CHECKPOINT_DIR/config.json" >"$config_tmp"

jq --arg vlm_model "$VLM_MODEL" \
  '(.steps[] | select(.registry_name == "tokenizer_processor") | .config.tokenizer_name) = $vlm_model' \
  "$SOURCE_CHECKPOINT_DIR/policy_preprocessor.json" >"$preprocessor_tmp"

mv "$config_tmp" "$CHECKPOINT_DIR/config.json"
mv "$preprocessor_tmp" "$CHECKPOINT_DIR/policy_preprocessor.json"

for name in \
  model.safetensors \
  policy_preprocessor_step_5_normalizer_processor.safetensors \
  policy_postprocessor.json \
  policy_postprocessor_step_0_unnormalizer_processor.safetensors \
  train_config.json; do
  if [[ -f "$SOURCE_CHECKPOINT_DIR/$name" ]]; then
    ln -sfn "$SOURCE_CHECKPOINT_DIR/$name" "$CHECKPOINT_DIR/$name"
  fi
done

printf '%s\n' "$CHECKPOINT_DIR"
