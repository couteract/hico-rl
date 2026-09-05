# HiCo-RL

This repository stores the reproducible launch assets for the HiCoVLA and
Evo-RL work. The model implementation remains in the working Evo-RL checkout
at `/home/zhang/xzw/two/Evo-RL-main`; this repository does not vendor a second
copy of that code.

## Imported 5090 reference

`scripts/reference/5090/` contains the active training and evaluation scripts
copied from `/media/zhang/KINGSTON/hico/5090数据拷贝/src`. The known failed
SmolVLA snapshots were deliberately excluded. These files preserve the old
machine paths for provenance and should not be launched unchanged.

The paired checkpoint is `/media/zhang/KINGSTON/040000/pretrained_model`.
Its saved LIBERO-10 evaluation is 43/50 successes (86%, five episodes per
task). The checkpoint records two cameras, an 8D state, a 7D action,
`chunk_size=50`, `n_action_steps=50`, Mamba temporal history 50, and ten flow
inference steps. The old evaluation wrapper changes rollout execution to five
steps and enables ACG.

## Current-machine templates

- `scripts/local/eval_libero10_040000.sh` evaluates the 040000 checkpoint
  with the local SmolVLM2 artifact and current Evo-RL checkout. It first builds
  a lightweight checkpoint overlay under `hico-rl-runs`; the 1.1 GB weights and
  normalization tensors remain symlinked to the read-only KINGSTON source.
- `scripts/local/prepare_checkpoint_overlay.sh` rewrites only the saved VLM and
  tokenizer paths in that overlay. It never edits the source checkpoint.
- `scripts/local/train_libero10_mambavision.sh` reproduces the recorded
  040000 training configuration with current paths.
- `scripts/local/train_real_template.sh` trains from recorded real-robot data;
  it never connects to a robot and requires explicit dataset/output paths.

All scripts are path-overridable through environment variables. No script
installs packages or activates a conda environment.

Set `DRY_RUN=1` on the LIBERO evaluator to prepare and print the full command
without loading the model or starting MuJoCo. The scripts use the active
environment's `python` by default; set `PYTHON_BIN` to an explicit interpreter
when validating from another shell. For staged rollout validation, use for
example `TASK_IDS='[0]' EPISODES_PER_TASK=1` before running all ten tasks.

Training uses the active environment's `python` by default. For the existing
environment, use `PYTHON_BIN=/home/zhang/anaconda3/envs/hico-rl/bin/python`.
The training template fixes the recorded checkpoint seed at `42` by default
and does not activate or install an environment.
