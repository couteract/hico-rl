# HiCo-RL Code Snapshot

This repository contains a code snapshot imported from:

`/home/zhang/xzw/two/Evo-RL-main`

The source checkout was used as the implementation workspace for the current
Evo-RL and SmolVLA experiments. The snapshot includes the source tree,
configuration, tools, examples, documentation, and the local launch scripts
under `scripts/local/`.

Large or machine-local artifacts are intentionally excluded:

- training outputs and rollout data
- checkpoints and model weights
- datasets and Parquet files
- videos and cached Python files
- local Claude/Codex permission settings
- Piper mesh assets whose source checkout only contained missing Git LFS pointers

Expected runtime paths remain configurable through the shell scripts. The
original Evo-RL workspace and its experiment outputs are not modified by this
snapshot operation.
