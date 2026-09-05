# Phase264 LIBERO rollouts in the PyTorch Evo-RL pipeline

## Decision

The 250 frame-ready Phase264 episodes are valid raw LIBERO transitions and can
be used by the PyTorch VALUE and ACP path after conversion to LeRobotDataset v3.
The 100 summary-only rows are labels without observations/actions and are never
imported. They cannot be reconstructed or counted as training episodes.

The source view mixes two task-numbering contracts. Forty Phase195 episodes use
the old suite/canonical numbering, while Phase196 uses the corrected task-text
mapping. The converter therefore never trusts source `canonical_task_id` as a
PyTorch `task_index`: it performs an exact text join against the task table used
by `/home/zhang/xzw/datasets/libero_10_image`. Source IDs and both mapping
versions remain episode provenance.

This is a simulation-only path. No human intervention is required and no
`complementary_info.is_intervention` column is synthesized. The current VALUE
inference implementation treats the absent field as all zeros.

## JAX-to-PyTorch contract

| JAX archive | LeRobot v3 field | Meaning |
|---|---|---|
| `camera1` | `observation.images.image` | Main camera, flipped H+W from simulator orientation |
| `camera2` | `observation.images.wrist_image` | Wrist camera, flipped H+W from simulator orientation |
| `robot_state` | `observation.state` | 8D eef position + axis-angle + gripper state |
| `env_action` | `action` | Exact 7D behavior action submitted to LIBERO |
| `policy_action` | `complementary_info.policy_action` | Pre-exploration policy output for provenance |
| `reward` | `next.reward` | Environment reward |
| `success` | `next.success` | Environment success signal |
| `terminated` or `truncated` | `next.done` | Closed episode boundary |
| `truncated` | `next.truncated` | Time-limit boundary |

The converted task order is the same as the existing PyTorch behavior-cloning
dataset:

```text
task_index = exact task-text index from
/home/zhang/xzw/datasets/libero_10_image/meta/tasks.parquet
suite_task_id = LIBERO environment-internal task number (provenance only)
```

Using `policy_action` as the training action is wrong for the 110 exploration
episodes because it did not cause the recorded next state. The converter always
uses `env_action` as `action` and keeps both arrays.

The JAX and PyTorch implementations of normalized VALUE targets, dense target
differences, undiscounted 50-step advantages, and per-task quantile thresholds
have the same equations. This does not imply numerical equality between JAX and
PyTorch model backbones.

The independent PyTorch audit is `tools/audit_phase264_pytorch_rl.py`. Its last
successful report is stored in
`outputs/phase264_libero10_lerobot_v3/PYTORCH_RL_AUDIT.json`. It checked all
82,034 numeric transitions, episode boundaries, provenance, folds, formulae,
and a synthetic ReplayBuffer success/timeout case. The global task maximum
length is 520 frames for every task.

## Historical JAX failures converted to gates

The JAX phases are evidence about failure modes, not the implementation
authority for this PyTorch run:

- Phase230 is rejected: VALUE covered only tasks 0-7, task 6 support was
  invalid, and failure frames received implausibly many positive indicators.
- Phase242 showed that a nominally legal dataset can still lack minority
  outcomes. Every fold report must therefore show task x outcome counts.
- Phase257 validated the n-step arithmetic but had no task 8/9 frame VALUE and
  had 4,632 stored indicator mismatches. Formula tests alone are insufficient;
  stored Parquet annotations must be re-read and compared.
- Phase264 stopped at the VALUE OOF gate. An analytic target is a training
  label, not a model prediction, and cannot fill a missing OOF value.
- Phase248 and Phase261 policy updates regressed sharply. A candidate must pass
  fixed-init-state paired evaluation and a retention/non-regression gate before
  it can replace the baseline.
- Evaluation/held-out archives never enter training. Failure actions remain in
  actor data because the recorded behavior action caused the next state; ACP
  chooses prompt context, not whether the transition exists.

These gates also prohibit labeling every expert frame positive. That shortcut
can measure expert-versus-policy distribution shift rather than useful VALUE
ranking on the behavior policy's own states.

## Two different PyTorch RL interfaces

The SmolVLA route in this project is the offline VALUE/ACP route:

```text
LeRobotDataset -> Pistar06 VALUE -> OOF values -> 50-step advantage
-> per-task ACP prompt indicator -> SmolVLA flow-MSE update
```

`src/lerobot/rl/buffer.py` serves the separate online SAC route:

```text
(state, action, reward, next_state, done, truncated) -> SAC critic/actor
```

The converter now preserves `next.truncated`, and ReplayBuffer import/export
preserves it as well. However, the Phase264 archive does not store the true
post-action final observation for a timeout. ReplayBuffer therefore repeats the
current state at a closed boundary and SAC masks it with `done`. That is safe
against crossing into the next episode, but it is not mathematically equivalent
to time-limit bootstrapping. Do not route SmolVLA through SAC. A future SAC
collector would need to persist `final_observation` explicitly before enabling
bootstrap across truncation.

## Convert

Activate the prepared environment and first run the complete read-only audit:

```bash
conda run -n hico-rl env PYTHONNOUSERSITE=1 \
  python tools/convert_phase264_libero_rollouts.py --audit-only
```

For a quick format smoke test, convert one episode to a fresh temporary path:

```bash
conda run -n hico-rl env PYTHONNOUSERSITE=1 \
  python tools/convert_phase264_libero_rollouts.py \
  --max-episodes=1 \
  --skip-sha256 \
  --output-root=/tmp/phase264_lerobot_smoke
```

Full conversion (the destination must not already exist):

```bash
conda run -n hico-rl env PYTHONNOUSERSITE=1 \
  python tools/convert_phase264_libero_rollouts.py \
  --output-root=outputs/phase264_libero10_lerobot_v3
```

The full conversion embeds two PNG images per frame and therefore needs extra
disk and conversion time. The source NPZ view is approximately 11 GB and stays
unchanged. The local filesystem currently has sufficient capacity, but the
conversion should not target the 58 GB KINGSTON volume.

The converted root contains `CONVERSION_MANIFEST.json` and
`episode_folds.json`. Folds are deterministic, episode-level, and stratified by
PyTorch task index and success/failure. Never make a random frame-level split.

After the task-text correction, complete frame-level episode counts are:

| task | success | failure |
|---:|---:|---:|
| 0 | 6 | 4 |
| 1 | 3 | 7 |
| 2 | 26 | 4 |
| 3 | 45 | 5 |
| 4 | 5 | 5 |
| 5 | 6 | 14 |
| 6 | 4 | 6 |
| 7 | 22 | 8 |
| 8 | 65 | 5 |
| 9 | 5 | 5 |

## PyTorch VALUE and ACP order

1. Load the converted dataset with `LeRobotDataset` and run a small batch smoke
   before starting training.
2. Train Pistar06 VALUE with the two camera keys and 8D state. Begin with one
   fold or a short fixed-step smoke; do not launch the full 8,000-step job until
   the first batch, loss, and checkpoint save pass.
3. Evaluate VALUE out of fold. A model must not annotate the same episodes it
   was trained on. Every fold must use the full-data normalization contract:
   `--acp.task_max_lengths='{0:520,1:520,2:520,3:520,4:520,5:520,6:520,7:520,8:520,9:520}'`.
   Also pass `--acp.oof_fold_index=<fold>` so every prediction records its
   held-out model provenance.
4. Merge all held-out VALUE/advantage rows, then compute the 70th percentile
   threshold once per task over the complete OOF set. Indicators computed
   independently inside each fold are fold-local and are not final ACP labels.
   After all three inference runs, execute
   `python tools/finalize_phase264_oof_acp.py --dry-run`; remove `--dry-run`
   only after it reports zero OOF fold mismatches.
5. Verify every frame has one finite OOF value/advantage and a binary indicator,
   with no episode-boundary crossing.
6. Train SmolVLA from the validated `/media/zhang/KINGSTON/040000` checkpoint
   using ACP-tagged task text. Keep untagged retention data to limit forgetting.
7. Evaluate the updated policy on fixed, held-out LIBERO init states and compare
   task-level success against the Phase139 baseline before collecting another
   round.

## Online simulation collection

Before starting LIBERO, verify that `~/.libero/config.yaml` points to the
resources installed in the active environment.  A stale config from another
LeRobot virtualenv makes `make_env` fail while the Python packages themselves
are installed.  For `hico-rl`, the relevant values are:

```yaml
assets: /home/zhang/.cache/hico-rl-libero-assets
bddl_files: /home/zhang/anaconda3/envs/hico-rl/lib/python3.10/site-packages/libero/libero/bddl_files
benchmark_root: /home/zhang/anaconda3/envs/hico-rl/lib/python3.10/site-packages/libero/libero
init_states: /home/zhang/anaconda3/envs/hico-rl/lib/python3.10/site-packages/libero/libero/init_files
```

`suite_task_id` is LIBERO's internal ordering and is not the PyTorch task
index.  The collector resolves the exact task text to the PyTorch task index
and stores both values.  This is required because LIBERO-10's internal order
differs from the task order in the existing PyTorch dataset.

For a new PyTorch simulation round, use `tools/collect_libero_pytorch_rollouts.py`.
It reuses the same `make_env`, LIBERO processor, and SmolVLA processor path as
`lerobot-eval`, but persists every transition instead of only aggregate metrics:

```bash
PYTHONNOUSERSITE=1 python tools/collect_libero_pytorch_rollouts.py \
  --policy-path=/home/zhang/xzw/two/Evo-RL-main/.cache/checkpoints/040000/pretrained_model \
  --output-root=outputs/libero_online_round_001 \
  --task-suite=libero_10 --task-ids=0,1,2,3,4,5,6,7,8,9 \
  --episodes-per-task=5 --device=cuda --seed=1000
```

The current container has no `/dev/nvidia*` device, so the command above must
be run on the host where `nvidia-smi` sees the RTX 5080.  The expected first
checks on that host are:

```bash
conda activate hico-rl
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
```

Then run a one-episode smoke before a full round:

```bash
PYTHONNOUSERSITE=1 python tools/collect_libero_pytorch_rollouts.py \
  --policy-path=/home/zhang/xzw/two/Evo-RL-main/.cache/checkpoints/040000/pretrained_model \
  --output-root=/tmp/libero_pytorch_smoke \
  --task-suite=libero_10 --task-ids=0 --episodes-per-task=1 \
  --device=cuda --seed=123
```

The smoke is successful only when it writes a closed episode with
`next.done`, `next.truncated`, `episode_success`, and the task-text mapping;
an aggregate evaluation score alone is insufficient.

The observation/action contract is:

```text
LIBERO raw pixels + robot_state
  -> preprocess_observation
  -> LiberoProcessorStep (H/W flip, quaternion -> axis-angle, 8D state)
  -> SmolVLA preprocessor -> policy action
  -> SmolVLA postprocessor -> LIBERO env action
```

The collector stores the behavior action as `action`, retains the unperturbed
unnormalized policy action for provenance, and never synthesizes a
human-intervention field. Its action processing order is identical to
`lerobot_eval.py`: policy output -> policy postprocessor -> `{action: tensor}`
-> environment postprocessor -> `env.step`.
Run `lerobot-dataset-report` and the episode-boundary audit after collection.

`lerobot-value-infer` supports the pinned `acp.task_max_lengths` contract. Do
not omit it during fold inference. It currently calculates quantile thresholds
on the loaded subset, so fold indicators must be replaced by one global
per-task threshold pass after all OOF advantages have been assembled. A
full-data smoke is useful for wiring only; it is not an unbiased VALUE
evaluation.

## What is still missing

- The full LeRobot v3 conversion is complete at
  `outputs/phase264_libero10_lerobot_v3`; the source NPZ view remains unchanged.
- Tasks 0, 1, 2, and 6 have fewer than five raw episodes in one outcome class.
  Conversion cannot repair this imbalance.
- A PyTorch VALUE checkpoint and true out-of-fold predictions do not exist yet.
- The global OOF ACP finalizer exists, but cannot run successfully until real
  three-fold predictions exist. ACP annotations and the subsequent SmolVLA
  policy update do not exist yet.
- For the next online simulation round, preserve suite task ID, canonical task
  ID/text, init state ID, checkpoint SHA, terminal flags, both policy and
  behavior actions, and exploration parameters exactly as the collector does.
- `lerobot-eval` already provides the correct PyTorch LIBERO observation and
  action interface and remains useful for aggregate metrics/video; the collector
  is the training-data path because it persists the complete episode payload.

Real-robot collection is a separate path. It additionally needs calibrated
cameras/robot, synchronized timestamps, safety stop, operator identity, genuine
takeover intervals, outcome labeling, latency/drop diagnostics, and hardware
configuration provenance. None of those human-intervention requirements should
be copied into LIBERO simulation data.
