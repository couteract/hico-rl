# 5090 Reference Scripts

These scripts were copied from the active 5090 machine path:

`/media/zhang/KINGSTON/hico/5090数据拷贝/src`

They are preserved as read-only reference assets. They are not the current
machine's entry points and still contain the original `/media/wdy/...` paths,
conda environment name, and project layout.

The four known failed policy snapshots are intentionally not included:

- `smolvla11失败版`
- `smolvla0107  92训练 acg90`
- `smolvla1221`
- `smolvla1228`

The paired checkpoint is `/media/zhang/KINGSTON/040000/pretrained_model`.
Its recorded LIBERO-10 evaluation is 43/50 (86%), with 5 episodes per task.

## Reference files

- `train_libero10_mambavision.sh`: simulation/LIBERO training reference
- `train_mambavision2.sh`: feature experiment training reference
- `train_mambavision_lora.sh`: LoRA training reference
- `train_real.sh`: real-robot training reference
- `run_smolvla_train.sh`: older standard SmolVLA training reference
- `eval_libero10_acg.sh`: LIBERO-10 ACG evaluation reference
- `eval_metaworld_acg.sh`: MetaWorld evaluation reference

The active 040000 configuration is summarized in the checkpoint itself. The
important values are `chunk_size=50`, `n_action_steps=50`, Mamba temporal
adapter enabled, `mamba_history_length=50`, `mamba_d_state=64`,
`mamba_d_conv=4`, `mamba_expand=2`, `num_vlm_layers=16`, and
`num_steps=10`. The old evaluation script overrides the rollout to
`chunk_size=5`, `n_action_steps=5`, enables ACG, and uses seed 1000.

Current-machine equivalents are in `../../local/`. They replace the old
absolute paths and keep the original reference scripts unchanged.
