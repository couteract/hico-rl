# Evo-RL + custom SmolVLA migration prompt

你正在把用户自定义的 SmolVLA policy 接入 Evo-RL。所有操作都以 Evo-RL 为主工程，不覆盖整个 LeRobot 源码树。

## 固定路径

- Evo-RL 主工程：`/home/zhang/xzw/two/Evo-RL-main`
- 自定义 policy 来源：`/home/zhang/xzw/mma-smolvla-master/src/lerobot/policies/smolvla`
- 目标环境：`hico-rl`

## 已确认硬件

- GPU：NVIDIA GeForce RTX 5080
- NVIDIA driver：570.153.02
- Driver reported CUDA：12.8
- Toolkit：CUDA 12.8
- RTX 5080 对应 Blackwell 架构，CUDA/PyTorch wheel 必须使用支持 Blackwell 的版本。
- 不修改系统 NVIDIA driver/CUDA，不把 sudo 密码写入任何文件。

## 版本策略

- Evo-RL 当前基线：LeRobot 0.4.4。
- 自定义 policy 来源声明：LeRobot 0.4.0。
- 两者属于同一 LeRobot 系列，优先认为 API 可兼容，但必须通过导入、配置、checkpoint 和一次前向验证确认。
- Python 固定 3.10。
- PyTorch 首选 2.7.1 + cu128，torchvision 0.22.1；默认使用上海交通大学 PyTorch 镜像（可用 `PYTORCH_INDEX_URL` 切换到官方源）。如 wheel 或驱动检查显示不适用，暂停并重新选择匹配版本。
- `mamba_ssm` 是可选但开启 Mamba 时必需的 CUDA 扩展；安装/编译失败必须显式报告。
- RTX 5080 的计算能力为 `sm_120`。`mamba-ssm 2.2.4` 默认构建脚本只包含到 `sm_90`，不能直接使用其默认 wheel；必须从带 `csrc/selective_scan` 的完整源码构建，并加入 `-gencode arch=compute_120,code=sm_120`。
- `mamba-ssm 2.2.4` 同时提供 Mamba-1/Mamba-2 Python 模块（包括 `mamba_ssm.modules.mamba2`）；`causal-conv1d` 是可选加速依赖，不等同于 Mamba-2 本体。

## 迁移范围

从 `mma-smolvla-master` 迁移或对照：

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `src/lerobot/policies/smolvla/processor_smolvla.py`
- `src/lerobot/policies/smolvla/temporal_adapter_mamba.py`
- `src/lerobot/policies/smolvla/smolvla_adapter.py`
- `src/lerobot/policies/mambavision/`

保留 Evo-RL 原有实现：

- `policies/factory.py`、`policies/pretrained.py`
- `configs/`、`rl/`、`values/`
- Evo1、自定义机器人、数据集和处理器扩展

禁止：

- 用 mma-smolvla-master 的完整 `src/lerobot` 覆盖 Evo-RL。
- 用 `sudo` 安装 Python 依赖。
- 在代码、日志、脚本、git 历史中保存密码。

## 必须检查的兼容点

1. `lerobot.policies.factory.get_policy_class("smolvla")` 仍返回 `SmolVLAPolicy`。
2. `make_policy_config("smolvla")` 能正常创建和序列化配置。
3. 自定义配置新增字段（LoRA、Mamba、ACG、loss）不会破坏 0.4.4 的 dataclass/processor。
4. `smolvla_adapter.py` 中旧的 `lerobot.policies.smolvla.smolvla` 导入必须改为实际模块 `modeling_smolvla`。
5. `use_mamba_temporal=False` 时基础 SmolVLA 可导入，不应强制导入 `mamba_ssm`。
6. `use_mamba_temporal=True` 时缺少 CUDA 扩展必须报出明确安装/驱动错误。
7. action 输出最后一维必须等于 Evo-RL 环境 action space。
8. Evo-RL 的 `lerobot-value-train`、`lerobot-value-infer` 和原有 policy import 必须继续工作。

## 验收命令

```bash
cd /home/zhang/xzw/two/Evo-RL-main
conda activate hico-rl
python scripts/check_smolvla_compat.py
```

CUDA 可用后再补充：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -c 'import mamba_ssm; print(mamba_ssm.__file__)'
```

## 工作原则

- 先备份/对比，再编辑。
- 每次只处理一个兼容边界。
- 先 CPU/import/config，再 CUDA/model/checkpoint，再 rollout。
- 没有完成实际 forward 和 checkpoint 加载，不得声称 policy 已完成迁移。

## 已完成的环境验证（2026-09-05）

- `hico-rl`: Python 3.10.21，PyTorch 2.7.1+cu128，CUDA 12.8，RTX 5080 可用。
- LeRobot 0.4.4 的 factory/config/SmolVLA modeling/processor 导入和配置序列化通过。
- 从用户提供的 `lerobot.zip.partaa/partab` 提取并复用 `av 15.1.0`、OpenCV 4.12、Mamba-2 Python 源码等；未覆盖 Evo-RL 的完整 `src/lerobot`。
- `mamba_ssm 2.2.4` 已在本机 CUDA 12.8 上以 `sm_120` 重新编译；`TemporalMambaAdapter` 在 RTX 5080 上前向/反向通过，输出形状和梯度均正常。
- 可复用 wheel 已保存到 `/home/zhang/.cache/hico-rl-wheels/mamba_ssm-2.2.4-cp310-cp310-linux_x86_64.whl`；环境脚本会优先使用它。
- 旧备份中的预编译 `selective_scan_cuda` 仅含 `sm_90`，在 RTX 5080 上会报 `no kernel image is available`，不得继续使用。
