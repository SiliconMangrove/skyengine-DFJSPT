# 使用说明 / Usage

## 1. 安装与目录

使用 [README](../README.md) 中的固定版本和安装命令。目录结构为：

```text
workspace/
├── skyengine/          # 800fc454dc29124e95e04e9abc58db3d87fd0383
└── skyengine-DFJSPT/   # v0.3.0
```

Python 使用 3.11。Docker 安装会构建平台与工厂引擎，并生成环境配置；前端实际地址以启动脚本输出为准。Windows 可通过 WSL2 运行上述 Linux 命令。

DFJSP-T 的联合策略使用配套环境内置 PIBT，无需另外选择 MAPF 算子。SkyEngine 其他容器算子的安装参见其固定版本的 [Quickstart](https://github.com/SiliconMangrove/skyengine/blob/800fc454dc29124e95e04e9abc58db3d87fd0383/docs/QUICKSTART.md)。

## 2. 数据与训练

以下路径相对于 `skyengine/`：

| 用途 | 路径 |
| --- | --- |
| 训练 | `dataset/dfjsp_t_train/train.jsonl` |
| 验证 | `dataset/dfjsp_t_validation/validation.jsonl` |
| Benchmark | `dataset/dfjsp_t_benchmark/benchmark.jsonl` |

这些是配套 SkyEngine 生成器产生的合成实例。训练只使用训练分区，模型选择使用验证分区，benchmark 留作最终评价。

网页路径：进入“算法实验与优化平台” → 创建训练实验 → 选择 `ctde_ppo@0.3.0` → 选择数据集和参数 → 校验、编译并执行。查看采样、网络更新及模型验证进度，完成后下载或选用生成的模型。

网页流水线 PPO 训练需要 CUDA GPU，采样和验证使用独立 CPU 核心。CPU 训练使用第 3 节的通用命令行入口；平台本身能以 CPU 模式启动，不代表网页流水线训练支持 CPU。

主要参数：

| 参数 | 含义 |
| --- | --- |
| `num_envs` | 并行采样环境数，首次使用可设为 1 |
| `device` | `cpu`、`auto` 或具体 CUDA 设备 |
| `candidate_limit` | 每轮候选事务数量，默认 24 |
| `scenario_count` | 候选估计场景数，默认 8 |
| `routing_horizon` | 路径预测步数，默认 24 |
| `decision_interval` | 平稳期间发起规划的间隔，默认 5 |
| `inference_budget_ms` | 每步协作式推理预算，默认 300 毫秒 |
| `graph_batch_size` | 单次合并编码的图数量，默认 16 |
| `sequence_length` | 学习序列长度，默认 32 |

推理采用协作式时间预算，执行时间会受机器负载影响。订单完成包含最终成品运输；查看完成率时同时查看被截断回合，不把达到步数上限当作完成。

## 3. 无界面训练

无界面路径直接使用正式工厂环境，不要求启动网页服务。先安装 `uv`，在 `skyengine/` 目录同步该版本的锁定依赖：

```bash
uv sync --locked --python 3.11
export PYTHONPATH="$PWD:$PWD/../skyengine-DFJSPT"
uv run --no-sync python -m experiment.rl_platform.cli train ../skyengine-DFJSPT/configs/ctde_ppo_formal.json
```

在 Windows PowerShell 中，路径分隔符使用分号：

```powershell
uv sync --locked --python 3.11
$env:PYTHONPATH = "$PWD;$PWD/../skyengine-DFJSPT"
uv run --no-sync python -m experiment.rl_platform.cli train ../skyengine-DFJSPT/configs/ctde_ppo_formal.json
```

运行前在配置中调整 `training.episodes`、`environment.num_envs` 和 `environment.max_steps`。随附的历史配置为 1000 回合、4 个进程和 1000 步上限；对需要完整交付的初始运行，可将步数上限与 `reward.kwargs.horizon` 同时设为 5000。`checkpoint.path` 相对于当前工作目录，默认写入 `skyengine/dataset/rl_platform_runs/`。

若需要以 Python 包形式安装算法，在已同步 SkyEngine 环境后执行：

```bash
uv pip install --python .venv/bin/python --no-deps -e ../skyengine-DFJSPT
```

Windows 对应解释器路径为 `.venv/Scripts/python.exe`。运行环境模块时仍需从 `skyengine/` 目录执行，并保留其在 `PYTHONPATH` 中。

## 4. 模型评价与继续训练

推荐通过网页平台对训练生成的模型创建“测试”任务：选择验证集或 benchmark、绑定对应的模型 artifact、执行后检查订单交付率与完工时间，并打开回放。

模型的策略状态采用 schema 4，应与 `ctde_ppo@0.3.0` 配套。网页平台保存的模型 artifact 与通用命令行 checkpoint 的外层封装不同，应在各自保存入口对应的流程中使用。

通用命令行训练支持在配置 `checkpoint.resume` 中指定其此前保存的 checkpoint，并再次运行同一条 `train` 命令；它会加载策略、优化器和随机状态，但命令行回合计数重新开始，`training.episodes` 表示本次调用的回合数。

配套平台旧版通用 CLI 的 `evaluate` / `replay` 子命令不自动恢复配置中的 checkpoint。已训练模型的评价请使用上面的网页 artifact 流程。

## 5. 生成独立合成数据

在 `skyengine/` 中，使用一个新的输出目录：

```bash
uv run --no-sync python -m dataset.dfjsp_t.generate_dfjsp_t --split train --count 100 --seed 20261021 --output dataset/dfjsp_t_custom_train
```

生成后将配置中的数据路径指向新 JSONL。验证和 benchmark 分别使用对应的 `--split`、不同的种子及不同目录。

## 6. 停止网页服务

在 `skyengine/` 执行 `bash stop.sh`。训练任务可先在监控页面请求协作式取消，等待任务结束后再停止服务。
