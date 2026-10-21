# SkyEngine DFJSP-T

[English](README.md) · [使用说明](docs/USAGE.md) · [版本说明](docs/RELEASE.md)

面向含运输的动态柔性作业车间，提供图策略、滚动联合调度和 PPO 训练组件。策略选择可执行的生产与运输事务，PIBT 在 SkyEngine 工厂环境中持续推进已承诺的车辆运输。

## 功能

- 联合选择加工机器、运输车辆与派工次序。
- 根据订单、机器、车辆和道路构建图观测，结合循环历史进行决策。
- 跨步候选搜索、缓冲资源准入和持续运输承诺。
- PPO 训练、图编码批处理、模型保存及实验进度监控。
- 正式工厂环境的无界面运行，以及 SkyEngine 网页训练、模型评价和回放。

## 配套版本

| 组件 | 版本 |
| --- | --- |
| 本仓库 | `v0.3.0`，控制器 `ctde_ppo@0.3.0`，策略 checkpoint schema 4 |
| SkyEngine | [`800fc454dc29124e95e04e9abc58db3d87fd0383`](https://github.com/SiliconMangrove/skyengine/tree/800fc454dc29124e95e04e9abc58db3d87fd0383) |
| Python | 3.11 |

本算法包与配套 SkyEngine 环境共同使用。按上表检出两个仓库，环境依赖版本记录在 SkyEngine 的 `uv.lock` 中。

## 快速开始：Linux / WSL2

准备 Git、Docker Engine 和 Docker Compose，并启动 Docker。在一个新目录中执行以下命令，保持两个仓库同级：

```bash
git clone --branch v0.3.0 https://github.com/SiliconMangrove/skyengine-DFJSPT.git
git clone https://github.com/SiliconMangrove/skyengine.git
cd skyengine
git checkout 800fc454dc29124e95e04e9abc58db3d87fd0383
bash install.sh
bash start.sh
```

打开启动脚本输出的前端地址，进入“算法实验与优化平台”，选择“DFJSP-T 图策略与滚动联合调度”及 `0.3.0` 版本，选择训练数据集并创建训练任务。支持 CPU；CUDA 训练需要 Docker 能访问 NVIDIA GPU。首次使用可设置单采样进程、CPU、少量训练回合及每回合 5000 步上限。

配套环境已包含独立的合成训练集、验证集和 benchmark。训练完成后，可在平台创建模型测试并查看工厂回放。数据路径、参数、无界面训练和模型使用见[使用说明](docs/USAGE.md)。

## 许可证

采用 [Apache License 2.0](LICENSE)，项目及依赖说明见 [NOTICE](NOTICE)。
