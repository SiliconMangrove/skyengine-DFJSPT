# SkyEngine DFJSP-T

[中文](README.zh-CN.md) · [Usage](docs/USAGE.md) · [Release notes](docs/RELEASE.md)

Graph-based scheduling and PPO training for dynamic flexible job shops with transportation. The controller selects executable production and transport transactions; PIBT advances committed vehicle movements in the SkyEngine factory environment.

## Features

- Joint machine selection, transport assignment and dispatch ordering.
- Graph observations of jobs, machines, vehicles and roads, with recurrent policy history.
- Rolling candidate search, buffer admission checks and persistent transport commitments.
- PPO training with batched graph encoding, checkpoint saving and SkyEngine experiment monitoring.
- Formal headless simulation and integration with SkyEngine's training, evaluation and replay interface.

## Supported version pair

| Component | Version |
| --- | --- |
| This repository | `v0.3.0` — controller `ctde_ppo@0.3.0`, policy checkpoint schema 4 |
| SkyEngine | [`800fc454dc29124e95e04e9abc58db3d87fd0383`](https://github.com/SiliconMangrove/skyengine/tree/800fc454dc29124e95e04e9abc58db3d87fd0383) |
| Python | 3.11 |

This package runs with the matching SkyEngine environment. Use the pinned pair for this release. The environment's `uv.lock` records the dependency versions.

## Quick start: Linux / WSL2

Install Git, Docker Engine with Compose, and start the Docker daemon. Run the following in a new directory; keep the two repositories as siblings.

```bash
git clone --branch v0.3.0 https://github.com/SiliconMangrove/skyengine-DFJSPT.git
git clone https://github.com/SiliconMangrove/skyengine.git
cd skyengine
git checkout 800fc454dc29124e95e04e9abc58db3d87fd0383
bash install.sh
bash start.sh
```

Open the frontend URL printed by `start.sh` and choose **算法实验与优化平台** (Algorithm Experiment Platform). Select **DFJSP-T 图策略与滚动联合调度**, version **0.3.0**, then choose the training dataset and create a training run. The web platform's pipelined PPO trainer requires a Docker-accessible NVIDIA GPU. For an initial run, use one sampling worker, one validation worker, a small episode count and a 5,000-step episode budget. For CPU training, use the [headless command-line workflow](docs/USAGE.md#3-无界面训练).

The companion repository includes separate synthetic training, validation and benchmark datasets. The platform provides model evaluation and factory replay after training. See [Usage](docs/USAGE.md) for dataset paths, training parameters, headless execution and checkpoint handling.

## Repository layout

```text
dfjsp_t_rl/       policy, candidate search, routing, rewards and PPO learner
configs/         headless training configurations
scripts/         existing profiling utility
tests/           algorithm regression tests
docs/            installation, usage and release information
```

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for project and dependency notices.
