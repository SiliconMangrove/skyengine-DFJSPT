# DFJSP-T 图策略与滚动联合调度

为 SkyEngine 提供联合调度策略、候选搜索、交通规划和 PPO 训练组件。

- [训练使用说明](../skyengine/docs/联合调度训练使用说明.md)
- [模块与参数](../docs/DFJSP-T算法仓库.md)
- [算法结构](../docs/DFJSP-T联合调度算法结构.md)
- [训练与数据生成](../docs/DFJSP-T训练平台与数据生成.md)

控制器使用第三版 checkpoint，默认每步 300ms 推理预算；路径和候选规划可跨步推进。新训练需使用当前控制器参数。

延迟检查（在 skyengine 根目录执行）：

```powershell
python ../skyengine-DFJSPT/scripts/profile_realtime.py --workers 4 --cases 4 --steps 100 --check-learning
```

该命令使用训练集有限步采样，不代表完整训练效果验收。
