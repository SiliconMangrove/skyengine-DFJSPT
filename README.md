# DFJSP-T 图策略与滚动联合调度

为 SkyEngine 提供联合调度策略、候选搜索、交通规划和 PPO 训练组件。

- [训练使用说明](../skyengine/docs/联合调度训练使用说明.md)
- [模块与参数](../docs/DFJSP-T算法仓库.md)
- [算法结构](../docs/DFJSP-T联合调度算法结构.md)
- [训练与数据生成](../docs/DFJSP-T训练平台与数据生成.md)

控制器使用第三版 checkpoint，默认每步 300ms 推理预算；路径和候选规划可跨步推进。新训练需使用当前控制器参数。

PPO 学习器将独立决策图合并编码，`graph_batch_size` 默认 16；循环历史仍按当前权重从头计算，片段内保留时间顺序和梯度。`input_cache_mb` 默认 256（MiB），限制本批输入张量缓存，0 关闭缓存；模型输出不跨优化器更新缓存。批量大小仅改变图编码执行方式，PPO 的 minibatch、epoch、样本顺序、GAE 和奖励保持原定义。批量浮点运算不保证与逐图计算逐位一致。

平台展示 epoch、小批进度、优化器更新次数和用时；日志在计算块边界按约 5 秒间隔刷新，首尾强制记录。更新 `skyengine-DFJSPT` 的同时需同步配套 `skyengine` 平台代码。此次改造未运行测试或性能评测，尚无加速倍数与训练效果的实测结论。

延迟检查（在 skyengine 根目录执行）：

```powershell
python ../skyengine-DFJSPT/scripts/profile_realtime.py --workers 4 --cases 4 --steps 100 --check-learning
```

该命令使用训练集有限步采样，不代表完整训练效果验收。
