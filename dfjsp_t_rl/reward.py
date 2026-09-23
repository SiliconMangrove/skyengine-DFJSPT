"""SMDP reward preserving the maximum-completion-time objective."""
from __future__ import annotations
from typing import Mapping

class SMDPMakespanReward:
    HYPERPARAMETERS = {"fields": [
        {"name": "horizon", "label": "时间归一化上限", "type": "number", "default": 1000.0, "min": 1.0, "step": 10.0},
        {"name": "shaping_scale", "label": "势函数塑形权重", "type": "number", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
    ]}

    def __init__(self, horizon: float = 1000.0, shaping_scale: float = .1):
        self.horizon = float(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon 必须大于 0")
        self.shaping_scale = float(shaping_scale)
    @classmethod
    def hyperparameter_schema(cls): return cls.HYPERPARAMETERS
    def potential(self, metrics: Mapping[str, float]) -> float:
        return -float(metrics.get("remaining_work_lower_bound", metrics.get("remaining_work", 0.0))) / max(self.horizon, 1.0)
    def compute(self, transition) -> float:
        before = transition.info.get("before_metrics", {})
        after = transition.info.get("metrics", {})
        dt = float(transition.info.get("delta_t", after.get("timeline", 0) - before.get("timeline", 0)))
        return -dt / self.horizon + self.shaping_scale * (self.potential(after) - self.potential(before))
    def terminal(self, metrics: Mapping[str, float]) -> float:
        # Elapsed time is already charged by compute(); do not charge Cmax twice.
        return 0.0
