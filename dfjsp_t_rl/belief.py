"""Conditional duration beliefs calibrated exclusively from completed observations."""
from __future__ import annotations

import hashlib
import math
from statistics import NormalDist


def duration_quantile(spec: dict | None, u: float, nominal: float = 1.0,
                      elapsed: float = 0.0, processing: bool = True) -> float:
    """Quantile of remaining duration, conditional on not having finished.

    Processing priors use the environment's positive, mean-preserving law.
    Repair priors use its integer duration and rounding convention.
    """
    spec = spec or {"dist": "fixed", "value": nominal}
    raw_kind: str = spec.get("dist", "fixed").lower()
    kind: str = raw_kind.removeprefix("multiplier_")
    unit: float = 1.0 if raw_kind.startswith("multiplier_") or not processing else nominal
    if kind == "fixed":
        total = nominal if processing else max(1, int(spec.get("value", 1)))
        return max(0.0, total - elapsed)
    low, high = float(spec.get("low", unit)), float(spec.get("high", unit))
    mode: float = float(spec.get("mode", (low + high) / 2 if processing else low))
    mean: float = (low + high) / 2
    if kind in {"triangular", "discrete_triangular"}:
        mean = (low + high + mode) / 3
    normal = NormalDist(float(spec.get("mean", unit)), max(float(spec.get("std", 0)), 1e-12))
    mass: float = 1 - normal.cdf(0)
    if kind == "normal":
        mean = normal.mean + normal.stdev * math.exp(-0.5 * (normal.mean / normal.stdev) ** 2) / (math.sqrt(2 * math.pi) * mass)
    if kind == "lognormal":
        mean = 1.0
    scale: float = nominal / mean if processing else 1.0
    threshold: float = elapsed / scale
    if not processing:
        if kind in {"uniform", "discrete_uniform"}:
            low = max(low, math.floor(elapsed) + 1)
            return max(1.0, min(high, low + math.floor(u * (high - low + 1))) - elapsed)
        threshold = math.floor(elapsed) + 0.5
    if high == low and kind in {"uniform", "discrete_uniform", "triangular", "discrete_triangular"}:
        return max(0.0, low * scale - elapsed)
    if kind in {"uniform", "discrete_uniform"}:
        if kind == "discrete_uniform":
            low = max(low, math.floor(threshold) + 1)
            draw = min(high, low + math.floor(u * (high - low + 1)))
        else:
            draw = max(low, threshold) + u * max(0.0, high - max(low, threshold))
    elif kind in {"triangular", "discrete_triangular"}:
        if threshold <= low:
            cdf = 0.0
        elif threshold >= high:
            cdf = 1.0
        elif threshold < mode:
            cdf = (threshold - low) ** 2 / ((high - low) * (mode - low))
        else:
            cdf = 1 - (high - threshold) ** 2 / ((high - low) * (high - mode))
        q = min(1.0, cdf + (1 - cdf) * u)
        pivot = (mode - low) / (high - low)
        draw = low + math.sqrt(q * (high - low) * (mode - low)) if q < pivot else high - math.sqrt((1 - q) * (high - low) * (high - mode))
    elif kind == "normal":
        cdf = normal.cdf(max(0.0, threshold))
        draw = normal.inv_cdf(min(1 - 1e-12, cdf + (1 - cdf) * u))
    elif kind == "lognormal":
        sigma = float(spec.get("sigma", .2))
        if sigma == 0:
            return max(0.0, nominal - elapsed)
        normal = NormalDist(-sigma * sigma / 2, sigma)
        cdf = normal.cdf(math.log(elapsed / nominal)) if elapsed > 0 else 0.0
        return max(0.0, nominal * math.exp(normal.inv_cdf(min(1 - 1e-12, cdf + (1 - cdf) * u))) - elapsed)
    else:
        raise ValueError(f"unsupported public duration distribution: {kind}")
    return max(0.0, (draw * scale if processing else max(1, round(draw))) - elapsed)


class DurationBelief:
    def __init__(self) -> None:
        self.completed: set[tuple[int, int]] = set()
        self.processing: dict[int, list[float]] = {}
        self.repairs: dict[str, list[float]] = {"machine_failure": [], "agv_failure": []}
        self.down: dict[tuple[str, int], float] = {}
        self.summaries: dict[tuple, tuple[float, float, float]] = {}
        self.histories: dict[tuple, list[float]] = {}

    def observe(self, scene: dict) -> None:
        previous_count: int = len(self.completed)
        for job in scene["jobs"]:
            for op in job["ops"]:
                key = (job["job_id"], op["op_id"])
                if op["status"] == "FINISHED" and key not in self.completed:
                    self.completed.add(key)
                    mid = op["assigned_machine"]
                    nominal = dict(op["machine_options_with_time"])[mid]
                    values = self.processing.setdefault(mid, [])
                    values.append(op["processed_time"] / nominal)
                    del values[:-256]
        for kind, entities in (("machine_failure", scene["machines"]), ("agv_failure", scene["agents"])):
            for item in entities:
                key = (kind, item["id"])
                if item["status"] != "OK":
                    self.down[key] = item["down_elapsed"]
                elif key in self.down:
                    self.repairs[kind].append(self.down.pop(key) + 1)
                    del self.repairs[kind][:-256]
        if len(self.completed) != previous_count:
            self.summaries.clear()
            self.histories.clear()
        else:
            # Keep nominal estimates; conditional residuals belong to one tick.
            self.summaries = {key: value for key, value in self.summaries.items() if key[3] == 0}
            self.histories = {key: value for key, value in self.histories.items() if key[2] == 0}

    @staticmethod
    def uniform(epoch: int, scenario: int, key: str) -> float:
        digest = hashlib.blake2b(f"belief:{epoch}:{scenario}:{key}".encode(), digest_size=8).digest()
        return (int.from_bytes(digest, "big") + .5) / 2**64

    def processing_time(self, op: dict, mid: int, u: float, elapsed: float = 0) -> float:
        nominal = float(dict(op["machine_options_with_time"])[mid])
        key = (mid, nominal, elapsed)
        if key not in self.histories:
            self.histories[key] = sorted(ratio * nominal - elapsed for ratio in self.processing.get(mid, []) if ratio * nominal > elapsed)
        history = self.histories[key]
        weight = len(history) / (len(history) + 32)
        if history and u < weight:
            return history[min(len(history) - 1, int(u / weight * len(history)))]
        value = duration_quantile(op["distributions"].get(str(mid)), (u - weight) / (1 - weight), nominal, elapsed)
        # A duration outside a bounded prior is evidence of model mismatch.
        # Use an explicit exponential residual model until new completions calibrate it.
        if value == 0 and op["status"] in {"PROCESSING", "SUSPENDED"}:
            return -max(1.0, nominal * .1) * math.log1p(-u)
        return value

    def repair_time(self, scene: dict, kind: str, elapsed: float, u: float) -> float:
        history = sorted(value - elapsed for value in self.repairs[kind] if value > elapsed)
        weight = len(history) / (len(history) + 32)
        if history and u < weight:
            return history[min(len(history) - 1, int(u / weight * len(history)))]
        spec = scene["failure_priors"][kind]["repair_time"]
        value = duration_quantile(spec, (u - weight) / (1 - weight), elapsed=elapsed, processing=False)
        return value if value > 0 else -max(1.0, elapsed * .1) * math.log1p(-u)

    def process_summary(self, op: dict, mid: int, elapsed: float = 0) -> tuple[float, float, float]:
        key = (op["job_id"], op["op_id"], mid, elapsed, op["status"])
        if key in self.summaries:
            return self.summaries[key]
        values = [self.processing_time(op, mid, (i + .5) / 32, elapsed) for i in range(32)]
        mean = sum(values) / len(values)
        result = mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), sorted(values)[30]
        self.summaries[key] = result
        return result
