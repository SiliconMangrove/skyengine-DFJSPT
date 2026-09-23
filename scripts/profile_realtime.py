"""Measure the complete policy.act call in independent formal CPU workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))


def collect_case(arguments: tuple) -> dict:
    instance, index, steps, check_learning = arguments
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from dfjsp_t_rl.policy import DFJSPTPolicyPlugin
    from dfjsp_t_rl.reward import SMDPMakespanReward
    from dfjsp_t_rl.trainer import CTDEPPOTrainer
    from experiment.rl_platform.env_adapter import SkyEngineTrainingEnv
    from experiment.rl_platform.rollout import RolloutCollector

    seed: int = 1548110269 + index
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    policy = DFJSPTPolicyPlugin(device="cpu")
    env = SkyEngineTrainingEnv(instance)
    latencies: list[float] = []
    original = policy.act

    def measured(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        latencies.append(time.perf_counter() - started)
        return result

    policy.act = measured
    collector = RolloutCollector(policy, SMDPMakespanReward(), training_only=True)
    started = time.perf_counter()
    trajectory = collector.collect(env, steps, episode_id=str(index), seed=1271865744 + index)
    seconds = time.perf_counter() - started
    scene = env.raw_observation["planning_observation"]
    total_duration = sum(t.info["delta_t"] for t in trajectory.transitions)
    if total_duration != len(latencies):
        raise AssertionError("cross-step rewards lost physical time")
    max_log_prob_error: float = 0.0
    with torch.no_grad():
        for transition in trajectory.transitions:
            sample = transition.info["_policy"]
            output = policy.model(sample["graph"], sample["candidates"], sample["history"])
            log_prob, _, _ = policy.model.statistics(output, sample["candidates"], sample["selected"])
            max_log_prob_error = max(max_log_prob_error, abs(float(log_prob) - sample["old_log_prob"]))
    if max_log_prob_error > 1e-5:
        raise AssertionError("stored candidate distribution does not reproduce the collection log probability")
    update = None
    if check_learning:
        if instance["split"] != "train":
            raise ValueError("the learning check requires training data")
        update = CTDEPPOTrainer(policy, epochs=1, minibatch_size=64, sequence_length=8).update([trajectory])
        if not all(np.isfinite(value) for value in update.values() if isinstance(value, (int, float))):
            raise AssertionError("nonfinite PPO update")
    ordered = sorted(latencies)
    report = {"instance": instance["instance_id"], "steps": len(latencies), "decisions": len(trajectory.transitions),
              "collection_seconds": seconds, "mean_ms": sum(latencies) / len(latencies) * 1000,
              "p95_ms": ordered[int(.95 * (len(ordered) - 1))] * 1000, "max_ms": max(latencies) * 1000,
              "max_step": max(range(len(latencies)), key=latencies.__getitem__),
              "over_300ms": sum(value > .3 for value in latencies), "log_prob_error": max_log_prob_error,
              "finished_operations": sum(op["status"] == "FINISHED" for job in scene["jobs"] for op in job["ops"]),
              "completed_jobs": sum(job["delivered"] for job in scene["jobs"]),
              "metrics": policy.planning_metrics, "learning_check": update}
    env.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/dfjsp_t_train/train.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--check-learning", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with open(args.dataset, encoding="utf-8") as stream:
        instances = [json.loads(next(stream)) for _ in range(args.cases)]
    jobs = [(instance, index, args.steps, args.check_learning) for index, instance in enumerate(instances)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(collect_case, jobs))
    report = {"workers": args.workers, "step_limit": args.steps, "results": results,
              "passed_latency": all(result["over_300ms"] == 0 for result in results)}
    content = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content, flush=True)
    if not report["passed_latency"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
