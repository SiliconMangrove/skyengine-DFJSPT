"""Sequence PPO for event decisions with correct time-limit bootstrapping."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
import torch
from torch import Tensor

from .batching import GraphInputCache


def ppo_loss(new_log_prob: Tensor, old_log_prob: Tensor, advantage: Tensor, clip_ratio: float = .2) -> Tensor:
    ratio = (new_log_prob - old_log_prob).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage).mean()


def gae(rewards: list, values: list, terminated: list, gamma: float = 1.0, lam: float = .95,
        delta_t: list | None = None, bootstrap: float = 0.0) -> list[float]:
    durations = delta_t if delta_t is not None else [1.0] * len(rewards)
    result, carry = [0.0] * len(rewards), 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else bootstrap
        discount = gamma ** durations[index]
        continuation = 0.0 if terminated[index] else 1.0
        carry = rewards[index] + discount * continuation * next_value - values[index] + discount * continuation * lam * carry
        result[index] = carry
    return result


class CTDEPPOTrainer:
    HYPERPARAMETERS = {"fields": [{"name": name, "type": kind, "default": value} for name, kind, value in (
        ("learning_rate", "number", .0003), ("gamma", "number", 1.0), ("gae_lambda", "number", .95),
        ("clip_ratio", "number", .2), ("epochs", "integer", 4), ("value_coef", "number", .5),
        ("entropy_coef", "number", .01), ("minibatch_size", "integer", 64), ("sequence_length", "integer", 32),
        ("graph_batch_size", "integer", 16), ("input_cache_mb", "integer", 256))]}
    METRICS = {"fields": [{"key": name, "label": name, "unit": ""} for name in
                          ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                           "explained_variance", "episode_reward_mean", "transitions", "sequences", "collecting_step")]}

    def __init__(self, policy, learning_rate: float = .0003, gamma: float = 1.0, gae_lambda: float = .95,
                 clip_ratio: float = .2, epochs: int = 4, value_coef: float = .5, entropy_coef: float = .01,
                 minibatch_size: int = 64, sequence_length: int = 32, device: str | None = None,
                 graph_batch_size: int = 16, input_cache_mb: int = 256,
                 gpu_ids=None, distributed: bool = False):
        if gamma != 1.0:
            raise ValueError("expected makespan uses undiscounted time cost: gamma must be 1")
        if distributed:
            raise ValueError("distributed learner is not implemented; num_envs controls parallel collection")
        self.policy, self.model = policy, policy.model
        if device is not None:
            self.model.to(device)
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip_ratio
        self.epochs, self.value_coef, self.entropy_coef = epochs, value_coef, entropy_coef
        self.minibatch_size, self.sequence_length = minibatch_size, sequence_length
        if graph_batch_size < 1 or input_cache_mb < 0:
            raise ValueError("graph_batch_size must be positive and input_cache_mb must be non-negative")
        self.graph_batch_size: int = graph_batch_size
        self.input_cache_bytes: int = input_cache_mb * 1024 * 1024
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @classmethod
    def hyperparameter_schema(cls) -> dict:
        return cls.HYPERPARAMETERS

    def update(self, trajectory_batch, *, cancel_check: Callable[[], None] | None = None,
               progress_callback: Callable[[dict], None] | None = None) -> dict:
        device = next(self.model.parameters()).device
        started: float = time.monotonic()
        last_progress: float = started
        episodes: list = []
        old_values: list[float] = []
        episode_rewards: list[float] = []
        advantage_values, returns = [], []
        for trajectory in trajectory_batch:
            transitions = trajectory.transitions
            if not transitions:
                continue
            samples = [t.info["_policy"] for t in transitions]
            values = [sample["old_value"] for sample in samples]
            old_values.extend(values)
            episode_rewards.append(sum(t.reward for t in transitions))
            advantages = gae([t.reward for t in transitions], values, [t.terminated for t in transitions],
                             self.gamma, self.lam, [t.info["delta_t"] for t in transitions],
                             float(trajectory.metadata["bootstrap_value"]))
            offset = len(advantage_values)
            episodes.append((samples, offset))
            advantage_values.extend(advantages)
            returns.extend(a + v for a, v in zip(advantages, values))
        if not advantage_values:
            return {"updated": False, "transitions": 0}
        advantages = torch.tensor(advantage_values, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        targets = torch.tensor(returns, dtype=torch.float32, device=device)
        target_variance: Tensor = targets.var(unbiased=False)
        explained_variance: float | None = None
        if target_variance.item() > 0:
            predictions: Tensor = torch.tensor(old_values, dtype=torch.float32, device=device)
            explained_variance = (1 - (targets - predictions).var(unbiased=False) / target_variance).item()
        chunks = [(episode, start, min(start + self.sequence_length, len(samples)))
                  for episode, (samples, _) in enumerate(episodes) for start in range(0, len(samples), self.sequence_length)]
        chunks_per_batch = max(1, self.minibatch_size // self.sequence_length)
        minibatches: int = (len(chunks) + chunks_per_batch - 1) // chunks_per_batch
        total_updates: int = minibatches * self.epochs
        inputs = GraphInputCache(device, self.input_cache_bytes)
        sums = torch.zeros(4, device=device)
        diagnostic_sums: Tensor = torch.zeros(2, device=device)
        diagnostic_samples: int = 0
        updates = 0

        def publish_progress(stage: str, epoch: int, minibatch: int, *, force: bool = False) -> None:
            nonlocal last_progress
            if progress_callback is None or (not force and time.monotonic() - last_progress < 5):
                return
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            last_progress = time.monotonic()
            progress_callback({"update_stage": stage, "epoch": epoch, "epochs": self.epochs,
                               "minibatch": minibatch, "minibatches": minibatches,
                               "optimizer_steps": updates, "optimizer_steps_total": total_updates,
                               "elapsed_seconds": last_progress - started})

        self.model.train()
        for epoch in range(self.epochs):
            random.shuffle(chunks)
            for first in range(0, len(chunks), chunks_per_batch):
                if cancel_check is not None:
                    cancel_check()
                minibatch: int = first // chunks_per_batch + 1
                publish_progress("history", epoch + 1, minibatch, force=updates == 0)
                log_probs, values_now, entropies, old_probs, indices = [], [], [], [], []
                selected_chunks: list[tuple[int, int, int]] = chunks[first:first + chunks_per_batch]
                prefix_starts: dict[int, set[int]] = {}
                for episode, start, _ in selected_chunks:
                    prefix_starts.setdefault(episode, set()).add(start)
                histories: dict[tuple[int, int], Tensor] = {}
                # Recompute with the current weights. Reuse prefixes only within
                # this optimizer step, and never attach gradients to burn-in.
                with torch.no_grad():
                    for episode, starts in prefix_starts.items():
                        samples, _ = episodes[episode]
                        history: Tensor = targets.new_zeros((1, self.model.hidden_dim))
                        histories[episode, 0] = history
                        for block in range(0, max(starts), self.graph_batch_size):
                            if cancel_check is not None:
                                cancel_check()
                            publish_progress("history", epoch + 1, minibatch)
                            block_end: int = min(block + self.graph_batch_size, max(starts))
                            graph, _ = inputs.get(episode, samples, block, block_end)
                            prefix_nodes, encoded = self.model.encode_graphs(graph)
                            del prefix_nodes
                            for position in range(block_end - block):
                                history = self.model.history_gru(encoded[position:position + 1], history)
                                if block + position + 1 in starts:
                                    histories[episode, block + position + 1] = history
                            del encoded, graph
                for episode, start, end in selected_chunks:
                    samples, offset = episodes[episode]
                    history = histories[episode, start]
                    for block in range(start, end, self.graph_batch_size):
                        if cancel_check is not None:
                            cancel_check()
                        publish_progress("forward", epoch + 1, minibatch)
                        block_end = min(block + self.graph_batch_size, end)
                        graph, candidates = inputs.get(episode, samples, block, block_end)
                        nodes, encoded = self.model.encode_graphs(graph)
                        node_parts: dict[str, tuple[Tensor, ...]] = {
                            name: values.split(graph["sizes"][name]) for name, values in nodes.items()}
                        for position, index in enumerate(range(block, block_end)):
                            sample = samples[index]
                            history = self.model.history_gru(encoded[position:position + 1], history)
                            output = self.model.decode({name: parts[position] for name, parts in node_parts.items()},
                                                       history.squeeze(0), candidates[position])
                            lp, value, entropy = self.model.statistics(output, candidates[position], sample["scope"], sample["selected"])
                            log_probs.append(lp)
                            values_now.append(value)
                            entropies.append(entropy)
                            old_probs.append(sample["old_log_prob"])
                            indices.append(offset + index)
                        del nodes, encoded, node_parts, graph, candidates
                histories.clear()
                log_probs = torch.stack(log_probs)
                old_log_probs: Tensor = torch.tensor(old_probs, device=device)
                actor = ppo_loss(log_probs, old_log_probs, advantages[indices], self.clip)
                with torch.no_grad():
                    log_ratio: Tensor = log_probs - old_log_probs
                    ratio: Tensor = log_ratio.exp()
                    diagnostic_sums += torch.stack([
                        (ratio - 1 - log_ratio).sum(),
                        ((ratio - 1).abs() > self.clip).float().sum(),
                    ])
                    diagnostic_samples += len(indices)
                critic = torch.nn.functional.mse_loss(torch.stack(values_now), targets[indices])
                entropy = torch.stack(entropies).mean()
                loss = actor + self.value_coef * critic - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                publish_progress("backward", epoch + 1, minibatch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                sums += torch.stack([loss.detach(), actor.detach(), critic.detach(), entropy.detach()])
                updates += 1
                publish_progress("minibatch_completed", epoch + 1, minibatch, force=updates == total_updates)
                del log_probs, values_now, entropies, output, history, loss, actor, critic, entropy, lp, value
        self.optimizer.zero_grad(set_to_none=True)
        inputs.entries.clear()
        self.model.eval()
        self.policy.sync_proposal()
        result = dict(zip(("loss", "policy_loss", "value_loss", "entropy"), (sums / updates).cpu().tolist()))
        diagnostics: dict[str, float] = dict(zip(
            ("approx_kl", "clip_fraction"), (diagnostic_sums / diagnostic_samples).cpu().tolist()))
        return {**result, **diagnostics, "explained_variance": explained_variance,
                "episode_reward_mean": sum(episode_rewards) / len(episode_rewards),
                "updated": True, "transitions": len(advantage_values), "sequences": len(chunks)}

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "sequence_length": self.sequence_length}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
