"""Sequence PPO for event decisions with correct time-limit bootstrapping."""
from __future__ import annotations

import random
from collections.abc import Callable
import torch
from torch import Tensor


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
        ("entropy_coef", "number", .01), ("minibatch_size", "integer", 64), ("sequence_length", "integer", 32))]}
    METRICS = {"fields": [{"key": name, "label": name, "unit": ""} for name in
                          ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                           "explained_variance", "episode_reward_mean", "transitions", "sequences", "collecting_step")]}

    def __init__(self, policy, learning_rate: float = .0003, gamma: float = 1.0, gae_lambda: float = .95,
                 clip_ratio: float = .2, epochs: int = 4, value_coef: float = .5, entropy_coef: float = .01,
                 minibatch_size: int = 64, sequence_length: int = 32, device: str | None = None,
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @classmethod
    def hyperparameter_schema(cls) -> dict:
        return cls.HYPERPARAMETERS

    def update(self, trajectory_batch, *, cancel_check: Callable[[], None] | None = None) -> dict:
        device = next(self.model.parameters()).device
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
        sums = torch.zeros(4, device=device)
        diagnostic_sums: Tensor = torch.zeros(2, device=device)
        diagnostic_samples: int = 0
        updates = 0
        self.model.train()
        for _ in range(self.epochs):
            random.shuffle(chunks)
            for first in range(0, len(chunks), chunks_per_batch):
                if cancel_check is not None:
                    cancel_check()
                log_probs, values_now, entropies, old_probs, indices = [], [], [], [], []
                for episode, start, end in chunks[first:first + chunks_per_batch]:
                    samples, offset = episodes[episode]
                    # Recompute the complete prefix under the current weights.
                    # Its graph is discarded; the selected chunk uses BPTT.
                    history = None
                    with torch.no_grad():
                        for sample in samples[:start]:
                            _, history = self.model.encode(sample["graph"], history)
                    for index in range(start, end):
                        sample = samples[index]
                        output = self.model(sample["graph"], sample["candidates"], history)
                        history = output["history"]
                        lp, value, entropy = self.model.statistics(output, sample["candidates"], sample["scope"], sample["selected"])
                        log_probs.append(lp)
                        values_now.append(value)
                        entropies.append(entropy)
                        old_probs.append(sample["old_log_prob"])
                        indices.append(offset + index)
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                sums += torch.stack([loss.detach(), actor.detach(), critic.detach(), entropy.detach()])
                updates += 1
                del log_probs, values_now, entropies, output, history, loss, actor, critic, entropy, lp, value
        self.optimizer.zero_grad(set_to_none=True)
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
