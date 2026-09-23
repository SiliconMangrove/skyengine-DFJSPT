"""Known probability ratios and returns exercise PPO diagnostic definitions."""
import math
import unittest
from types import SimpleNamespace

import torch

from dfjsp_t_rl.trainer import CTDEPPOTrainer
from dfjsp_t_rl.policy import DEFAULT_NODE_DIMS, RELATIONS
from experiment.rl_platform.api import Trajectory, Transition


class TinyPolicyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim: int = 1
        self.logit = torch.nn.Parameter(torch.tensor(math.log(3.0)))

    def encode_graphs(self, graph):
        return {}, torch.zeros(graph["count"], 1)

    def history_gru(self, encoded, history):
        return torch.zeros_like(history)

    def decode(self, nodes, context, candidates):
        return {"history": context}

    def statistics(self, output, candidates, scope, selected):
        distribution = torch.distributions.Categorical(logits=torch.stack([self.logit, self.logit * 0]))
        return distribution.log_prob(torch.tensor(0)), self.logit * 0, distribution.entropy()


def trajectory(old_probability: float, rewards: list[float], values: list[float]) -> Trajectory:
    transitions: list[Transition] = []
    for index, (reward, value) in enumerate(zip(rewards, values)):
        sample: dict = {"old_log_prob": math.log(old_probability), "old_value": value,
                       "graph": {"nodes": {name: torch.empty(0, dim) for name, dim in DEFAULT_NODE_DIMS.items()},
                                 "edges": {name: torch.empty(2, 0, dtype=torch.long) for name in RELATIONS},
                                 "edge_features": {name: torch.empty(0, 4) for name in RELATIONS}},
                       "candidates": {}, "scope": 0, "selected": 0}
        transitions.append(Transition(None, None, reward, None, index == len(rewards) - 1, False,
                                      {"delta_t": 1.0, "_policy": sample}))
    return Trajectory(transitions, metadata={"bootstrap_value": 0.0})


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_known_ratio_and_perfect_behavior_values(self) -> None:
        policy = SimpleNamespace(model=TinyPolicyModel(), sync_proposal=lambda: None)
        trainer = CTDEPPOTrainer(policy, learning_rate=0.0, gae_lambda=1.0, epochs=2)
        result: dict = trainer.update([trajectory(0.5, [1.0, 2.0], [3.0, 2.0])])
        self.assertAlmostEqual(result["approx_kl"], 0.5 - math.log(1.5), places=6)
        self.assertEqual(result["clip_fraction"], 1.0)
        self.assertAlmostEqual(result["explained_variance"], 1.0)
        self.assertEqual(result["episode_reward_mean"], 3.0)

    def test_unchanged_policy_and_constant_target(self) -> None:
        policy = SimpleNamespace(model=TinyPolicyModel(), sync_proposal=lambda: None)
        trainer = CTDEPPOTrainer(policy, learning_rate=0.0, gae_lambda=1.0, epochs=1)
        result: dict = trainer.update([trajectory(0.75, [0.0, 2.0], [0.0, 0.0])])
        self.assertAlmostEqual(result["approx_kl"], 0.0, places=6)
        self.assertEqual(result["clip_fraction"], 0.0)
        self.assertIsNone(result["explained_variance"])

    def test_cancellation_prevents_the_next_minibatch_update(self) -> None:
        policy = SimpleNamespace(model=TinyPolicyModel(), sync_proposal=lambda: None)
        trainer = CTDEPPOTrainer(policy, learning_rate=0.1, epochs=1)
        before: torch.Tensor = policy.model.logit.detach().clone()

        def cancel() -> None:
            raise InterruptedError("cancelled")

        with self.assertRaises(InterruptedError):
            trainer.update([trajectory(0.5, [1.0, 2.0], [0.0, 0.0])], cancel_check=cancel)
        self.assertEqual(policy.model.logit.item(), before.item())
        self.assertEqual(trainer.optimizer.state, {})


if __name__ == "__main__":
    unittest.main()
