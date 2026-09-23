"""Incremental search must keep ranking when a proposal is infeasible."""
import unittest
from unittest.mock import patch

import torch

from dfjsp_t_rl.policy import (
    DEFAULT_NODE_DIMS, RELATIONS, DFJSPTPolicyPlugin, candidate_tensors,
)
from dfjsp_t_rl.search import Candidate, CandidateSearch
from dfjsp_t_rl.traffic import TrafficPlan


class CandidateRankingTest(unittest.TestCase):
    def test_repeated_candidate_prefix_reuses_scores(self) -> None:
        policy: DFJSPTPolicyPlugin = DFJSPTPolicyPlugin(hidden_dim=128)
        with torch.no_grad():
            policy.proposal_model.plan_head[-1].weight.zero_()
            policy.proposal_model.plan_head[-1].bias.zero_()
        scene: dict = {"jobs": [], "machines": []}
        graph: dict = {
            "nodes": {name: torch.zeros((1, dim)) for name, dim in DEFAULT_NODE_DIMS.items()},
            "edges": {name: torch.empty((2, 0), dtype=torch.long) for name in RELATIONS},
            "edge_features": {name: torch.empty((0, 4)) for name in RELATIONS},
        }
        keep: Candidate = Candidate(0, cost=2.0, tail=2.0, traffic=TrafficPlan({}, {}, 0, 0, [], []))
        better: Candidate = Candidate(1, cost=1.0, tail=1.0, traffic=TrafficPlan({}, {}, 0, 0, [], []))
        selected: list[int] = []

        def repeated_prefix(search: CandidateSearch, scene: dict, affected: dict, ranker):
            plans: list[Candidate] = [keep]
            selected.append((yield from ranker(plans)))
            # An infeasible proposal does not extend the accepted candidate list.
            selected.append((yield from ranker(plans)))
            plans.append(better)
            selected.append((yield from ranker(plans)))
            selected.append((yield from ranker(plans)))
            yield plans

        with patch("dfjsp_t_rl.policy.graph_observation", return_value=(graph, {})), \
                patch.object(CandidateSearch, "build_steps", repeated_prefix), \
                patch.object(policy.proposal_model, "decode", wraps=policy.proposal_model.decode) as decode:
            list(policy._search_steps(scene, {}))
        self.assertEqual(selected, [0, 0, 1, 1])
        self.assertEqual(decode.call_count, 2)
        self.assertEqual([len(call.args[2]["scopes"]) for call in decode.call_args_list], [1, 1])

    def test_empty_candidate_tensors_preserve_feature_width(self) -> None:
        policy: DFJSPTPolicyPlugin = DFJSPTPolicyPlugin(hidden_dim=128)
        values: dict = candidate_tensors({"jobs": [], "machines": []}, [], {}, policy.belief, policy.router)
        self.assertEqual(tuple(values["summaries"].shape), (0, 8))
        nodes: dict = {name: torch.empty((0, 128)) for name in DEFAULT_NODE_DIMS}
        result: dict = policy.model.decode(nodes, torch.zeros(128), values)
        self.assertEqual(tuple(result["plan_logits"].shape), (0,))


if __name__ == "__main__":
    unittest.main()
