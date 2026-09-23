"""Disjoint graph inputs shared across PPO epochs, never model activations."""
from __future__ import annotations

from collections import OrderedDict
from itertools import accumulate

import torch
from torch import Tensor

from .policy import DEFAULT_NODE_DIMS, RELATIONS


class GraphInputCache:
    """Bound device inputs by bytes; release all entries after one PPO update."""

    def __init__(self, device: torch.device, capacity_bytes: int):
        self.device: torch.device = device
        self.capacity_bytes: int = capacity_bytes
        self.entries: OrderedDict[tuple[int, int, int], tuple[dict, list[dict], int]] = OrderedDict()
        self.bytes: int = 0

    def get(self, episode: int, samples: list[dict], first: int, end: int) -> tuple[dict, list[dict]]:
        key: tuple[int, int, int] = (episode, first, end)
        if key in self.entries:
            graph, candidates, _ = self.entries[key]
            self.entries.move_to_end(key)
            return graph, candidates

        graphs: list[dict] = [sample["graph"] for sample in samples[first:end]]
        sizes: dict[str, list[int]] = {
            name: [len(graph["nodes"][name]) for graph in graphs] for name in DEFAULT_NODE_DIMS}
        offsets: dict[str, list[int]] = {name: list(accumulate(lengths, initial=0)) for name, lengths in sizes.items()}
        nodes: dict[str, Tensor] = {name: torch.cat([graph["nodes"][name] for graph in graphs]) for name in sizes}
        edges: dict[str, Tensor] = {}
        for name, (source, target) in RELATIONS.items():
            edges[name] = torch.cat([
                graph["edges"][name] + torch.tensor([[offsets[source][index]], [offsets[target][index]]])
                for index, graph in enumerate(graphs)], dim=1)
        attributes: dict[str, Tensor] = {
            name: torch.cat([graph["edge_features"][name] for graph in graphs]) for name in RELATIONS}
        degrees: dict[str, Tensor] = {name: torch.zeros(len(value), 1) for name, value in nodes.items()}
        for name, (source, target) in RELATIONS.items():
            edge: Tensor = edges[name]
            degrees[source].index_add_(0, edge[0], torch.ones(edge.shape[1], 1))
            degrees[target].index_add_(0, edge[1], torch.ones(edge.shape[1], 1))
        graph: dict = {
            "nodes": nodes, "edges": edges, "edge_features": attributes, "degrees": degrees,
            "graph_ids": {name: torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor(lengths))
                          for name, lengths in sizes.items()},
            "node_counts": {name: torch.tensor(lengths, dtype=torch.float32).unsqueeze(1)
                            for name, lengths in sizes.items()},
        }
        candidates: list[dict] = [sample["candidates"] for sample in samples[first:end]]
        size: int = sum(value.numel() * value.element_size() for fields in graph.values() for value in fields.values())
        size += sum(value.numel() * value.element_size() for item in candidates for value in item.values())
        # Evict before allocating replacements; autograd owns any still-live inputs.
        while self.entries and self.bytes + size > self.capacity_bytes:
            _, (_, _, removed_bytes) = self.entries.popitem(last=False)
            self.bytes -= removed_bytes
        graph = {name: {key: value.to(self.device) for key, value in fields.items()}
                 for name, fields in graph.items()}
        candidates = [{name: value.to(self.device) for name, value in item.items()} for item in candidates]
        graph["sizes"] = sizes
        graph["count"] = len(graphs)
        if size <= self.capacity_bytes:
            self.entries[key] = graph, candidates, size
            self.bytes += size
        return graph, candidates
