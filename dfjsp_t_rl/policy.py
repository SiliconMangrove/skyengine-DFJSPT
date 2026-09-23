"""Event-driven graph PPO over executable joint scheduling candidates."""
from __future__ import annotations

from typing import Mapping
from copy import deepcopy
import math
import time
import torch
from torch import nn
from torch.distributions import Categorical

from .belief import DurationBelief
from .search import CandidateSearch, Candidate
from .traffic import RollingTrafficPlanner, TrafficPlan
from .realtime import PlanningJob, applicable, materialize
from experiment.algorithm_platform_plugins.dfjsp_t.pibt import PIBTRouter

DEFAULT_NODE_DIMS: dict[str, int] = {"jobs": 8, "operations": 8, "machines": 8, "agvs": 10, "roads": 6}
RELATIONS: dict[str, tuple[str, str]] = {
    "job_operation": ("jobs", "operations"), "precedence": ("operations", "operations"),
    "operation_machine": ("operations", "machines"), "machine_road": ("machines", "roads"),
    "agv_road": ("agvs", "roads"), "road_neighbor": ("roads", "roads"),
    "agv_operation": ("agvs", "operations"), "operation_source": ("operations", "roads"),
}


def graph_observation(scene: dict, belief: DurationBelief, router: RollingTrafficPlanner) -> tuple[dict, dict]:
    router.configure(scene)
    jobs, machines, agents = scene["jobs"], scene["machines"], scene["agents"]
    operations = [(job, op) for job in jobs for op in job["ops"]]
    lookup = {"jobs": {j["job_id"]: i for i, j in enumerate(jobs)},
              "operations": {(j["job_id"], op["op_id"]): i for i, (j, op) in enumerate(operations)},
              "machines": {m["id"]: i for i, m in enumerate(machines)},
              "agvs": {a["id"]: i for i, a in enumerate(agents)}}
    roads = sorted(router.free)
    road_index = {cell: i for i, cell in enumerate(roads)}
    height, width = len(scene["grid"]), len(scene["grid"][0])
    nodes: dict[str, list] = {name: [] for name in DEFAULT_NODE_DIMS}
    relations: dict = {name: [] for name in RELATIONS}
    attributes: dict = {name: [] for name in RELATIONS}
    for name in RELATIONS:
        relations[name] = []
    for job in jobs:
        remaining = [op for op in job["ops"] if op["status"] != "FINISHED"]
        location = job["material_location"] or (-height, -width)
        nodes["jobs"].append([len(remaining) / 10, len(job["ops"]) / 10, float(job["delivered"]),
                              max(0, scene["time"] - job["release"]) / 100, float(job["carrier_id"] is not None), float(job["material_location"] is not None),
                              location[0] / height, location[1] / width])
    for index, (job, op) in enumerate(operations):
        options = op["machine_options_with_time"]
        mid = op["assigned_machine"] if op["assigned_machine"] is not None else min(options, key=lambda p: p[1])[0]
        mean, std, tail = belief.process_summary(op, mid, op["processed_time"])
        nodes["operations"].append([float(op["status"] == value) for value in ("PENDING", "PROCESSING", "SUSPENDED", "FINISHED")]
                                    + [mean / 100, std / 100, tail / 100, float(op["transfer_requested"])])
        relations["job_operation"].append((lookup["jobs"][job["job_id"]], index))
        if job["material_location"] is not None and tuple(job["material_location"]) in road_index:
            relations["operation_source"].append((index, road_index[tuple(job["material_location"])]))
        if op["op_id"]:
            relations["precedence"].append((lookup["operations"][(job["job_id"], op["op_id"] - 1)], index))
        for option_mid, _ in options:
            target = machines[lookup["machines"][option_mid]]
            mean, std, tail = belief.process_summary(op, option_mid)
            distance = router.distance(job["material_location"], target["location"]) if job["material_location"] is not None else 0.0
            relations["operation_machine"].append((index, lookup["machines"][option_mid]))
            attributes["operation_machine"].append([mean / 100, std / 100, tail / 100, min(distance, 10000) / 100])
    for machine in machines:
        x, y = machine["location"]
        repair = belief.repair_time(scene, "machine_failure", machine["down_elapsed"], .5) if machine["status"] != "OK" else 0.0
        nodes["machines"].append([x / height, y / width, float(machine["status"] == "OK"), repair / 100,
                                  len(machine["input_queue"]) / 10, len(machine["buffer_jobs"]) / machine["buffer_capacity"],
                                  float(machine["current_op"] is not None), machine["down_elapsed"] / 100])
        relations["machine_road"].append((lookup["machines"][machine["id"]], road_index[(x, y)]))
    for agent in agents:
        x, y = agent["pos"]
        task = agent["current_task"]
        target = agent["pos"] if task is None else task["destination"] if agent["loaded"] else task["source"]
        nodes["agvs"].append([x / height, y / width, float(agent["status"] == "OK"), float(agent["loaded"]),
                              float(task is None), agent["handling_remaining"] / 10,
                              min(router.distance(agent["pos"], target), 10000) / 100, agent["down_elapsed"] / 100,
                              target[0] / height, target[1] / width])
        relations["agv_road"].append((lookup["agvs"][agent["id"]], road_index[(x, y)]))
        if task is not None:
            job = jobs[lookup["jobs"][task["job_id"]]]
            key = (task["job_id"], min(task["op_id"], len(job["ops"]) - 1))
            relations["agv_operation"].append((lookup["agvs"][agent["id"]], lookup["operations"][key]))
    for cell in roads:
        neighbours = [(cell[0] + dx, cell[1] + dy) for dx, dy in router.moves[1:] if (cell[0] + dx, cell[1] + dy) in road_index]
        occupants = [a for a in agents if tuple(a["pos"]) == cell]
        service = [m for m in machines if tuple(m["location"]) == cell]
        nodes["roads"].append([cell[0] / height, cell[1] / width, len(neighbours) / 4,
                               len(occupants), float(any(a["status"] != "OK" for a in occupants)),
                               sum(len(m["buffer_jobs"]) / m["buffer_capacity"] for m in service)])
        relations["road_neighbor"].extend((road_index[cell], road_index[other]) for other in neighbours)
    return {"nodes": {name: torch.tensor(rows, dtype=torch.float32).reshape(-1, DEFAULT_NODE_DIMS[name]) for name, rows in nodes.items()},
            "edges": {name: torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T for name, pairs in relations.items()},
            "edge_features": {name: torch.tensor(attributes[name] or [[0.0] * 4 for _ in pairs], dtype=torch.float32).reshape(-1, 4) for name, pairs in relations.items()}}, lookup


def candidate_tensors(scene: dict, plans: list[Candidate], lookup: dict, belief: DurationBelief, router: RollingTrafficPlanner) -> dict:
    jobs = {j["job_id"]: j for j in scene["jobs"]}
    machines = {m["id"]: m for m in scene["machines"]}
    tokens, features, owners, summary, scopes = [], [], [], [], []
    for index, plan in enumerate(plans):
        scopes.append(plan.scope)
        summary.append([min(plan.cost, 1e6) / 100, min(plan.tail, 1e6) / 100, plan.traffic.waits / 100,
                        len(plan.traffic.blocked) / 10, len(plan.cancel) / 10, len(plan.release) / 10,
                        len(plan.dispatch) / 10, len(plan.assignments) / 10])
        entries = list(plan.dispatch.items())
        # Include existing pending and finished-goods assignments as tokens too.
        seen = {key for key, _ in entries}
        for task in plan.assignments.values():
            key = (task["job_id"], task["op_id"])
            if key not in seen:
                mid = next((mid for mid, m in machines.items() if tuple(m["location"]) == tuple(task["destination"])), None)
                entries.append((key, mid))
                seen.add(key)
        for key, mid in entries:
            aid = next((aid for aid, task in plan.assignments.items() if (task["job_id"], task["op_id"]) == key), None)
            op = next((op for op in jobs[key[0]]["ops"] if op["op_id"] == key[1]), None)
            mean, std, _ = belief.process_summary(op, mid) if op is not None and mid is not None else (0, 0, 0)
            tokens.append([lookup["operations"].get(key, -1), lookup["machines"].get(mid, -1), lookup["agvs"].get(aid, -1)])
            features.append([mean / 100, std / 100, min(plan.traffic.arrivals.get(aid, 0), 10000) / 100,
                             float(key in plan.cancel), float(aid in plan.release), float(aid is None),
                             len(machines[mid]["buffer_jobs"]) / machines[mid]["buffer_capacity"] if mid is not None else 0,
                             float(op is None)])
            owners.append(index)
    return {"tokens": torch.tensor(tokens, dtype=torch.long).reshape(-1, 3),
            "token_features": torch.tensor(features, dtype=torch.float32).reshape(-1, 8),
            "owners": torch.tensor(owners, dtype=torch.long), "summaries": torch.tensor(summary, dtype=torch.float32).reshape(-1, 8),
            "scopes": torch.tensor(scopes, dtype=torch.long)}


class JointPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)) for name, dim in DEFAULT_NODE_DIMS.items()})
        self.messages = nn.ModuleDict({name: nn.Linear(hidden_dim + 4, hidden_dim) for name in RELATIONS})
        self.reverse = nn.ModuleDict({name: nn.Linear(hidden_dim + 4, hidden_dim) for name in RELATIONS})
        self.updates = nn.ModuleDict({name: nn.GRUCell(hidden_dim, hidden_dim) for name in DEFAULT_NODE_DIMS})
        self.global_encoder = nn.Sequential(nn.Linear((hidden_dim + 1) * len(DEFAULT_NODE_DIMS), hidden_dim), nn.SiLU())
        self.history_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.scope_embedding = nn.Embedding(5, hidden_dim)
        self.joint_token = nn.Sequential(nn.Linear(hidden_dim * 3 + 8, hidden_dim), nn.SiLU())
        self.plan_head = nn.Sequential(nn.Linear(hidden_dim * 3 + 8, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode_nodes(self, graph: dict) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        nodes = {name: self.encoders[name](values.to(device)) for name, values in graph["nodes"].items()}
        edges: dict[str, torch.Tensor] = {name: value.to(device) for name, value in graph["edges"].items()}
        attributes: dict[str, torch.Tensor] = {name: value.to(device) for name, value in graph["edge_features"].items()}
        if "degrees" in graph:
            counts: dict[str, torch.Tensor] = graph["degrees"]
        else:
            counts = {name: value.new_zeros((len(value), 1)) for name, value in nodes.items()}
            for name, (source, target) in RELATIONS.items():
                edge: torch.Tensor = edges[name]
                counts[target].index_add_(0, edge[1], nodes[target].new_ones((edge.shape[1], 1)))
                counts[source].index_add_(0, edge[0], nodes[source].new_ones((edge.shape[1], 1)))
        counts = {name: value.clamp_min(1) for name, value in counts.items()}
        for _ in range(3):
            messages = {name: torch.zeros_like(value) for name, value in nodes.items()}
            for name, (source, target) in RELATIONS.items():
                edge = edges[name]
                attrs = attributes[name]
                if not edge.shape[1]:
                    continue
                messages[target].index_add_(0, edge[1], self.messages[name](torch.cat([nodes[source][edge[0]], attrs], -1)))
                messages[source].index_add_(0, edge[0], self.reverse[name](torch.cat([nodes[target][edge[1]], attrs], -1)))
            nodes = {name: self.updates[name](messages[name] / counts[name], value) for name, value in nodes.items()}
        return nodes

    def encode_graphs(self, graph: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Encode disjoint graphs together, without mixing nodes or pooling across graphs."""
        nodes: dict[str, torch.Tensor] = self._encode_nodes(graph)
        pooled: list[torch.Tensor] = []
        for name, values in nodes.items():
            sums: torch.Tensor = values.new_zeros((graph["count"], self.hidden_dim))
            sums.index_add_(0, graph["graph_ids"][name], values)
            counts: torch.Tensor = graph["node_counts"][name]
            pooled.append(torch.cat([sums / counts.clamp_min(1), counts.log1p()], dim=1))
        return nodes, self.global_encoder(torch.cat(pooled, dim=1))

    def encode(self, graph: dict, history: torch.Tensor | None) -> tuple[dict, torch.Tensor]:
        nodes: dict[str, torch.Tensor] = self._encode_nodes(graph)
        pooled = torch.cat([torch.cat([value.mean(0) if len(value) else value.new_zeros(self.hidden_dim),
                                       value.new_tensor([math.log1p(len(value))])]) for value in nodes.values()])
        encoded = self.global_encoder(pooled)
        previous = torch.zeros_like(encoded) if history is None else history.to(encoded.device)
        context = self.history_gru(encoded.unsqueeze(0), previous.unsqueeze(0)).squeeze(0)
        return nodes, context

    def forward(self, graph: dict, candidates: dict, history: torch.Tensor | None = None) -> dict:
        nodes, context = self.encode(graph, history)
        return self.decode(nodes, context, candidates)

    def decode(self, nodes: dict, context: torch.Tensor, candidates: dict) -> dict:
        values = {name: value.to(context.device) for name, value in candidates.items()}
        count = len(values["scopes"])
        tokens = values["tokens"]
        parts = []
        for column, name in enumerate(("operations", "machines", "agvs")):
            padded = torch.cat([nodes[name], context.new_zeros((1, self.hidden_dim))])
            parts.append(padded[torch.where(tokens[:, column] < 0, len(nodes[name]), tokens[:, column])])
        encoded_tokens = self.joint_token(torch.cat([*parts, values["token_features"]], -1))
        pooled = context.new_zeros((count, self.hidden_dim))
        pooled.index_add_(0, values["owners"], encoded_tokens)
        counts = torch.bincount(values["owners"], minlength=count).clamp_min(1).unsqueeze(1)
        pooled = pooled / counts
        plan_logits = self.plan_head(torch.cat([pooled, context.expand(count, -1), self.scope_embedding(values["scopes"]), values["summaries"]], -1)).squeeze(-1)
        return {"plan_logits": plan_logits, "value": self.value_head(context).squeeze(), "history": context}

    def statistics(self, output: dict, candidates: dict, selected: int) -> tuple:
        distribution = Categorical(logits=output["plan_logits"])
        index: torch.Tensor = candidates["scopes"].new_tensor(selected, device=output["plan_logits"].device)
        return distribution.log_prob(index), output["value"], distribution.entropy()


class DFJSPTPolicyPlugin:
    HYPERPARAMETERS = {"fields": [{"name": name, "type": kind, "default": default, "min": minimum} for name, kind, default, minimum in (
        ("hidden_dim", "integer", 128, 16), ("candidate_limit", "integer", 24, 6),
        ("scenario_count", "integer", 8, 1), ("search_seconds", "number", 0.0, 0.0),
        ("routing_horizon", "integer", 24, 2), ("decision_interval", "integer", 5, 1),
        ("inference_budget_ms", "number", 300.0, 50.0))]}

    def __init__(self, hidden_dim: int = 128, device: str = "cpu", candidate_limit: int = 24,
                 scenario_count: int = 8, search_seconds: float = 0.0, routing_horizon: int = 24,
                 decision_interval: int = 5, max_production_actions: int = 32, max_logistics_actions: int = 32,
                 checkpoint: str | None = None, inference_budget_ms: float = 300.0):
        payload = None
        if checkpoint:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if payload["schema_version"] != 4:
                raise ValueError("transaction-policy architecture requires a version 4 checkpoint")
            parameters = payload["policy_parameters"]
            hidden_dim = parameters["hidden_dim"]
            candidate_limit, scenario_count = parameters["candidate_limit"], parameters["scenario_count"]
            search_seconds, routing_horizon = parameters["search_seconds"], parameters["routing_horizon"]
            decision_interval = parameters["decision_interval"]
            inference_budget_ms = parameters["inference_budget_ms"]
            max_production_actions, max_logistics_actions = parameters["max_production_actions"], parameters["max_logistics_actions"]
        if not 50 <= inference_budget_ms <= 300:
            raise ValueError("inference_budget_ms must be between 50 and 300")
        self.inference_budget = inference_budget_ms / 1000
        self.search_slice = search_seconds
        if str(device) == "cpu":
            torch.set_num_threads(1)
        self.model = JointPolicy(hidden_dim).to(device)
        # Frozen during each collection/update batch. Search is an observation
        # generator; PPO differentiates only the conditional final selection.
        self.proposal_model = deepcopy(self.model).requires_grad_(False).eval()
        self.settings = (candidate_limit, scenario_count, search_seconds, routing_horizon, decision_interval,
                         max_production_actions, max_logistics_actions)
        self.decision_interval = decision_interval
        self.reset()
        if payload is not None:
            self.load_state_dict(payload["policy"])
        self.model.eval()

    @classmethod
    def hyperparameter_schema(cls) -> dict:
        return cls.HYPERPARAMETERS

    @property
    def checkpoint_parameters(self) -> dict:
        candidates, scenarios, seconds, horizon, interval, production, logistics = self.settings
        return {"hidden_dim": self.model.hidden_dim, "device": "cpu", "candidate_limit": candidates,
                "scenario_count": scenarios, "search_seconds": seconds, "routing_horizon": horizon,
                "decision_interval": interval, "max_production_actions": production, "max_logistics_actions": logistics,
                "inference_budget_ms": self.inference_budget * 1000}

    def reset(self) -> None:
        candidates, scenarios, seconds, horizon, _, production, logistics = self.settings
        self.belief = DurationBelief()
        self.router = RollingTrafficPlanner(horizon)
        self.search_router = RollingTrafficPlanner(horizon)
        self.traffic = PIBTRouter(torch.initial_seed())
        self.history: torch.Tensor | None = None
        self.last_action_data = None
        self.last_decision = -10**9
        self.last_reschedule = -10**9
        self.ready_keys: set = set()
        self.pending: PlanningJob | None = None
        self.last_search = -10**9
        self.last_recovery = -10**9
        self.requested = True
        self.last_positions = None
        self.stalled = 0
        self.recovery_until = -1
        self.diagnostics: dict = {}
        self.planning_metrics: dict[str, float] = {name: 0.0 for name in ("planning_decisions", "planning_seconds", "planning_max_seconds",
            "inference_steps", "inference_seconds", "inference_max_seconds", "inference_over_budget_steps",
            "search_slices", "search_completed", "search_invalidated", "routing_steps")}

    def state_dict(self) -> dict:
        return {**{f"actor.{key}": value for key, value in self.model.state_dict().items()},
                **{f"proposal.{key}": value for key, value in self.proposal_model.state_dict().items()}}

    def load_state_dict(self, state: Mapping) -> None:
        self.model.load_state_dict({key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")})
        self.proposal_model.load_state_dict({key.removeprefix("proposal."): value for key, value in state.items() if key.startswith("proposal.")})

    def sync_proposal(self) -> None:
        self.proposal_model.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def bootstrap_value(self, observation: dict) -> float:
        scene = observation["planning_observation"]
        self.belief.observe(scene)
        graph, _ = graph_observation(scene, self.belief, self.router)
        _, context = self.model.encode(graph, self.history)
        return float(self.model.value_head(context).squeeze())

    def _search_steps(self, scene: dict, affected: dict):
        # Beliefs and common random numbers stay frozen for the planning epoch.
        search = CandidateSearch(self.search_router, deepcopy(self.belief), *self.settings[:3],
                                 self.settings[5], self.settings[6], self.decision_interval)
        encoded = None
        history = self.history
        ranked_count = 0
        scores: list[float] = []

        def ranker(plans: list[Candidate]):
            nonlocal encoded, ranked_count
            if encoded is None:
                graph, lookup = graph_observation(scene, search.belief, search.router)
                yield None
                nodes, context = self.proposal_model.encode(graph, history)
                yield None
                encoded = nodes, context, lookup
            nodes, context, lookup = encoded
            new: list[Candidate] = plans[ranked_count:]
            if new:
                tokens: dict = candidate_tensors(scene, new, lookup, search.belief, search.router)
                yield None
                scores.extend(self.proposal_model.decode(nodes, context, tokens)["plan_logits"].tolist())
                yield None
                ranked_count = len(plans)
            return max(range(len(scores)), key=scores.__getitem__)

        yield from search.build_steps(scene, affected, ranker)

    def _choose(self, scene: dict, plans: list[Candidate], deterministic: bool) -> Candidate:
        graph, lookup = graph_observation(scene, self.belief, self.router)
        candidates = candidate_tensors(scene, plans, lookup, self.belief, self.router)
        output = self.model(graph, candidates, self.history)
        selected: int = int(output["plan_logits"].argmax() if deterministic else Categorical(logits=output["plan_logits"]).sample())
        log_prob, value, _ = self.model.statistics(output, candidates, selected)
        self.last_action_data = {"graph": graph, "candidates": candidates, "selected": selected,
                                 "history": None if self.history is None else self.history.detach().cpu(),
                                 "old_log_prob": float(log_prob), "old_value": float(value)}
        self.history = output["history"].detach()
        self.last_decision = scene["time"]
        self.planning_metrics["planning_decisions"] += 1
        return plans[selected]

    @torch.no_grad()
    def act(self, observation: dict, action_mask=None, deterministic: bool = False) -> dict:
        started: float = time.perf_counter()
        from sky_executor.grid_factory.factory.Component.RaceFMS.recovery import RecoveryWindow
        scene = observation["planning_observation"]
        self.last_action_data = None
        self.belief.observe(scene)
        self.router.configure(scene)
        initial_decision = self.last_decision < 0
        if initial_decision:
            # Cold graph/kernel work consumes this tick's budget before search.
            route = TrafficPlan({}, {}, 0.0, 0, [], [0] * len(scene["agents"]))
            self._choose(scene, [Candidate(0, traffic=route, cost=0.0, tail=0.0)], deterministic)
        affected = RecoveryWindow.inspect(scene)
        positions = tuple(tuple(a["pos"]) for a in scene["agents"])
        moving = any(a["status"] == "OK" and a["task_phase"] in {"TO_PICKUP", "TO_DROPOFF"} for a in scene["agents"])
        self.stalled = self.stalled + 1 if positions == self.last_positions and moving else 0
        self.last_positions = positions
        recovery = self.stalled >= self.decision_interval and scene["time"] - self.last_recovery >= self.router.horizon
        if recovery:
            self.last_recovery = scene["time"]
            self.recovery_until = scene["time"] + self.router.horizon
            affected["hard"] = True
            affected["agents"].update(a["id"] for a in scene["agents"] if a["current_task"] is not None)
            affected["jobs"].update(a["current_task"]["job_id"] for a in scene["agents"] if a["current_task"] is not None)
        affected["allow_reschedule"] = affected["hard"] or scene["time"] - self.last_reschedule >= self.decision_interval
        ready = {("operation", j["job_id"], op["op_id"]) for j in scene["jobs"] for index, op in enumerate(j["ops"])
                 if op["status"] == "PENDING" and op["assigned_machine"] is None
                 and (index == 0 or j["ops"][index - 1]["status"] == "FINISHED")}
        ready.update(("transport", t["task_id"]) for t in scene["tasks"] if t["assigned_agent_id"] is None)
        event = bool(affected["event_types"] & {"job_release", "machine_recovery", "agv_recovery", "processing_time_deviation"})
        self.requested |= bool(ready - self.ready_keys) or event or affected["hard"]
        self.ready_keys = ready
        if affected["hard"] or (self.pending is not None and self.pending.scene["grid"] != scene["grid"]):
            if self.pending is not None:
                self.planning_metrics["search_invalidated"] += 1
            self.pending = None

        # Search proposes scheduling transactions; the traffic executor advances
        # committed work every tick regardless of proposal age or search progress.
        moves: list[int] = [0] * len(scene["agents"])
        action = {"job_actions": {"transfer_requests": []}, "assign_actions": {"assignments": {}, "pending_transfers": []},
                  "agent_actions": moves}
        if self.pending is None and (self.requested or scene["time"] - self.last_search >= self.decision_interval):
            self.pending = PlanningJob(self._search_steps(scene, affected), scene)
            self.last_search = scene["time"]
            self.requested = False
        work_deadline = started + self.inference_budget * .65
        if self.search_slice > 0:
            work_deadline = min(work_deadline, time.perf_counter() + self.search_slice)
        if self.pending is not None and not self.pending.done and self.pending.revision == self.pending.consumed:
            self.pending.advance(work_deadline)
            self.planning_metrics["search_slices"] += 1
        plans = None
        if (not initial_decision and self.pending is not None and self.pending.revision > self.pending.consumed
                and time.perf_counter() < started + self.inference_budget * .60):
            before = self.pending.scene
            plans = [deepcopy(plan) for plan in self.pending.result if applicable(plan, before, scene)]
            age = scene["time"] - before["time"]
            for plan in plans:
                materialize(plan, scene)
                plan.cost, plan.tail = max(0.0, plan.cost - age), max(0.0, plan.tail - age)
            self.pending.consumed = self.pending.revision
        if self.pending is not None and self.pending.done and self.pending.consumed == self.pending.revision:
            self.planning_metrics["search_completed"] += 1
            self.pending = None
        if plans:
            progressing: bool = any(a["current_task"] is not None for a in scene["agents"]) or any(
                op["status"] in {"PROCESSING", "SUSPENDED"}
                for job in scene["jobs"] for op in job["ops"] if op["status"] != "FINISHED")
            productive: list[Candidate] = [plan for plan in plans if plan.dispatch or plan.assignments]
            # An idle plant with admissible work has no useful WAIT action.
            # Waiting remains available while previously committed work advances.
            if productive and not progressing:
                plans = productive
            plan = self._choose(scene, plans, deterministic)
            plan.traffic.moves = moves
            action = plan.action(scene)
            changed = bool(plan.dispatch or plan.assignments or plan.cancel or plan.release or plan.queues)
            if changed:
                self.pending = None
                self.requested = True
            if plan.cancel or plan.release or plan.queues:
                self.last_reschedule = scene["time"]
        assigned: set[int] = set(action["assign_actions"]["assignments"])
        assigned.update(action.get("reschedule", {}).get("release_agvs", ()))
        routes: list[dict] = self.traffic.plan(scene, assigned)
        action["agent_actions"] = [item["path"][0] for item in routes]
        self.planning_metrics["routing_steps"] += 1
        seconds = time.perf_counter() - started
        self.planning_metrics["inference_steps"] += 1
        self.planning_metrics["inference_seconds"] += seconds
        self.planning_metrics["inference_max_seconds"] = max(self.planning_metrics["inference_max_seconds"], seconds)
        self.planning_metrics["inference_over_budget_steps"] += int(seconds > self.inference_budget)
        self.planning_metrics["planning_seconds"] += seconds
        self.planning_metrics["planning_max_seconds"] = self.planning_metrics["inference_max_seconds"]
        self.diagnostics = {"decision_due": self.last_action_data is not None, "decision_seconds": seconds,
                            "inference_budget_ms": self.inference_budget * 1000, "over_budget": seconds > self.inference_budget,
                            "search_pending": self.pending is not None, "search_completed": self.planning_metrics["search_completed"]}
        return action


__all__ = ["DEFAULT_NODE_DIMS", "JointPolicy", "DFJSPTPolicyPlugin"]
