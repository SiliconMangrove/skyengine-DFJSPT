"""Conditional production/transport construction and common-scenario local search."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from bisect import bisect_right

from .belief import DurationBelief
from .traffic import RollingTrafficPlanner, TrafficPlan
from experiment.algorithm_platform_plugins.dfjsp_t.buffer_safety import admission_state, buffer_claims, safe_to_admit


@dataclass
class Candidate:
    scope: int
    dispatch: dict[tuple[int, int], int] = field(default_factory=dict)
    assignments: dict[int, dict] = field(default_factory=dict)
    cancel: set[tuple[int, int]] = field(default_factory=set)
    release: set[int] = field(default_factory=set)
    queues: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    traffic: TrafficPlan | None = None
    cost: float = math.inf
    tail: float = math.inf
    feasible: bool = True
    samples: list[float] = field(default_factory=list)
    route_order: list[int] | None = None
    evacuation: bool = False

    def identity(self) -> tuple:
        """Physical transaction identity; search scope is not a second action."""
        return (tuple(sorted(self.dispatch.items())),
                tuple(sorted((aid, task["job_id"], task["op_id"]) for aid, task in self.assignments.items())),
                tuple(sorted(self.cancel)), tuple(sorted(self.release)),
                tuple(sorted((mid, tuple(keys)) for mid, keys in self.queues.items())))

    def action(self, scene: dict) -> dict:
        machines = {m["id"]: m for m in scene["machines"]}
        jobs = {j["job_id"]: j for j in scene["jobs"]}
        return {
            "reschedule": {"cancel_operations": sorted(self.cancel), "release_agvs": sorted(self.release),
                           "machine_queues": {str(mid): keys for mid, keys in self.queues.items()}} if self.cancel or self.release or self.queues else {},
            "job_actions": {"transfer_requests": [
                {"task_id": task_id(scene, key), "job_id": key[0], "op_id": key[1],
                 "source": list(jobs[key[0]]["material_location"]), "destination": machines[mid]["location"],
                 "destination_machine_id": mid, "candidate_machines": [mid], "ready_time": scene["time"], "priority": jobs[key[0]]["priority"]}
                for key, mid in self.dispatch.items()]},
            "assign_actions": {"assignments": {aid: task["task_id"] for aid, task in self.assignments.items()}, "pending_transfers": []},
            "agent_actions": self.traffic.moves,
        }


def task_id(scene: dict, key: tuple[int, int]) -> int:
    # Allocate from the observed namespace; do not assume op IDs fit in 10000.
    base = max([task["task_id"] for task in scene["tasks"]] + [0]) + 1
    keys = [(j["job_id"], op["op_id"]) for j in scene["jobs"] for op in j["ops"]]
    return base + keys.index(key)


class CandidateSearch:
    def __init__(self, router: RollingTrafficPlanner, belief: DurationBelief, candidate_limit: int = 24,
                 scenarios: int = 8, search_seconds: float = 0.0, max_production_actions: int = 32,
                 max_logistics_actions: int = 32, decision_interval: int = 5):
        self.router, self.belief = router, belief
        self.limit, self.scenarios, self.seconds = candidate_limit, scenarios, search_seconds
        self.max_production, self.max_logistics = max_production_actions, max_logistics_actions
        self.decision_interval = decision_interval
        self.draws: dict[tuple, float] = {}
        self.failures: dict[tuple, tuple[list[int], list[float]]] = {}
        self.durations: dict[tuple, float] = {}

    def _construct(self, scene: dict, scope: int, affected: dict, variant: int,
                   parent: Candidate | None = None) -> Candidate:
        candidate = Candidate(scope)
        machines = {m["id"]: m for m in scene["machines"]}
        jobs = {j["job_id"]: j for j in scene["jobs"]}
        agents = {a["id"]: a for a in scene["agents"]}
        held_jobs = {m["current_op"][0] for m in machines.values() if m["current_op"] is not None}
        if scope >= 2:
            # An accepted transport owns its vehicle until delivery or failure.
            # Reoptimizing a healthy pickup must not reset handling indefinitely.
            candidate.release = {a["id"] for a in agents.values() if a["current_task"] is not None and not a["loaded"]
                                 and a["status"] != "OK"}
        loaded = {(a["current_task"]["job_id"], a["current_task"]["op_id"]) for a in agents.values() if a["loaded"]}
        preserve_queue = {tuple(key) for machine in machines.values() for key in machine["input_queue"]
                          if machine["status"] == "OK" and variant % 3 == 1}
        if scope >= 3:
            candidate.cancel = {(j["job_id"], op["op_id"]) for j in jobs.values() for op in j["ops"]
                                if op["status"] == "PENDING" and op["assigned_machine"] is not None
                                and (j["job_id"], op["op_id"]) not in loaded
                                and (j["job_id"], op["op_id"]) not in preserve_queue
                                and machines[op["assigned_machine"]]["status"] != "OK"}
            candidate.release.update(a["id"] for a in agents.values() if a["current_task"] is not None
                                     and (a["current_task"]["job_id"], a["current_task"]["op_id"]) in candidate.cancel)
        proposed: dict = admission_state(scene, candidate.cancel)
        jobs = {j["job_id"]: j for j in proposed["jobs"]}
        claims: dict[int, set[int]] = buffer_claims(proposed)
        operations: list = [(j, op) for j in jobs.values() for i, op in enumerate(j["ops"])
                            if op["status"] == "PENDING" and (i == 0 or j["ops"][i - 1]["status"] == "FINISHED")
                            and (op["assigned_machine"] is None or (j["job_id"], op["op_id"]) in candidate.cancel)
                            and j["carrier_id"] is None and j["job_id"] not in held_jobs]
        # Distinct constructive orders plus one-change neighborhoods give
        # machine, task order and AGV decisions a common search space.
        operations.sort(key=lambda pair: (sum(min(t for _, t in op["machine_options_with_time"]) for op in pair[0]["ops"] if op["status"] != "FINISHED"), pair[0]["job_id"]), reverse=variant % 2 == 0)
        reservations = {mid: len(m["input_queue"]) for mid, m in machines.items()}
        for index, (job, op) in enumerate(operations[:self.max_production]):
            key = (job["job_id"], op["op_id"])
            options = [(self.belief.process_summary(op, mid)[0] + self.router.distance(job["material_location"], machines[mid]["location"])
                        + reservations[mid] * duration, mid) for mid, duration in op["machine_options_with_time"]
                       if machines[mid]["status"] == "OK"
                       and safe_to_admit(proposed, claims, key[0], key[1], mid)]
            options.sort()
            if not options or not math.isfinite(options[0][0]):
                continue
            if parent is not None and parent.dispatch.get(key) in {mid for _, mid in options}:
                chosen = parent.dispatch[key]
                if index == variant % max(1, len(operations)):
                    alternatives = [mid for _, mid in options if mid != chosen]
                    chosen = alternatives[variant % len(alternatives)] if alternatives else chosen
            else:
                chosen = options[(variant // 2 + index) % min(len(options), 2)][1]
            candidate.dispatch[key] = chosen
            claims[chosen].add(key[0])
            op["assigned_machine"] = chosen
            reservations[chosen] += 1
        tasks = {t["task_id"]: deepcopy(t) for t in scene["tasks"]
                 if (t["job_id"], t["op_id"]) not in candidate.cancel
                 and (t["assigned_agent_id"] is None or t["assigned_agent_id"] in candidate.release)}
        for key, mid in candidate.dispatch.items():
            job = jobs[key[0]]
            if tuple(job["material_location"]) == tuple(machines[mid]["location"]):
                continue
            ident = task_id(scene, key)
            tasks[ident] = {"task_id": ident, "job_id": key[0], "op_id": key[1], "source": job["material_location"],
                           "destination": machines[mid]["location"], "kind": "operation", "assigned_agent_id": None,
                           "ready_time": scene["time"], "pickup_required": True, "priority": job["priority"]}
        available = {aid for aid, a in agents.items() if a["status"] == "OK" and (a["current_task"] is None or aid in candidate.release)}
        ordered_tasks = sorted(tasks.values(), key=lambda t: (t["kind"] != "finished_goods", t["ready_time"], t["task_id"]))
        if variant % 3 == 2:
            ordered_tasks.reverse()
        claimed_jobs: set[int] = set()
        for task in ordered_tasks:
            job = jobs[task["job_id"]]
            if not available or len(candidate.assignments) >= self.max_logistics:
                break
            if task["job_id"] in claimed_jobs or task["ready_time"] > scene["time"] or job["material_location"] is None:
                continue
            if task["kind"] == "operation" and any(op["status"] != "FINISHED" for op in job["ops"] if op["op_id"] < task["op_id"]):
                continue
            task["source"] = job["material_location"]
            source_machine = next((m for m in machines.values() if tuple(m["location"]) == tuple(task["source"])), None)
            at_raw = task["op_id"] == 0 and job["ops"][0]["arrive_machine_at"] < 0 and tuple(task["source"]) == tuple(job["raw_material_source"])
            if source_machine is not None and job["job_id"] not in source_machine["buffer_jobs"] and not at_raw:
                continue
            if task["kind"] == "operation" and tuple(task["source"]) == tuple(task["destination"]):
                continue
            choices = sorted(available, key=lambda aid: (self.router.distance(agents[aid]["pos"], task["source"]), aid))
            aid = choices[(variant // 3) % min(2, len(choices))]
            candidate.assignments[aid] = task
            available.remove(aid)
            claimed_jobs.add(task["job_id"])
        if scope >= 3:
            for mid, machine in machines.items():
                keys = [tuple(key) for key in machine["input_queue"] if tuple(key) not in candidate.cancel]
                if len(keys) > 1 and (scope == 4 or mid in affected["machines"]):
                    keys.sort(key=lambda key: dict(jobs[key[0]]["ops"][key[1]]["machine_options_with_time"])[mid], reverse=bool(variant % 2))
                    if keys != [tuple(key) for key in machine["input_queue"] if tuple(key) not in candidate.cancel]:
                        candidate.queues[mid] = keys
        return candidate

    def _evaluate_steps(self, scene: dict, candidate: Candidate, variant: int) -> None:
        ids = [a["id"] for a in scene["agents"]]
        order = ids[variant % len(ids):] + ids[:variant % len(ids)] if ids else []
        candidate.route_order, candidate.evacuation = order, bool(variant % 2)
        candidate.traffic = yield from self.router.plan_steps(scene, candidate.assignments, candidate.release, order, evacuation=bool(variant % 2))
        costs: list[float] = []
        for scenario in range(self.scenarios):
            costs.append((yield from self._scenario_steps(scene, candidate, scenario)))
            yield None
        candidate.samples = costs
        candidate.cost = sum(costs) / len(costs)
        candidate.tail = sorted(costs)[min(len(costs) - 1, math.floor(.9 * len(costs)))]
        candidate.feasible = math.isfinite(candidate.cost)

    def _scenario_steps(self, scene: dict, plan: Candidate, scenario: int) -> float:
        """Resource-calendar rollout through all visible jobs, including terminal shipping.

        Current transports use the reservation planner. Beyond that window,
        transport is a queueing approximation; no hidden simulator state is used.
        """
        machines = {m["id"]: m for m in scene["machines"]}
        operations = {(j["job_id"], op["op_id"]): op for j in scene["jobs"] for op in j["ops"]}
        jobs = {j["job_id"]: j for j in scene["jobs"]}
        def uniform(key: tuple) -> float:
            cache_key = (scenario, key)
            if cache_key not in self.draws:
                self.draws[cache_key] = self.belief.uniform(scene["time"], scenario, str(key))
            return self.draws[cache_key]

        def interrupted_duration(kind: str, duration: float, identity: tuple):
            prior = scene["failure_priors"][kind]
            if not prior["enabled"] or not math.isfinite(duration):
                return duration
            probability = float(prior.get("prob_per_machine_step" if kind == "machine_failure" else "prob_per_agv_step", 0))
            if probability == 0 and prior.get("mtbf_steps"):
                probability = 1 / float(prior["mtbf_steps"])
            if probability == 0:
                return duration
            cache_key = (scenario, kind, identity)
            ticks, delays = self.failures.setdefault(cache_key, ([], [0.0]))
            # Geometric gaps exactly sample independent per-working-tick failures.
            # Share the same event stream across every candidate in this epoch.
            while not ticks or ticks[-1] <= math.ceil(duration):
                index = len(ticks)
                gap = 1 if probability == 1 else 1 + math.floor(math.log1p(-uniform((kind, "gap", *identity, index))) / math.log1p(-probability))
                ticks.append((ticks[-1] if ticks else 0) + gap)
                delays.append(delays[-1] + self.belief.repair_time(scene, kind, 0, uniform((kind, "repair", *identity, index))))
                if index % 32 == 0:
                    yield None
            return duration + delays[bisect_right(ticks, math.ceil(duration))]

        def processing(op: dict, mid: int, key: tuple) -> float:
            cache_key = (scenario, key, mid)
            if cache_key not in self.durations:
                self.durations[cache_key] = self.belief.processing_time(op, mid, uniform(("p", *key)), op["processed_time"])
            return self.durations[cache_key]
        machine_ready = {mid: self.belief.repair_time(scene, "machine_failure", m["down_elapsed"], uniform(("repair", mid))) if m["status"] != "OK" else 0.0 for mid, m in machines.items()}
        agent_ready = {a["id"]: self.belief.repair_time(scene, "agv_failure", a["down_elapsed"], uniform(("agv_repair", a["id"]))) if a["status"] != "OK" else 0.0 for a in scene["agents"]}
        agent_pos = {a["id"]: tuple(a["pos"]) for a in scene["agents"]}
        job_ready = {j["job_id"]: 0.0 for j in jobs.values()}
        job_pos = {j["job_id"]: j["material_location"] for j in jobs.values()}
        active = {a["id"]: a["current_task"] for a in scene["agents"] if a["current_task"] is not None and a["id"] not in plan.release}
        active.update(plan.assignments)
        delivery: dict[tuple, float] = {}
        shipping: dict[int, float] = {}
        buffer_release = {mid: {("free", index): 0.0 for index in range(m["buffer_capacity"] - len(m["buffer_jobs"]))}
                          for mid, m in machines.items()}
        for mid, m in machines.items():
            for jid in m["buffer_jobs"]:
                departures = [self.router.distance(agent_pos[aid], task["source"]) + scene["pickup_steps"]
                              for aid, task in active.items() if task["job_id"] == jid and tuple(task["source"]) == tuple(m["location"])]
                buffer_release[mid][("job", jid)] = min(departures) if departures else float(self.router.horizon)
        for aid, task in active.items():
            agent = next(a for a in scene["agents"] if a["id"] == aid)
            transport = plan.traffic.arrivals[aid]
            if agent["status"] != "OK":
                transport = self.router.distance(agent["pos"], task["destination"]) + scene["dropoff_steps"] if agent["loaded"] else self.router.distance(agent["pos"], task["source"]) + scene["pickup_steps"] + self.router.distance(task["source"], task["destination"]) + scene["dropoff_steps"]
            arrival = agent_ready[aid] + (yield from interrupted_duration("agv_failure", transport, (aid, task["job_id"], task["op_id"])))
            agent_ready[aid] = arrival
            agent_pos[aid] = tuple(task["destination"])
            delivery[(task["job_id"], task["op_id"])] = arrival
            if task["kind"] == "finished_goods":
                shipping[task["job_id"]] = arrival
            job_pos[task["job_id"]] = task["destination"]
        completed: dict[tuple, float] = {}
        for mid, machine in machines.items():
            if machine["current_op"] is not None:
                key = tuple(machine["current_op"])
                op = operations[key]
                if op["status"] != "FINISHED":
                    duration = processing(op, mid, key)
                    machine_ready[mid] += yield from interrupted_duration("machine_failure", duration, (mid, *key))
                completed[key] = machine_ready[mid]
                job_ready[key[0]] = machine_ready[mid]
                job_pos[key[0]] = machine["location"]
        rank = {tuple(key): index for mid, machine in machines.items() for index, key in enumerate(plan.queues.get(mid, machine["input_queue"]))}
        next_check = min([float(self.decision_interval), *[value for value in machine_ready.values() if value > 0],
                          *[value for value in agent_ready.values() if value > 0]])
        order = sorted(jobs, key=lambda jid: (min((rank.get((jid, op["op_id"]), 10**6) for op in jobs[jid]["ops"]), default=10**6), jid))
        for jid in order:
            job = jobs[jid]
            if job["delivered"]:
                continue
            if jid in shipping:
                job_ready[jid] = shipping[jid]
                continue
            for op in job["ops"]:
                yield None
                key = (jid, op["op_id"])
                if op["status"] == "FINISHED" or key in completed:
                    continue
                source = job_pos[jid]
                fixed = plan.dispatch.get(key, None if key in plan.cancel else op["assigned_machine"])
                options = [mid for mid, _ in op["machine_options_with_time"] if fixed is None or mid == fixed]
                choices: list = []
                for mid in options:
                    target = tuple(machines[mid]["location"])
                    aid = None
                    arrival = job_ready[jid]
                    if op["status"] == "PENDING" and op["arrive_machine_at"] < 0 and key not in delivery and key not in completed:
                        # STOP really defers the first uncommitted operation;
                        # it is not evaluated as if it were dispatched now.
                        arrival = max(arrival, next_check)
                        if key in plan.dispatch and source is not None and tuple(source) == target:
                            arrival = job_ready[jid]
                    if key in delivery:
                        arrival = max(arrival, delivery[key])
                    elif source is not None and tuple(source) != target:
                        if not agent_ready:
                            return math.inf
                        aid = min(agent_ready, key=lambda a: max(agent_ready[a] + self.router.distance(agent_pos[a], source), arrival))
                        transit = scene["pickup_steps"] + self.router.distance(source, target) + scene["dropoff_steps"]
                        arrival = max(agent_ready[aid] + self.router.distance(agent_pos[aid], source), arrival) + (yield from interrupted_duration("agv_failure", transit, (aid, *key)))
                    own_slot = ("job", jid)
                    slot = own_slot if own_slot in buffer_release[mid] else min(buffer_release[mid], key=buffer_release[mid].get)
                    available = 0.0 if slot == own_slot else buffer_release[mid][slot]
                    start = max(arrival, machine_ready[mid], available)
                    duration = processing(op, mid, key)
                    duration = yield from interrupted_duration("machine_failure", duration, (mid, *key))
                    choices.append((start + duration, mid, aid, arrival, slot))
                finish, mid, aid, arrival, slot = min(choices)
                machine_ready[mid] = job_ready[jid] = finish
                job_pos[jid] = machines[mid]["location"]
                del buffer_release[mid][slot]
                buffer_release[mid][("output", jid, op["op_id"])] = finish + scene["pickup_steps"]
                if aid is not None:
                    agent_ready[aid], agent_pos[aid] = arrival, tuple(job_pos[jid])
            if job_pos[jid] is not None:
                if not agent_ready:
                    return math.inf
                source, target = job_pos[jid], job["finished_goods_destination"]
                aid = min(agent_ready, key=lambda a: max(agent_ready[a] + self.router.distance(agent_pos[a], source), job_ready[jid]))
                transit = scene["pickup_steps"] + self.router.distance(source, target) + scene["dropoff_steps"]
                arrival = max(agent_ready[aid] + self.router.distance(agent_pos[aid], source), job_ready[jid]) + (yield from interrupted_duration("agv_failure", transit, (aid, jid, "shipping")))
                agent_ready[aid], agent_pos[aid], job_ready[jid] = arrival, tuple(target), arrival
        return max(job_ready.values(), default=0.0) + len(plan.traffic.blocked) * self.router.horizon

    def build_steps(self, scene: dict, affected: dict, ranker=None):
        self.router.configure(scene)
        keep = Candidate(0)
        yield from self._evaluate_steps(scene, keep, 0)
        candidates: list[Candidate] = [keep]
        for index in range(self.limit - 1):
            yield None
            scope = index % 5
            if scope >= 2 and not affected.get("allow_reschedule", True):
                continue
            parent = None
            if index >= 10:
                parent = candidates[(yield from ranker(candidates))] if ranker is not None else min(candidates, key=lambda c: c.cost)
            candidate = self._construct(scene, scope, affected, index, parent)
            yield from self._evaluate_steps(scene, candidate, index)
            if candidate.feasible:
                candidates.append(candidate)
            if (index == 2 and not scene["tasks"]) or index % 6 == 4:
                yield self._accepted(candidates, affected["hard"])
        # Explicit STOP/partial commitment alternatives, not forced maximal dispatch.
        best = min(candidates, key=lambda c: c.cost)
        if best.dispatch:
            partial = deepcopy(best)
            key = list(partial.dispatch)[-1]
            del partial.dispatch[key]
            partial.assignments = {aid: t for aid, t in partial.assignments.items() if (t["job_id"], t["op_id"]) != key}
            yield from self._evaluate_steps(scene, partial, 0)
            if partial.feasible:
                candidates.append(partial)
        yield self._accepted(candidates, affected["hard"])

    def _accepted(self, candidates: list[Candidate], hard: bool) -> list[Candidate]:
        # Modelled completion time is an input to the actor, not an oracle that
        # vetoes its exploration. Deduplicate equivalent physical transactions.
        unique: dict[tuple, Candidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.identity(), candidate)
        return list(unique.values())
