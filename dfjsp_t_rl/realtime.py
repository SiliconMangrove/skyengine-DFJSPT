"""Cooperative planning jobs and validation at the live simulator boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import time

from .search import Candidate, task_id
from .traffic import TrafficPlan
from experiment.algorithm_platform_plugins.dfjsp_t.buffer_safety import safe_dispatches


@dataclass
class PlanningJob:
    iterator: Iterator
    scene: dict
    result: object = None
    done: bool = False
    slices: int = 0
    route: bool = False
    revision: int = 0
    consumed: int = 0

    def advance(self, deadline: float) -> None:
        self.slices += 1
        while not self.done and time.perf_counter() < deadline:
            # StopIteration is the coroutine completion protocol, not recovery.
            try:
                result = next(self.iterator)
            except StopIteration:
                self.done = True
                break
            if isinstance(result, TrafficPlan if self.route else list):
                self.result = result
                self.revision += 1


def applicable(plan: Candidate, before: dict, now: dict) -> bool:
    """Validate commitments, not elapsed time or ordinary vehicle movement."""
    jobs = {job["job_id"]: job for job in now["jobs"]}
    old_jobs = {job["job_id"]: job for job in before["jobs"]}
    machines = {machine["id"]: machine for machine in now["machines"]}
    agents = {agent["id"]: agent for agent in now["agents"]}
    old_agents = {agent["id"]: agent for agent in before["agents"]}
    operations = {(job["job_id"], op["op_id"]): op for job in now["jobs"] for op in job["ops"]}
    old_ops = {(job["job_id"], op["op_id"]): op for job in before["jobs"] for op in job["ops"]}
    loaded = {agent["current_task"]["job_id"] for agent in agents.values() if agent["loaded"]}
    held = {machine["current_op"][0] for machine in machines.values() if machine["current_op"] is not None}
    for key in plan.cancel | plan.dispatch.keys():
        op, job = operations[key], jobs[key[0]]
        if op["status"] != "PENDING" or op["assigned_machine"] != old_ops[key]["assigned_machine"] or key[0] in loaded:
            return False
        if job["material_location"] != old_jobs[key[0]]["material_location"]:
            return False
        if key in plan.dispatch and (key[0] in held or job["carrier_id"] is not None
                                    or machines[plan.dispatch[key]]["status"] != "OK"):
            return False
    for aid in plan.release:
        agent, previous = agents[aid], old_agents[aid]
        if agent["loaded"] or agent["current_task"] is None or previous["current_task"] is None:
            return False
        if agent["current_task"]["task_id"] != previous["current_task"]["task_id"]:
            return False
    pending = {task["task_id"]: task for task in now["tasks"]}
    for aid, task in plan.assignments.items():
        agent, job = agents[aid], jobs[task["job_id"]]
        if agent["status"] != "OK" or (agent["current_task"] is not None and aid not in plan.release):
            return False
        if job["carrier_id"] is not None or job["material_location"] != task["source"]:
            return False
        key = (task["job_id"], task["op_id"])
        if key not in plan.dispatch:
            current = pending.get(task["task_id"])
            if current is None or (current["assigned_agent_id"] is not None and current["assigned_agent_id"] not in plan.release):
                return False
            if current["source"] != task["source"] or current["destination"] != task["destination"]:
                return False
    for mid, keys in plan.queues.items():
        if set(map(tuple, keys)) != {tuple(key) for key in machines[mid]["input_queue"] if tuple(key) not in plan.cancel}:
            return False
    return safe_dispatches(now, plan.dispatch, plan.cancel)


def materialize(plan: Candidate, scene: dict) -> None:
    """Allocate fresh task IDs only when an executable plan is committed."""
    for task in plan.assignments.values():
        key = (task["job_id"], task["op_id"])
        if key in plan.dispatch:
            task["task_id"] = task_id(scene, key)


def route_moves(route: TrafficPlan, before: dict, now: dict) -> list[int] | None:
    """Reuse a reservation suffix only when all vehicles still match it."""
    offset = int(now["time"] - before["time"])
    if before["grid"] != now["grid"] or len(before["agents"]) != len(now["agents"]):
        return None
    old = {agent["id"]: agent for agent in before["agents"]}
    deltas = [tuple(move) for move in now["moves"]]
    moves: list[int] = []
    for agent in now["agents"]:
        aid = agent["id"]
        path = route.paths[aid]
        if offset < 0 or offset + 1 >= len(path) or path[offset] != tuple(agent["pos"]):
            return None
        previous_task, task = old[aid]["current_task"], agent["current_task"]
        previous_key = None if previous_task is None else (previous_task["task_id"], previous_task["source"], previous_task["destination"])
        key = None if task is None else (task["task_id"], task["source"], task["destination"])
        if previous_key != key or old[aid]["status"] != agent["status"]:
            return None
        move = deltas.index((path[offset + 1][0] - path[offset][0], path[offset + 1][1] - path[offset][1]))
        if move and (agent["status"] != "OK" or agent["task_phase"] in {"PICKING", "DROPPING"}):
            return None
        moves.append(move)
    return moves
