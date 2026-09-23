"""Rolling space-time routing with vertex/edge reservations and handling dwell."""
from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count
from dataclasses import dataclass
import math


@dataclass
class TrafficPlan:
    paths: dict[int, list[tuple[int, int]]]
    arrivals: dict[int, float]
    waits: float
    unfinished: int
    blocked: list[int]
    moves: list[int]
    solver: str = "prioritized"


class RollingTrafficPlanner:
    def __init__(self, horizon: int = 24, order_trials: int = 3):
        self.horizon, self.order_trials = int(horizon), int(order_trials)
        self._grid: tuple = ()
        self.distances: dict[tuple[int, int], dict] = {}

    def configure(self, scene: dict) -> None:
        grid = tuple(tuple(row) for row in scene["grid"])
        if grid != self._grid:
            self._grid = grid
            self.distances.clear()
            self.free = {(x, y) for x, row in enumerate(grid) for y, value in enumerate(row) if value == 0}
        self.moves = [tuple(move) for move in scene["moves"]]

    def distance_map(self, target: tuple) -> dict:
        target = tuple(target)
        if target not in self.distances:
            if target not in self.free:
                self.distances[target] = {}
                return self.distances[target]
            distances: dict = {target: 0}
            queue = deque([target])
            while queue:
                cell = queue.popleft()
                for dx, dy in self.moves[1:]:
                    other = (cell[0] + dx, cell[1] + dy)
                    if other in self.free and other not in distances:
                        distances[other] = distances[cell] + 1
                        queue.append(other)
            self.distances[target] = distances
        return self.distances[target]

    def distance(self, source, target) -> float:
        return self.distance_map(tuple(target)).get(tuple(source), math.inf)

    def _leg_steps(self, start: tuple, target: tuple, start_time: int, dwell: int,
             vertices: set, edges: set, earliest: float = 0) -> tuple[list[tuple], bool]:
        distances = self.distance_map(target)
        queue: list = [(distances.get(start, math.inf), 0, start_time, start)]
        parent: dict = {(start_time, start): None}
        best = None
        best_key = (distances.get(start, math.inf), start_time)
        found = False
        expanded: int = 0
        while queue:
            expanded += 1
            if expanded % 32 == 0:
                yield None
            _, cost, time, cell = heappop(queue)
            # Every partial endpoint must remain safe for the rest of the window.
            hold = all((future, cell) not in vertices for future in range(time + 1, self.horizon + 1))
            key = (distances.get(cell, math.inf), time)
            if hold and (best is None or key < best_key):
                best, best_key = (time, cell), key
            if cell == target and hold and time >= earliest and time + dwell <= self.horizon:
                best, found = (time, cell), True
                break
            if time == self.horizon:
                continue
            for dx, dy in self.moves:
                nxt = (cell[0] + dx, cell[1] + dy)
                state = (time + 1, nxt)
                if nxt not in self.free or state in parent or state in vertices or (time + 1, nxt, cell) in edges:
                    continue
                parent[state] = (time, cell)
                heappush(queue, (cost + 1 + distances.get(nxt, math.inf), cost + 1, time + 1, nxt))
        path: list[tuple] = []
        current = best
        while current is not None:
            path.append(current[1])
            current = parent[current]
        path.reverse()
        if found:
            path.extend([target] * dwell)
        return path, found

    def plan_steps(self, scene: dict, assignments: dict[int, dict], releases: set[int],
             priority: list[int] | None = None, evacuation: bool = False) -> TrafficPlan:
        self.configure(scene)
        yield None
        agents = {item["id"]: item for item in scene["agents"]}
        goals: dict[int, list[tuple[tuple, int]]] = {}
        service = {tuple(m["location"]) for m in scene["machines"]}
        service.update(tuple(j[name]) for j in scene["jobs"] for name in ("raw_material_source", "finished_goods_destination"))
        occupied = {tuple(a["pos"]) for a in agents.values()}
        parking: set = set()
        # Predict buffer slots from currently observable outbound pickups and
        # queued operation starts. Incoming deliveries reserve a slot in turn.
        tasks = {aid: assignments.get(aid, None if aid in releases else a["current_task"]) for aid, a in agents.items()}
        buffer_slots: dict[tuple, list[float]] = {}
        operation_by_key = {(j["job_id"], op["op_id"]): op for j in scene["jobs"] for op in j["ops"]}
        for machine in scene["machines"]:
            location = tuple(machine["location"])
            slots = [0.0] * (machine["buffer_capacity"] - len(machine["buffer_jobs"]))
            for jid in machine["buffer_jobs"]:
                release_time: float = self.horizon + 1
                for aid, task in tasks.items():
                    if task is not None and task["job_id"] == jid and not agents[aid]["loaded"] and agents[aid]["status"] == "OK":
                        release_time = min(release_time, self.distance(agents[aid]["pos"], location) + scene["pickup_steps"] + int(aid in assignments))
                if machine["status"] == "OK" and any(key[0] == jid for key in machine["input_queue"]):
                    current = machine["current_op"]
                    delay = 0.0 if current is None else operation_by_key[tuple(current)]["remaining_proc_time"] + 1
                    release_time = min(release_time, delay)
                slots.append(release_time)
            buffer_slots[location] = slots
        earliest_dropoff: dict[int, float] = {}
        frozen: dict[int, int] = {}
        for aid, agent in agents.items():
            task = assignments.get(aid, None if aid in releases else agent["current_task"])
            if task is not None and tuple(task["destination"]) in buffer_slots:
                slots = buffer_slots[tuple(task["destination"])]
                availability = min(slots)
                earliest_dropoff[aid] = availability
                slots.remove(availability)
                slots.append(float(self.horizon + 1))
            phase = "TO_PICKUP" if aid in assignments else "IDLE" if aid in releases else agent["task_phase"]
            frozen[aid] = min(self.horizon, 1 if aid in assignments else max(1, agent["handling_remaining"]) if phase in {"PICKING", "DROPPING"} else 0)
            if agent["status"] != "OK":
                goals[aid] = []
            elif task is None:
                start = tuple(agent["pos"])
                choices = self.free - service - occupied - parking
                if choices and (start in service or evacuation):
                    target = min(choices, key=lambda cell: (self.distance_map(start).get(cell, math.inf), cell))
                    parking.add(target)
                    goals[aid] = [(target, 0)]
                else:
                    goals[aid] = []
            elif phase == "DROPPING":
                goals[aid] = [(tuple(task["destination"]), max(0, agent["handling_remaining"] - frozen[aid]))]
            elif agent["loaded"]:
                goals[aid] = [(tuple(task["destination"]), scene["dropoff_steps"])]
            else:
                pickup = max(0, agent["handling_remaining"] - frozen[aid]) if phase == "PICKING" else scene["pickup_steps"]
                goals[aid] = [(tuple(task["source"]), pickup), (tuple(task["destination"]), scene["dropoff_steps"])]
        base_order = priority or sorted(agents, key=lambda aid: (not agents[aid]["loaded"], agents[aid]["current_task"] is None, aid))
        if priority is None and base_order:
            shift = (scene["time"] // self.horizon) % len(base_order)
            base_order = base_order[shift:] + base_order[:shift]
        best: TrafficPlan | None = None
        for trial in range(self.order_trials):
            order = base_order[trial % len(base_order):] + base_order[:trial % len(base_order)] if base_order else []
            vertices: set = {(t, tuple(a["pos"])) for a in agents.values() for t in range(1, self.horizon + 1)}
            edges: set = set()
            paths: dict = {}
            arrivals: dict = {}
            blocked: list[int] = []
            unfinished, waits = 0, 0.0
            for aid in order:
                yield None
                start = tuple(agents[aid]["pos"])
                vertices.difference_update((t, start) for t in range(1, self.horizon + 1))
                # The formal environment freezes newly assigned AGVs for one slot.
                path = [start] * (frozen[aid] + 1)
                completed = True
                for leg_index, (target, dwell) in enumerate(goals[aid]):
                    earliest = earliest_dropoff.get(aid, 0) if leg_index == len(goals[aid]) - 1 else 0
                    segment, reached = yield from self._leg_steps(path[-1], target, len(path) - 1, dwell, vertices, edges, earliest)
                    if not segment:
                        path = [start] * (frozen[aid] + 1)
                        completed = False
                        arrivals[aid] = math.inf
                        break
                    path.extend(segment[1:])
                    if not reached:
                        completed = False
                        remainder = self.distance(path[-1], target) + dwell
                        for next_target, next_dwell in goals[aid][leg_index + 1:]:
                            remainder += self.distance(target, next_target) + next_dwell
                            target = next_target
                        arrivals[aid] = max(len(path) - 1 + remainder, earliest + dwell)
                        break
                if (not completed and evacuation and len(set(path)) == 1
                        and agents[aid]["status"] == "OK"
                        and agents[aid]["task_phase"] not in {"PICKING", "DROPPING"}):
                    # A loaded vehicle may retreat to an empty passing cell;
                    # its workpiece and destination stay committed.
                    choices = self.free - service - occupied - parking
                    for target in sorted(choices, key=lambda cell: (self.distance_map(start).get(cell, math.inf), cell)):
                        initial_time = frozen[aid]
                        segment, reached = yield from self._leg_steps(start, target, initial_time, 0, vertices, edges)
                        if reached and len(segment) > 1:
                            path = [start] * initial_time + segment
                            parking.add(target)
                            arrivals[aid] = len(path) - 1 + self.distance(target, goals[aid][-1][0]) + sum(dwell for _, dwell in goals[aid])
                            break
                if completed:
                    arrivals[aid] = len(path) - 1
                elif goals[aid]:
                    unfinished += 1
                    if len(set(path)) == 1:
                        blocked.append(aid)
                path.extend([path[-1]] * (self.horizon + 1 - len(path)))
                paths[aid] = path
                vertices.update((t, cell) for t, cell in enumerate(path) if t)
                edges.update((t, path[t - 1], path[t]) for t in range(1, len(path)))
                if goals[aid]:
                    waits += sum(a == b for a, b in zip(path, path[1:]))
            moves = [self.moves.index((paths[a["id"]][1][0] - a["pos"][0], paths[a["id"]][1][1] - a["pos"][1])) for a in scene["agents"]]
            result = TrafficPlan(paths, arrivals, waits, unfinished, blocked, moves)
            key = (len(blocked), sum(arrivals.values()), waits)
            if best is None or key < (len(best.blocked), sum(best.arrivals.values()), best.waits):
                best = result
        if best.blocked and evacuation:
            coordinated = yield from self._conflict_search_steps(scene, goals, earliest_dropoff, frozen)
            if coordinated is not None and (len(coordinated.blocked), sum(coordinated.arrivals.values())) < (len(best.blocked), sum(best.arrivals.values())):
                best = coordinated
        yield best
        return best

    def plan(self, scene: dict, assignments: dict[int, dict], releases: set[int],
             priority: list[int] | None = None, evacuation: bool = False) -> TrafficPlan:
        result = None
        for result in self.plan_steps(scene, assignments, releases, priority, evacuation):
            pass
        return result

    def _conflict_search_steps(self, scene: dict, goals: dict, earliest_dropoff: dict, frozen: dict) -> TrafficPlan | None:
        """Bounded window CBS repair permits coordinated cycles and corridor swaps.

        Failure to find a repair within 128 expansions leaves the already-safe
        reservation plan in place; the caller receives its blocked-agent list.
        """
        agents = {a["id"]: a for a in scene["agents"]}
        def route(aid: int, constraints: dict) -> tuple | None:
            vertices, edges = constraints.get(aid, (set(), set()))
            start = tuple(agents[aid]["pos"])
            path = [start] * (frozen[aid] + 1)
            if any((tick, start) in vertices for tick in range(1, frozen[aid] + 1)):
                return None
            arrival = 0.0
            completed = True
            for index, (target, dwell) in enumerate(goals[aid]):
                earliest = earliest_dropoff.get(aid, 0) if index == len(goals[aid]) - 1 else 0
                segment, reached = yield from self._leg_steps(path[-1], target, len(path) - 1, dwell, vertices, edges, earliest)
                if not segment:
                    return None
                path.extend(segment[1:])
                arrival = len(path) - 1
                if not reached:
                    completed = False
                    arrival = max(arrival + self.distance(path[-1], target) + dwell, earliest + dwell)
                    for other, remaining_dwell in goals[aid][index + 1:]:
                        arrival += self.distance(target, other) + remaining_dwell
                        target = other
                    break
            path.extend([path[-1]] * (self.horizon + 1 - len(path)))
            if any((t, cell) in vertices or (t, cell, path[t - 1]) in edges for t, cell in enumerate(path) if t):
                return None
            return path, arrival, completed

        initial: dict = {}
        for aid in agents:
            initial[aid] = yield from route(aid, {})
            if initial[aid] is None:
                return None
        serial = count()
        queue: list = [(sum(value[1] for value in initial.values()), next(serial), {}, initial)]
        for _ in range(128):
            yield None
            if not queue:
                return None
            _, _, constraints, paths = heappop(queue)
            conflict = None
            for tick in range(1, self.horizon + 1):
                vertices, edges = {}, {}
                for aid, (path, _, _) in paths.items():
                    cell, previous = path[tick], path[tick - 1]
                    if cell in vertices:
                        conflict = (aid, vertices[cell], tick, "vertex", cell)
                        break
                    if cell != previous and (cell, previous) in edges:
                        conflict = (aid, edges[(cell, previous)], tick, "edge", None)
                        break
                    vertices[cell], edges[(previous, cell)] = aid, aid
                if conflict is not None:
                    break
            if conflict is None:
                blocked = [aid for aid, (path, _, done) in paths.items() if not done and len(set(path)) == 1]
                moves = [self.moves.index((paths[a["id"]][0][1][0] - a["pos"][0], paths[a["id"]][0][1][1] - a["pos"][1])) for a in scene["agents"]]
                return TrafficPlan({aid: value[0] for aid, value in paths.items()}, {aid: value[1] for aid, value in paths.items()},
                                   float(sum(sum(a == b for a, b in zip(value[0], value[0][1:])) for value in paths.values())),
                                   sum(not value[2] for value in paths.values()), blocked, moves, "window_cbs")
            left, right, tick, kind, cell = conflict
            for aid in (left, right):
                child = {key: (set(value[0]), set(value[1])) for key, value in constraints.items()}
                vertices, edges = child.setdefault(aid, (set(), set()))
                if kind == "vertex":
                    vertices.add((tick, cell))
                else:
                    path = paths[aid][0]
                    # _leg checks reverse edge reservations.
                    edges.add((tick, path[tick], path[tick - 1]))
                replacement = yield from route(aid, child)
                if replacement is not None:
                    updated = {**paths, aid: replacement}
                    heappush(queue, (sum(value[1] for value in updated.values()), next(serial), child, updated))
        return None
