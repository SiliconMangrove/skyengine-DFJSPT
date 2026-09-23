"""Safety properties at the boundary between suspended planning and execution."""
from copy import deepcopy
import time
import unittest

from dfjsp_t_rl.realtime import PlanningJob, applicable, route_moves
from dfjsp_t_rl.search import Candidate
from dfjsp_t_rl.traffic import RollingTrafficPlanner, TrafficPlan


class RealtimeSafetyTest(unittest.TestCase):
    def test_suspended_route_is_jointly_collision_free(self):
        scene = {"time": 0, "grid": [[0] * 4 for _ in range(4)],
                 "moves": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
                 "machines": [], "jobs": [], "pickup_steps": 1, "dropoff_steps": 1,
                 "agents": []}
        for aid, start, target in ((0, (1, 0), (1, 3)), (1, (1, 3), (1, 0))):
            task = {"task_id": aid, "source": start, "destination": target}
            scene["agents"].append({"id": aid, "pos": start, "status": "OK", "loaded": True,
                                    "current_task": task, "task_phase": "TO_DROPOFF", "handling_remaining": 0})
        router = RollingTrafficPlanner(horizon=8)
        job = PlanningJob(router.plan_steps(scene, {}, set()), scene, route=True)
        job.advance(0)
        self.assertIsNone(job.result)
        job.advance(time.perf_counter() + 2)
        self.assertTrue(job.done)
        paths = job.result.paths
        for tick in range(9):
            self.assertEqual(len({path[tick] for path in paths.values()}), len(paths))
            if tick:
                self.assertFalse(paths[0][tick] == paths[1][tick - 1] and paths[1][tick] == paths[0][tick - 1])
        current = deepcopy(scene)
        current["time"] = 1
        for agent in current["agents"]:
            agent["pos"] = paths[agent["id"]][1]
        self.assertIsNotNone(route_moves(job.result, scene, current))
        current["agents"][0]["status"] = "DOWN"
        self.assertIsNone(route_moves(job.result, scene, current))

    def test_stale_dispatch_is_not_committed_after_work_started(self):
        scene = {"jobs": [{"job_id": 0, "material_location": [0, 0], "carrier_id": None,
                           "ops": [{"op_id": 0, "status": "PENDING", "assigned_machine": None,
                                    "machine_options_with_time": [(0, 10)]}]}],
                 "machines": [{"id": 0, "status": "OK", "current_op": None, "input_queue": [],
                               "location": [1, 1], "buffer_capacity": 1, "buffer_jobs": [], "suspended_ops": []}],
                 "agents": [], "tasks": []}
        plan = Candidate(0, dispatch={(0, 0): 0})
        self.assertTrue(applicable(plan, scene, scene))
        current = deepcopy(scene)
        current["jobs"][0]["ops"][0].update(status="PROCESSING", assigned_machine=0)
        self.assertFalse(applicable(plan, scene, current))
        current = deepcopy(scene)
        current["machines"][0]["status"] = "DOWN"
        self.assertFalse(applicable(plan, scene, current))

        # A still-unstarted plan also becomes invalid if another commitment
        # consumed the destination capacity while its search was suspended.
        current = deepcopy(scene)
        current["jobs"].append({"job_id": 1, "ops": [], "material_location": [1, 1]})
        current["machines"][0]["buffer_jobs"] = [1]
        self.assertFalse(applicable(plan, scene, current))

    def test_cached_move_cannot_bypass_handling_freeze(self):
        task = {"task_id": 1, "source": [0, 0], "destination": [0, 1]}
        before = {"time": 0, "grid": [[0, 0]], "moves": [(0, 0), (0, 1), (0, -1)],
                  "agents": [{"id": 0, "pos": [0, 0], "current_task": task, "status": "OK", "task_phase": "TO_DROPOFF"}]}
        route = TrafficPlan({0: [(0, 0), (0, 1)]}, {0: 1}, 0, 0, [], [1])
        now = deepcopy(before)
        now["agents"][0]["task_phase"] = "DROPPING"
        self.assertIsNone(route_moves(route, before, now))


if __name__ == "__main__":
    unittest.main()
