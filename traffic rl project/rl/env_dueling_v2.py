import traci
import os
import numpy as np
import random


class TrafficSignalEnvDuelingV2:
    def __init__(
        self,
        sumo_binary="sumo",
        step_length=1.0,
        auto_close=True,
        scenario="baseline",
        seed=0
    ):
        self.sumo_binary = sumo_binary
        self.step_length = step_length
        self.auto_close = auto_close
        self.scenario = scenario
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir

        # FIX 1 — absolute path for SUMO config
        self.sumo_config = os.path.abspath(
            os.path.join(base_dir, "..", "sumo", "config.sumocfg")
        )

        # FIX 2 — removed extra "sumo/"
        self.ROUTE_MAP = {
            "baseline": "routes_baseline.rou.xml",
            "low_demand": "routes_low_demand.rou.xml",
            "high_demand": "routes_high_demand.rou.xml",
            "emergency_stress": "routes_emergency_stress.rou.xml",
            "unseen": "routes_unseen.rou.xml",
            "emergency_low": "routes_emergency_low.rou.xml",
            "emergency_medium": "routes_emergency_medium.rou.xml",
            "emergency_high": "routes_emergency_high.rou.xml"
        }

        self.tls_id = None
        self.last_phase = None

    # =========================
    # Environment
    # =========================
    def reset(self):
        if traci.isLoaded():
            traci.close()

        route_file = self.ROUTE_MAP[self.scenario]

        # FIX 3 — absolute path for route file
        route_path = os.path.abspath(
            os.path.join(self.base_dir, "..", "sumo", route_file)
        )

        traci.start([
            self.sumo_binary,
            "-c", self.sumo_config,
            "--route-files", route_path,
            "--seed", str(self.seed),
            "--no-step-log", "true"
        ])

        self.tls_id = traci.trafficlight.getIDList()[0]
        self.last_phase = traci.trafficlight.getPhase(self.tls_id)

        traci.simulationStep()
        return self._get_state()

    def step(self, action):
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        if action == 1:
            next_phase = (current_phase + 1) % len(
                traci.trafficlight
                .getCompleteRedYellowGreenDefinition(self.tls_id)[0]
                .phases
            )
            traci.trafficlight.setPhase(self.tls_id, next_phase)

        elif action == 2:
            traci.trafficlight.setPhaseDuration(self.tls_id, 15)

        traci.simulationStep()

        reward = self._compute_reward(current_phase)
        next_state = self._get_state()

        done = traci.simulation.getMinExpectedNumber() == 0

        self.last_phase = current_phase

        if done and self.auto_close:
            traci.close()

        return next_state, reward, done, {}

    # =========================
    # State
    # =========================
    def _get_state(self):
        q_ns = (
            traci.lane.getLastStepHaltingNumber("n_in_0")
            + traci.lane.getLastStepHaltingNumber("s_in_0")
        )
        q_ew = (
            traci.lane.getLastStepHaltingNumber("e_in_0")
            + traci.lane.getLastStepHaltingNumber("w_in_0")
        )

        w_ns, w_ew = 0.0, 0.0
        c_ns, c_ew = 0, 0

        emergency_present = 0
        emergency_dist = 1000.0

        for vid in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(vid)
            edge = traci.vehicle.getRoadID(vid)
            vtype = traci.vehicle.getTypeID(vid)

            if edge in ["n_in", "s_in"]:
                w_ns += wt
                c_ns += 1
            elif edge in ["e_in", "w_in"]:
                w_ew += wt
                c_ew += 1

            if vtype == "emergency":
                emergency_present = 1
                emergency_dist = traci.vehicle.getLanePosition(vid)

        w_ns = w_ns / c_ns if c_ns else 0.0
        w_ew = w_ew / c_ew if c_ew else 0.0

        phase = traci.trafficlight.getPhase(self.tls_id)

        return np.array([
            q_ns, q_ew,
            w_ns, w_ew,
            emergency_present,
            emergency_dist,
            phase
        ], dtype=np.float32)

    # =========================
    # Reward
    # =========================
    def _compute_reward(self, prev_phase):
        emergency_wait = []
        normal_wait = []
        queue = 0

        for vid in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(vid)
            if traci.vehicle.getTypeID(vid) == "emergency":
                emergency_wait.append(wt)
            else:
                normal_wait.append(wt)

        avg_emergency = np.mean(emergency_wait) if emergency_wait else 0.0
        avg_normal = np.mean(normal_wait) if normal_wait else 0.0

        for lane in traci.lane.getIDList():
            queue += traci.lane.getLastStepHaltingNumber(lane)

        switch_penalty = 1.0 if prev_phase != self.last_phase else 0.0

        norm_emergency = min(avg_emergency / 30.0, 1.0)
        norm_normal = min(avg_normal / 20.0, 1.0)
        norm_queue = min(queue / 40.0, 1.0)

        reward = (
            - 8.0 * norm_emergency
            - 2.0 * norm_normal
            - 1.5 * norm_queue
            - 0.3 * switch_penalty
        )

        return reward