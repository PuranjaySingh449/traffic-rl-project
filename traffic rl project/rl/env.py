import traci
import os
import numpy as np
import random


class TrafficSignalEnv:
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

        self.sumo_config = os.path.join(base_dir, "..", "sumo", "config.sumocfg")

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

        # reward weights
        self.alpha = 10.0
        self.beta = 1.0
        self.gamma = 0.5
        self.delta = 1.0

    # -----------------------------
    # Environment control
    # -----------------------------
    def reset(self):

        if traci.isLoaded():
            traci.close()

        route_file = self.ROUTE_MAP.get(
            self.scenario,
            self.ROUTE_MAP["baseline"]
        )

        route_path = os.path.join(self.base_dir, "..", "sumo", route_file)

        traci.start([
            self.sumo_binary,
            "-c", self.sumo_config,
            "--route-files", route_path,
            "--seed", str(self.seed),
            "--random",
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
        info = {}

        self.last_phase = current_phase

        if done and self.auto_close:
            traci.close()

        return next_state, reward, done, info

    # -----------------------------
    # State representation
    # -----------------------------
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
        count_ns, count_ew = 0, 0

        emergency_present = 0
        emergency_distance = 1e3

        for vid in traci.vehicle.getIDList():

            edge = traci.vehicle.getRoadID(vid)
            vtype = traci.vehicle.getTypeID(vid)

            if edge in ["n_in", "s_in"]:
                w_ns += traci.vehicle.getWaitingTime(vid)
                count_ns += 1

            elif edge in ["e_in", "w_in"]:
                w_ew += traci.vehicle.getWaitingTime(vid)
                count_ew += 1

            if vtype == "emergency":
                emergency_present = 1
                emergency_distance = traci.vehicle.getLanePosition(vid)

        w_ns = w_ns / count_ns if count_ns > 0 else 0.0
        w_ew = w_ew / count_ew if count_ew > 0 else 0.0

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        return np.array([
            q_ns,
            q_ew,
            w_ns,
            w_ew,
            emergency_present,
            emergency_distance,
            current_phase
        ], dtype=np.float32)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def _get_emergency_wait(self):

        wait = 0.0

        for vid in traci.vehicle.getIDList():

            if traci.vehicle.getTypeID(vid) == "emergency":
                wait += traci.vehicle.getWaitingTime(vid)

        return wait

    def _get_queue_length(self):

        q = 0.0

        for lane in traci.lane.getIDList():
            q += traci.lane.getLastStepHaltingNumber(lane)

        return q

    # -----------------------------
    # Reward function
    # -----------------------------
    def _compute_reward(self, prev_phase):

        emergency_delay = self._get_emergency_wait()
        queue_length = self._get_queue_length()

        normal_delay = 0.0

        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(vid) != "emergency":
                normal_delay += traci.vehicle.getWaitingTime(vid)

        congestion = min(queue_length / 50.0, 1.0)

        switch_penalty = 1.0 if prev_phase != self.last_phase else 0.0

        reward = (
            - self.alpha * emergency_delay * (1.0 - 0.5 * congestion)
            - self.beta * normal_delay
            - self.gamma * queue_length
            - self.delta * switch_penalty
        )

        reward = reward / 100.0
        reward = np.clip(reward, -20.0, 5.0)

        return reward