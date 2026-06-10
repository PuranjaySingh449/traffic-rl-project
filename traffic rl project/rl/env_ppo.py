import traci
import os
import numpy as np


class TrafficSignalEnvPPO:

    ROUTE_MAP = {
        "baseline": "routes_baseline.rou.xml",
        "low_demand": "routes_low_demand.rou.xml",
        "high_demand": "routes_high_demand.rou.xml",
        "emergency_stress": "routes_emergency_stress.rou.xml",
        "unseen": "routes_unseen.rou.xml",

        "emergency_low": "routes_emergency_low.rou.xml",
        "emergency_medium": "routes_emergency_medium.rou.xml",
        "emergency_high": "routes_emergency_high.rou.xml"
    }

    def __init__(self, sumo_binary="sumo", step_length=1.0, auto_close=True):

        self.sumo_binary = sumo_binary
        self.step_length = step_length
        self.auto_close = auto_close

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_dir = os.path.join(base_dir, "..", "sumo")

        self.net_file = os.path.join(self.sumo_dir, "network.net.xml")

        self.tls_id = None
        self.last_phase = None
        self.scenario = "baseline"

        # reward weights
        self.w_emergency = 5.0
        self.w_normal = 1.0
        self.w_queue = 0.5
        self.w_switch = 0.2
        self.w_throughput = 0.1
        self.w_emergency_clear = 5.0

    # -----------------------------
    # Environment control
    # -----------------------------
    def reset(self, scenario="baseline"):

        self.scenario = scenario

        if traci.isLoaded():
            traci.close()

        route_file = os.path.join(
            self.sumo_dir,
            self.ROUTE_MAP[self.scenario]
        )

        traci.start([
            self.sumo_binary,
            "-n", self.net_file,
            "-r", route_file,
            "--no-warnings",
            "--no-step-log"
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

    # -----------------------------
    # State
    # -----------------------------
    def _get_state(self):

        q_ns = (
            traci.lane.getLastStepHaltingNumber("n_in_0") +
            traci.lane.getLastStepHaltingNumber("s_in_0")
        )

        q_ew = (
            traci.lane.getLastStepHaltingNumber("e_in_0") +
            traci.lane.getLastStepHaltingNumber("w_in_0")
        )

        w_ns, w_ew = 0.0, 0.0
        c_ns, c_ew = 0, 0

        emergency_present = 0
        emergency_distance = 1000

        for vid in traci.vehicle.getIDList():

            edge = traci.vehicle.getRoadID(vid)
            vtype = traci.vehicle.getTypeID(vid)

            if edge in ["n_in", "s_in"]:
                w_ns += traci.vehicle.getWaitingTime(vid)
                c_ns += 1

            elif edge in ["e_in", "w_in"]:
                w_ew += traci.vehicle.getWaitingTime(vid)
                c_ew += 1

            if vtype == "emergency":
                emergency_present = 1
                emergency_distance = traci.vehicle.getLanePosition(vid)

        w_ns = w_ns / c_ns if c_ns > 0 else 0
        w_ew = w_ew / c_ew if c_ew > 0 else 0

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
    # PPO reward
    # -----------------------------
    def _compute_reward(self, prev_phase):

        emergency_delay = 0.0
        normal_delay = 0.0
        queue_length = 0
        vehicles_passed = 0
        emergency_present = False

        for vid in traci.vehicle.getIDList():

            if traci.vehicle.getTypeID(vid) == "emergency":

                emergency_present = True
                emergency_delay += traci.vehicle.getWaitingTime(vid)

            else:

                normal_delay += traci.vehicle.getWaitingTime(vid)

        for lane_id in traci.lane.getIDList():

            queue_length += traci.lane.getLastStepHaltingNumber(lane_id)

            vehicles_passed += traci.lane.getLastStepVehicleNumber(lane_id)

        switch_penalty = 1.0 if prev_phase != self.last_phase else 0.0

        reward = (
            - self.w_emergency * emergency_delay
            - self.w_normal * normal_delay
            - self.w_queue * queue_length
            - self.w_switch * switch_penalty
        )

        reward += self.w_throughput * vehicles_passed

        if emergency_present and emergency_delay == 0:
            reward += self.w_emergency_clear

        reward /= 10.0

        reward = np.clip(reward, -10.0, 10.0)

        return reward