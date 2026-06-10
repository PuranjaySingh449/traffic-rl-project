import traci
import numpy as np
from rl.env import TrafficSignalEnv

FIXED_GREEN = 30  # seconds


def evaluate_fixed_time_once(seed: int, scenario: str = "baseline"):
    env = TrafficSignalEnv(auto_close=False, scenario=scenario)
    env.reset()

    phase_timer = 0
    current_phase = traci.trafficlight.getPhase(env.tls_id)

    # ---- Proper per-vehicle tracking ----
    vehicle_wait = {}
    prev_wait = {}
    vehicle_type = {}

    queue_lengths = []
    done = False

    while not done:
        phase_timer += 1

        if phase_timer >= FIXED_GREEN:
            next_phase = (current_phase + 1) % len(
                traci.trafficlight
                .getCompleteRedYellowGreenDefinition(env.tls_id)[0]
                .phases
            )
            traci.trafficlight.setPhase(env.tls_id, next_phase)
            current_phase = next_phase
            phase_timer = 0

        traci.simulationStep()

        # ---- Delta-based waiting accumulation ----
        for vid in traci.vehicle.getIDList():
            cur_wait = traci.vehicle.getWaitingTime(vid)

            if vid not in vehicle_wait:
                vehicle_wait[vid] = 0.0
                prev_wait[vid] = cur_wait
                vehicle_type[vid] = traci.vehicle.getTypeID(vid)  # store type once
            else:
                delta = max(0.0, cur_wait - prev_wait[vid])
                vehicle_wait[vid] += delta
                prev_wait[vid] = cur_wait

        queue_lengths.append(
            sum(traci.lane.getLastStepHaltingNumber(l)
                for l in traci.lane.getIDList())
        )

        done = traci.simulation.getMinExpectedNumber() == 0

    # ---- Close TraCI safely ----
    if traci.isLoaded():
        traci.close()

    # ---- Final aggregation (NO TraCI calls here) ----
    emergency_wait = 0.0
    normal_wait = 0.0
    normal_count = 0

    for vid, wt in vehicle_wait.items():
        if vehicle_type.get(vid) == "emergency":
            emergency_wait += wt
        else:
            normal_wait += wt
            normal_count += 1

    return {
        "emergency": emergency_wait,
        "normal": normal_wait / max(1, normal_count),
        "queue": float(np.mean(queue_lengths))
    }


if __name__ == "__main__":
    out = evaluate_fixed_time_once(seed=0, scenario="baseline")
    print("\n===== FIXED-TIME RESULTS =====")
    print(out)