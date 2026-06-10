import os
import torch
import numpy as np
import traci
import random

from rl.env import TrafficSignalEnv
from rl.dqn_dueling import DuelingQNetwork
import rl.utils_seed as utils_seed

# =========================
# Paths & constants
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dqn_dueling_tls.pt")

STATE_DIM = 7
ACTION_DIM = 3
EVAL_EPSILON = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_dqn_dueling_once(seed: int, scenario: str = "baseline"):

    # ---- Seed control ----
    utils_seed.set_seed(seed)
    random.seed(seed)

    # ---- Load model ----
    policy_net = DuelingQNetwork(STATE_DIM, ACTION_DIM).to(device)
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()

    # ---- Environment ----
    env = TrafficSignalEnv(auto_close=False, scenario=scenario)
    state = env.reset()

    # ---- Proper per-vehicle tracking ----
    vehicle_wait = {}
    prev_wait = {}
    vehicle_type = {}

    queue_lengths = []
    done = False

    while not done:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            if random.random() < EVAL_EPSILON:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                action = policy_net(s).argmax(dim=1).item()

        state, _, done, _ = env.step(action)

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
        "queue": float(np.mean(queue_lengths)),
    }


if __name__ == "__main__":
    out = evaluate_dqn_dueling_once(seed=0, scenario="baseline")
    print("\n===== DUELING DQN EVALUATION =====")
    print(out)