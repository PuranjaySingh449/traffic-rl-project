import os
import torch
import numpy as np
import traci
import random

from rl.env import TrafficSignalEnv
from rl.dqn import QNetwork
import rl.utils_seed as utils_seed

# =========================
# Paths & constants
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "dqn_tls.pt")

STATE_DIM = 7
ACTION_DIM = 3

# Small exploration for robustness during evaluation
EVAL_EPSILON = 0.05

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_dqn_once(seed: int, scenario: str = "baseline"):
    # =========================
    # Seed control
    # =========================
    utils_seed.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # =========================
    # Load trained DQN model
    # =========================
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        policy_net.load_state_dict(checkpoint)

    policy_net.eval()

    # =========================
    # Environment
    # =========================
    env = TrafficSignalEnv(auto_close=False, scenario=scenario)
    state = env.reset()

    # =========================
    # FIX: Proper waiting-time accumulation
    # =========================
    vehicle_wait = {}     # total accumulated waiting time per vehicle
    prev_wait = {}        # previous waiting time snapshot

    queue_lengths = []
    done = False

    # =========================
    # Simulation loop
    # =========================
    while not done:
        with torch.no_grad():
            state_t = (
                torch.tensor(state, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )

            if random.random() < EVAL_EPSILON:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                action = policy_net(state_t).argmax(dim=1).item()

        state, _, done, _ = env.step(action)

        # ---- Correct per-step waiting accumulation ----
        for vid in traci.vehicle.getIDList():
            cur_wait = traci.vehicle.getWaitingTime(vid)

            if vid not in vehicle_wait:
                vehicle_wait[vid] = 0.0
                prev_wait[vid] = cur_wait
            else:
                delta = max(0.0, cur_wait - prev_wait[vid])
                vehicle_wait[vid] += delta
                prev_wait[vid] = cur_wait

        # Queue length (network-wide)
        queue_lengths.append(
            sum(
                traci.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in traci.lane.getIDList()
            )
        )

    # =========================
    # Close SUMO cleanly
    # =========================
    if traci.isLoaded():
        traci.close()

    # =========================
    # Final metrics
    # =========================
    emergency_wait = 0.0
    normal_wait = 0.0
    normal_count = 0

    for vid, wt in vehicle_wait.items():
        if "emergency" in vid:
            emergency_wait += wt
        else:
            normal_wait += wt
            normal_count += 1

    return {
        "emergency": emergency_wait,
        "normal": normal_wait / max(1, normal_count),
        "queue": float(np.mean(queue_lengths)),
    }


# =========================
# Standalone execution
# =========================
if __name__ == "__main__":
    out = evaluate_dqn_once(seed=0, scenario="baseline")
    print("\n===== DQN EVALUATION RESULTS =====")
    for k, v in out.items():
        print(f"{k}: {v:.2f}")
