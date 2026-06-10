import os
import torch
import numpy as np
import traci
import torch.nn as nn

from rl.env_ppo_v2 import TrafficSignalEnvPPO
import rl.utils_seed as utils_seed

STATE_DIM = 7
ACTION_DIM = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_tls.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, ACTION_DIM)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


def evaluate_ppo_once(seed: int, scenario: str = "baseline"):
    utils_seed.set_seed(seed)

    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    env = TrafficSignalEnvPPO(auto_close=False, scenario=scenario)
    state = env.reset()

    # ---- Proper per-vehicle tracking ----
    vehicle_wait = {}
    prev_wait = {}
    vehicle_type = {}

    queue_lengths = []
    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(state_t)
            action = torch.argmax(logits, dim=1).item()

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
        "queue": float(np.mean(queue_lengths))
    }


if __name__ == "__main__":
    out = evaluate_ppo_once(seed=0, scenario="baseline")
    print("\n===== PPO EVALUATION RESULTS =====")
    print(out)
