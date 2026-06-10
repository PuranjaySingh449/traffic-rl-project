import torch
import numpy as np
import random
import traci

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.pae_dqn import PAEDuelingDQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/pae_dqn_v2.pth"


def evaluate_pae_dqn_v2_once(seed=0, scenario="baseline"):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = TrafficSignalEnvDuelingV2(
        scenario=scenario,
        seed=seed,
        auto_close=False
    )

    model = PAEDuelingDQN(7, 3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    state = env.reset()
    done = False

    emergency_wait = 0.0
    normal_wait = 0.0
    normal_count = 0
    queue_vals = []

    while not done:

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = torch.argmax(model(s)).item()

        state, _, done, _ = env.step(action)

        for vid in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(vid)

            if traci.vehicle.getTypeID(vid) == "emergency":
                emergency_wait += wt
            else:
                normal_wait += wt
                normal_count += 1

        queue_vals.append(
            sum(traci.lane.getLastStepHaltingNumber(l)
                for l in traci.lane.getIDList())
        )

    if traci.isLoaded():
        traci.close()

    return {
        "emergency": float(emergency_wait),
        "normal": float(normal_wait / max(1, normal_count)),
        "queue": float(np.mean(queue_vals)) if queue_vals else 0.0,
    }