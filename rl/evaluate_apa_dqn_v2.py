import torch
import random
import numpy as np
import traci

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.apa_dqn_v2 import APADuelingDQN_V2
from rl.apa_agent_v2 import APAAgentV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/apa_dqn_v2.pth"


def evaluate_apa_dqn_v2_once(seed=0, scenario="baseline"):

    # =========================
    # SEED CONTROL
    # =========================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # =========================
    # ENV
    # =========================
    env = TrafficSignalEnvDuelingV2(
        scenario=scenario,
        seed=seed,
        auto_close=False
    )

    # =========================
    # AGENT
    # =========================
    agent = APAAgentV2(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
        model_class=APADuelingDQN_V2
    )

    agent.q_net.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    agent.q_net.eval()

    # =========================
    # RUN EPISODE
    # =========================
    state = env.reset()
    done = False

    emergency_wait = 0.0
    normal_wait = 0.0
    normal_count = 0
    queue_vals = []

    try:
        while not done:

            action = agent.act(state, epsilon=0.0)
            state, _, done, _ = env.step(action)

            # =========================
            # METRICS (TRACI)
            # =========================
            for vid in traci.vehicle.getIDList():

                wt = traci.vehicle.getWaitingTime(vid)

                if traci.vehicle.getTypeID(vid) == "emergency":
                    emergency_wait += wt
                else:
                    normal_wait += wt
                    normal_count += 1

            queue_vals.append(
                sum(
                    traci.lane.getLastStepHaltingNumber(l)
                    for l in traci.lane.getIDList()
                )
            )

    finally:
        # Clean shutdown
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass

    return {
        "emergency": float(emergency_wait),
        "normal": float(normal_wait / max(1, normal_count)),
        "queue": float(np.mean(queue_vals)) if queue_vals else 0.0,
    }


# =========================
# SOLO TEST
# =========================
if __name__ == "__main__":

    out = evaluate_apa_dqn_v2_once(seed=0, scenario="baseline")

    print("\n===== APA-DQN-V2 SOLO EVAL =====")
    print(out)