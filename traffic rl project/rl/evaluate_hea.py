import torch
import random
import numpy as np
import traci

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.hea_agent import HEAAgent

# Experts
from rl.dqn_dueling import DuelingDQNAgent
from rl.apa_agent_v2 import APAAgentV2
from rl.apa_dqn_v2 import APADuelingDQN_V2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD EXPERTS
# =========================
def load_emergency_agent():
    agent = DuelingDQNAgent(
        state_dim=7,
        action_dim=3,
        device=DEVICE
    )
    agent.q_net.load_state_dict(
        torch.load("models/dqn_dueling_tls.pt", map_location=DEVICE)
    )
    agent.q_net.eval()
    return agent


def load_normal_agent():
    agent = APAAgentV2(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
        model_class=APADuelingDQN_V2
    )
    agent.q_net.load_state_dict(
        torch.load("models/apa_dqn_v2.pth", map_location=DEVICE)
    )
    agent.q_net.eval()
    return agent


# =========================
# LOAD HEA
# =========================
def load_hea():
    emergency_agent = load_emergency_agent()
    normal_agent = load_normal_agent()

    agent = HEAAgent(
        state_dim=7,
        device=DEVICE,
        emergency_agent=emergency_agent,
        normal_agent=normal_agent
    )

    agent.selector.load_state_dict(
        torch.load("models/hea_selector.pth", map_location=DEVICE)
    )
    agent.selector.eval()

    return agent


# =========================
# SINGLE EVAL
# =========================
def evaluate_hea_once(seed=0, scenario="baseline"):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = TrafficSignalEnvDuelingV2(
        scenario=scenario,
        seed=seed,
        auto_close=False
    )

    agent = load_hea()

    state = env.reset()
    done = False

    emergency_wait = 0.0
    normal_wait = 0.0
    normal_count = 0
    queue_vals = []

    try:
        while not done:

            action, _ = agent.act(state)
            state, _, done, _ = env.step(action)

            # =========================
            # METRICS
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
# TEST RUN
# =========================
if __name__ == "__main__":

    out = evaluate_hea_once(seed=0, scenario="baseline")

    print("\n===== HEA SOLO EVAL =====")
    print(out)