import os
import random
import numpy as np
import torch

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.hea_agent import HEAAgent

# ✅ Correct imports
from rl.dqn_dueling import DuelingDQNAgent
from rl.apa_agent_v2 import APAAgentV2
from rl.apa_dqn_v2 import APADuelingDQN_V2

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", DEVICE)

# =========================
# CONFIG
# =========================
EPISODES = 300
MAX_STEPS = 1000

SCENARIOS = [
    "baseline",
    "low_demand",
    "high_demand",
    "emergency_stress",
    "unseen"
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD EMERGENCY AGENT
# =========================
def load_emergency_agent():

    model_path = "models/dqn_dueling_tls.pt"  # ✅ FIXED

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    agent = DuelingDQNAgent(
        state_dim=7,
        action_dim=3,
        device=DEVICE
    )

    agent.q_net.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    agent.q_net.eval()

    return agent


# =========================
# LOAD NORMAL AGENT
# =========================
def load_normal_agent():

    model_path = "models/apa_dqn_v2.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    agent = APAAgentV2(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
        model_class=APADuelingDQN_V2
    )

    agent.q_net.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    agent.q_net.eval()

    return agent


# =========================
# MAIN TRAIN LOOP
# =========================
def main():

    print("🔥 HEA-RL TRAINING STARTED")

    emergency_agent = load_emergency_agent()
    normal_agent = load_normal_agent()

    agent = HEAAgent(
        state_dim=7,
        device=DEVICE,
        emergency_agent=emergency_agent,
        normal_agent=normal_agent
    )

    for ep in range(1, EPISODES + 1):

        scenario = random.choice(SCENARIOS)
        env = TrafficSignalEnvDuelingV2(scenario=scenario, seed=ep)

        state = env.reset()
        done = False

        total_reward = 0.0
        steps = 0

        try:
            while not done and steps < MAX_STEPS:

                action, mode = agent.act(state)

                next_state, reward, done, _ = env.step(action)

                agent.remember(state, mode, reward, next_state, done)
                agent.learn()

                state = next_state
                total_reward += reward
                steps += 1

        finally:
            # Clean SUMO shutdown
            try:
                import traci
                if traci.isLoaded():
                    traci.close()
            except:
                pass

        if ep % 10 == 0:
            print(
                f"HEA-RL | Ep {ep}/{EPISODES} | "
                f"Scenario: {scenario:17s} | "
                f"Reward: {total_reward:8.2f}"
            )

        # Save selector checkpoints
        if ep % 50 == 0:
            torch.save(
                agent.selector.state_dict(),
                os.path.join(MODEL_DIR, f"hea_selector_ep{ep}.pth")
            )

    # Final save
    torch.save(
        agent.selector.state_dict(),
        os.path.join(MODEL_DIR, "hea_selector.pth")
    )

    print("✅ HEA-RL training complete")


if __name__ == "__main__":
    main()