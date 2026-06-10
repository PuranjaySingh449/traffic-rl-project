import os
import random
import numpy as np
import torch

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.apa_dqn import APADuelingDQN
from rl.apa_agent import APAAgent

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 500
MAX_STEPS = 1000

SCENARIOS = [
    "baseline",
    "low_demand",
    "high_demand",
    "emergency_stress",
    "unseen"
]

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

SEED = 42

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# SEED
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

# =========================
# MAIN
# =========================
def main():

    print("🔥 APA-DQN TRAINING STARTED")

    agent = APAAgent(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
        model_class=APADuelingDQN
    )

    epsilon = EPS_START

    for ep in range(1, EPISODES + 1):

        scenario = random.choice(SCENARIOS)
        env = TrafficSignalEnvDuelingV2(scenario=scenario, seed=ep)

        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:

            action = agent.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Logging
        if ep % 10 == 0:
            print(
                f"APA-DQN | Ep {ep}/{EPISODES} | "
                f"Scenario: {scenario:17s} | "
                f"Reward: {total_reward:8.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # Save checkpoints
        if ep % 50 == 0:
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(MODEL_DIR, f"apa_dqn_ep{ep}.pth")
            )

    # Final save
    torch.save(
        agent.q_net.state_dict(),
        os.path.join(MODEL_DIR, "apa_dqn.pth")
    )

    print("✅ APA-DQN training complete")


if __name__ == "__main__":
    main()