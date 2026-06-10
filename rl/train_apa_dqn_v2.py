import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.apa_dqn_v2 import APADuelingDQN_V2
from rl.apa_agent_v2 import APAAgentV2

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

    print("🔥 APA-DQN-V2 TRAINING STARTED")

    agent = APAAgentV2(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
        model_class=APADuelingDQN_V2
    )

    epsilon = EPS_START

    # 🔥 ADD: reward tracking
    rewards_per_episode = []

    for ep in range(1, EPISODES + 1):

        scenario = random.choice(SCENARIOS)
        env = TrafficSignalEnvDuelingV2(scenario=scenario, seed=ep)

        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        try:
            while not done and steps < MAX_STEPS:

                action = agent.act(state, epsilon)

                next_state, reward, done, _ = env.step(action)

                agent.remember(state, action, reward, next_state, done)
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

        # 🔥 ADD: store reward
        rewards_per_episode.append(total_reward)

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Logging
        if ep % 10 == 0:
            print(
                f"APA-DQN-V2 | Ep {ep}/{EPISODES} | "
                f"Scenario: {scenario:17s} | "
                f"Reward: {total_reward:8.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # Save checkpoints
        if ep % 50 == 0:
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(MODEL_DIR, f"apa_dqn_v2_ep{ep}.pth")
            )

    # Final save
    torch.save(
        agent.q_net.state_dict(),
        os.path.join(MODEL_DIR, "apa_dqn_v2.pth")
    )

    print("✅ APA-DQN-V2 training complete")

    # =========================
    # SAVE CSV
    # =========================
    with open("rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards_per_episode):
            writer.writerow([i + 1, r])

    print("📁 rewards.csv saved")

    # =========================
    # PLOT GRAPH (IMPROVED)
    # =========================

    rewards = np.array(rewards_per_episode)

    # Moving average smoothing
    window = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    plt.figure()

    # Raw rewards (light, transparent)
    plt.plot(rewards, alpha=0.3, label="Raw Reward")

    # Smoothed rewards (main curve)
    plt.plot(range(window-1, len(rewards)), smoothed, linewidth=2, label="Smoothed Reward")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Convergence of APA-DQN-V2")
    plt.legend()
    plt.grid()

    plt.savefig("training_curve.png")
    plt.close()

    print("📊 training_curve.png saved (smoothed)")


if __name__ == "__main__":
    main()