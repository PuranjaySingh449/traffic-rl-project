import os
import random
import numpy as np
import torch

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.rcd_dqn import RCDDuelingDQN
from rl.rcd_agent import RCDAgent

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
# SEED CONTROL
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

# =========================
# MAIN TRAINING LOOP
# =========================
def main():

    print("🔥 RCD-DQN TRAINING STARTED")

    agent = RCDAgent(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
        model_class=RCDDuelingDQN
    )

    epsilon = EPS_START

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

                # =========================
                # 🔥 REWARD SHAPING (IMPORTANT)
                # =========================
                # strengthen emergency priority signal
                # assuming reward is negative delay already
                shaped_reward = reward

                # stronger penalty for bad states
                if reward < -50:
                    shaped_reward += reward * 0.5

                agent.remember(state, action, shaped_reward, next_state, done)
                agent.learn()

                state = next_state
                total_reward += shaped_reward
                steps += 1

        finally:
            # clean SUMO shutdown
            try:
                import traci
                if traci.isLoaded():
                    traci.close()
            except:
                pass

        # =========================
        # EPSILON DECAY
        # =========================
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # =========================
        # LOGGING
        # =========================
        if ep % 10 == 0:
            print(
                f"RCD-DQN | Ep {ep}/{EPISODES} | "
                f"Scenario: {scenario:17s} | "
                f"Reward: {total_reward:8.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # =========================
        # CHECKPOINTS
        # =========================
        if ep % 50 == 0:
            torch.save(
                agent.q_net.state_dict(),
                os.path.join(MODEL_DIR, f"rcd_dqn_ep{ep}.pth")
            )

    # =========================
    # FINAL SAVE
    # =========================
    torch.save(
        agent.q_net.state_dict(),
        os.path.join(MODEL_DIR, "rcd_dqn.pth")
    )

    print("✅ RCD-DQN training complete")


if __name__ == "__main__":
    main()