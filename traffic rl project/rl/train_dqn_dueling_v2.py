import random
import numpy as np
import torch

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.dqn_dueling import DuelingDQNAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 500
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


def main():
    agent = DuelingDQNAgent(
        state_dim=7,
        action_dim=3,
        device=DEVICE,
    )

    epsilon = EPS_START

    for ep in range(1, EPISODES + 1):
        scenario = random.choice(SCENARIOS)
        env = TrafficSignalEnvDuelingV2(scenario=scenario, seed=ep)

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # ✅ CORRECT action call
            action = agent.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)

            # ✅ CORRECT replay storage
            agent.remember(state, action, reward, next_state, done)

            # ✅ CORRECT learning step
            agent.learn()

            state = next_state
            total_reward += reward

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % 10 == 0:
            print(
                f"Dueling-DQN-V2 | Ep {ep}/{EPISODES} | "
                f"Scenario: {scenario:17s} | "
                f"Reward: {total_reward:8.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    # Save model
    torch.save(agent.q_net.state_dict(), "dueling_dqn_v2.pth")
    print("✅ Dueling DQN V2 training complete")


if __name__ == "__main__":
    main()
