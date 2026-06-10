import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

from rl.env import TrafficSignalEnv
from rl.dqn import QNetwork, ReplayBuffer

# =========================
# Hyperparameters
# =========================
EPISODES = 80
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_BUFFER = 2000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.992
TARGET_UPDATE = 5

STATE_DIM = 7
ACTION_DIM = 3

# =========================
# Training scenarios
# =========================
SCENARIOS = [
    "baseline",
    "low_demand",
    "high_demand",
    "emergency_stress",
]

# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
target_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPS_START
episode_rewards = []

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# =========================
# Training Loop
# =========================
for episode in range(EPISODES):

    # 🔥 Scenario randomization (KEY FIX)
    scenario = random.choice(SCENARIOS)
    env = TrafficSignalEnv(auto_close=True, scenario=scenario)

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:

        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, ACTION_DIM - 1)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(s).argmax(dim=1).item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # -------------------------
        # Learn
        # -------------------------
        if len(replay_buffer) >= MIN_BUFFER:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(np.array(states)).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                q_next = target_net(next_states).max(1)[0]
                q_target = rewards + GAMMA * q_next * (1 - dones)

            loss = (q - q_target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

    episode_rewards.append(total_reward)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(
        f"DQN | Ep {episode+1}/{EPISODES} | "
        f"Scenario: {scenario:<17} | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {epsilon:.3f}"
    )

# =========================
# Save Model + Curve
# =========================
torch.save(
    {"model_state_dict": policy_net.state_dict()},
    "models/dqn_tls.pt"
)

np.save("logs/dqn_rewards.npy", episode_rewards)

plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Curve (Multi-Scenario)")
plt.grid()
plt.tight_layout()
plt.savefig("plots/dqn_training_curve.png")
plt.show()
