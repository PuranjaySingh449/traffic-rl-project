import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from rl.env import TrafficSignalEnv
from rl.pae_dqn import PAEDuelingDQN   # 🔥 NEW MODEL
from rl.dqn_dueling import ReplayBuffer

# =========================
# Hyperparameters
# =========================
EPISODES = 50
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50000
MIN_BUFFER = 1000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 5

STATE_DIM = 7
ACTION_DIM = 3

# =========================
# Setup
# =========================
env = TrafficSignalEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 USE PAE-DQN INSTEAD
policy_net = PAEDuelingDQN(STATE_DIM, ACTION_DIM).to(device)
target_net = PAEDuelingDQN(STATE_DIM, ACTION_DIM).to(device)

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

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:

        if np.random.rand() < epsilon:
            action = np.random.randint(ACTION_DIM)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(s).argmax().item()

        next_state, reward, done, info = env.step(action)

        # 🔥 OPTIONAL: Priority reward boost
        if isinstance(info, dict) and info.get("emergency_present", False):
            reward *= 1.5   # boost emergency importance

        total_reward += reward

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) >= MIN_BUFFER:

            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = states.to(device)
            next_states = next_states.to(device)
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

            # ✅ stability
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

            optimizer.step()

    episode_rewards.append(total_reward)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if (episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(
        f"PAE-DQN | Episode {episode+1}/{EPISODES} | "
        f"Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}"
    )

# =========================
# Save Model & Logs
# =========================
torch.save(policy_net.state_dict(), "models/pae_dqn_tls.pt")

np.save("logs/pae_dqn_rewards.npy", np.array(episode_rewards))

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PAE-DQN Training Curve")
plt.grid()
plt.savefig("plots/pae_dqn_training_curve.png")
plt.show()

print("✅ PAE-DQN training completed.")