import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import traci

from rl.env_ppo_v2 import TrafficSignalEnvPPO
from utils.seeding import set_all_seeds

# =====================================================
# SEEDING
# =====================================================
SEED = 0
set_all_seeds(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# HYPERPARAMETERS
# =====================================================
EPISODES = 300
GAMMA = 0.99
LR = 3e-4

CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GRAD_CLIP = 0.5

STATE_DIM = 7
ACTION_DIM = 3

# =====================================================
# ACTOR–CRITIC NETWORK
# =====================================================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, ACTION_DIM)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

policy = ActorCritic().to(device)
optimizer = optim.Adam(policy.parameters(), lr=LR)

# =====================================================
# ENVIRONMENT
# =====================================================
env = TrafficSignalEnvPPO(
    auto_close=False,
    scenario="baseline",
    seed=SEED
)

# =====================================================
# LOGGING
# =====================================================
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

episode_rewards = []

# =====================================================
# TRAINING LOOP
# =====================================================
for episode in range(EPISODES):
    state = env.reset()

    states, actions, rewards = [], [], []
    log_probs, values = [], []

    total_reward = 0.0

    while True:
        state_t = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        logits, value = policy(state_t)
        dist = torch.distributions.Categorical(logits=logits)

        # ✅ CORRECT SAMPLING (compatible)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())

        states.append(state_t)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob.detach())
        values.append(value.squeeze(-1))

        total_reward += reward
        state = next_state

        if done:
            break

    episode_rewards.append(total_reward)

    # -------- Returns --------
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    values = torch.stack(values)

    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # -------- PPO Update --------
    states_tensor = torch.cat(states)
    actions_tensor = torch.cat(actions)
    old_log_probs_tensor = torch.cat(log_probs)

    logits, value_preds = policy(states_tensor)
    dist = torch.distributions.Categorical(logits=logits)

    new_log_probs = dist.log_prob(actions_tensor)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs_tensor)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (returns - value_preds.squeeze(-1)).pow(2).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    optimizer.step()

    print(f"PPO | Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f}")

# =====================================================
# SAVE MODEL & LOGS
# =====================================================
torch.save(policy.state_dict(), f"models/ppo_tls_seed{SEED}.pt")
np.save(f"logs/ppo_rewards_seed{SEED}.npy", np.array(episode_rewards))

# =====================================================
# PLOT
# =====================================================
plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, label="PPO")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Training Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/ppo_training_curve_seed{SEED}.png")
plt.show()

# =====================================================
# CLEAN SHUTDOWN
# =====================================================
if traci.isLoaded():
    traci.close()

print("✅ PPO training completed successfully.")
