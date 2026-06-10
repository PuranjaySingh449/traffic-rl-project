import random
import numpy as np
import torch

from rl.env_dueling_v2 import TrafficSignalEnvDuelingV2
from rl.pae_dqn import PAEDuelingDQN

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

GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 50000

# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)

        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s_next),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)


def main():

    policy_net = PAEDuelingDQN(7, 3).to(DEVICE)
    target_net = PAEDuelingDQN(7, 3).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START

    for ep in range(1, EPISODES + 1):

        scenario = random.choice(SCENARIOS)
        env = TrafficSignalEnvDuelingV2(scenario=scenario, seed=ep)

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:

            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = torch.argmax(policy_net(s)).item()

            next_state, reward, done, _ = env.step(action)

            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # ================= LEARNING =================
            if len(memory) > BATCH_SIZE:

                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states = states.to(DEVICE)
                next_states = next_states.to(DEVICE)
                actions = actions.to(DEVICE)
                rewards = rewards.to(DEVICE)
                dones = dones.to(DEVICE)

                q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0]
                    q_target = rewards + GAMMA * q_next * (1 - dones)

                loss = (q - q_target).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % 10 == 0:
            print(
                f"PAE-DQN | Ep {ep}/{EPISODES} | "
                f"Scenario: {scenario:17s} | "
                f"Reward: {total_reward:8.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        if ep % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), "models/pae_dqn_v2.pth")
    print("✅ PAE-DQN training complete")


if __name__ == "__main__":
    main()