import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class APAAgent:
    def __init__(self, state_dim, action_dim, device, model_class):

        self.device = device
        self.action_dim = action_dim

        # =========================
        # Networks
        # =========================
        self.q_net = model_class(state_dim, action_dim).to(device)
        self.target_net = model_class(state_dim, action_dim).to(device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=5e-4)

        # =========================
        # Replay Buffer
        # =========================
        self.memory = []
        self.max_memory = 50000
        self.batch_size = 64

        self.gamma = 0.99

        # =========================
        # Target update
        # =========================
        self.update_target_every = 1000
        self.learn_step = 0

    # =========================
    # ACTION
    # =========================
    def act(self, state, epsilon):

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values, _ = self.q_net(state)

        return torch.argmax(q_values).item()

    # =========================
    # MEMORY
    # =========================
    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    # =========================
    # LEARNING (OPTIMIZED)
    # =========================
    def learn(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        # ✅ FAST tensor conversion (FIXED)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.LongTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device)

        # =========================
        # CURRENT Q
        # =========================
        q_values, weights = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # =========================
        # TARGET Q
        # =========================
        with torch.no_grad():
            next_q, _ = self.target_net(next_states)
            max_next_q = next_q.max(1)[0]

        # =========================
        # 🔥 PRIORITY-AWARE REWARD
        # =========================
        emergency_weight = weights[:, 0]

        # smoother scaling (more stable than previous)
        adjusted_reward = rewards * (1.0 + 0.5 * emergency_weight)

        target = adjusted_reward + self.gamma * max_next_q * (1 - dones)

        # =========================
        # LOSS
        # =========================
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)

        self.optimizer.step()

        # =========================
        # TARGET UPDATE
        # =========================
        self.learn_step += 1

        if self.learn_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())