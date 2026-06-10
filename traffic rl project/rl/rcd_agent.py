import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class RCDAgent:
    """
    Risk-Constrained Distributional DQN Agent

    ✔ Quantile Regression (QR-DQN)
    ✔ CVaR-based action selection (risk-aware)
    ✔ Emergency constraint penalty
    """

    def __init__(self, state_dim, action_dim, device, model_class):

        self.device = device
        self.action_dim = action_dim
        self.n_quantiles = 51

        # Networks
        self.q_net = model_class(state_dim, action_dim).to(device)
        self.target_net = model_class(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)

        # Replay
        self.memory = []
        self.max_memory = 50000
        self.batch_size = 64

        self.gamma = 0.99

        # CVaR settings
        self.cvar_alpha = 0.1  # focus on worst 10%
        self.k = int(self.n_quantiles * self.cvar_alpha)

        # Target update
        self.learn_step = 0
        self.update_target_every = 1000

    # =========================
    # ACTION (CVaR-based)
    # =========================
    def act(self, state, epsilon):

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            quantiles = self.q_net(state)  # [1, A, N]

        # CVaR: mean of lowest k quantiles
        cvar = quantiles[:, :, :self.k].mean(dim=2)

        return torch.argmax(cvar).item()

    # =========================
    # MEMORY
    # =========================
    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    # =========================
    # LEARN
    # =========================
    def learn(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.LongTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device)

        # =========================
        # CURRENT QUANTILES
        # =========================
        quantiles = self.q_net(states)
        quantiles = quantiles[range(self.batch_size), actions]  # [B, N]

        # =========================
        # TARGET QUANTILES
        # =========================
        with torch.no_grad():

            next_q = self.target_net(next_states)

            # CVaR action selection
            next_cvar = next_q[:, :, :self.k].mean(dim=2)
            next_actions = torch.argmax(next_cvar, dim=1)

            next_quantiles = next_q[range(self.batch_size), next_actions]

            target = rewards.unsqueeze(1) + self.gamma * next_quantiles * (1 - dones.unsqueeze(1))

        # =========================
        # 🔥 EMERGENCY CONSTRAINT
        # =========================
        # Penalize high negative rewards (proxy for emergency spikes)
        penalty = torch.clamp(-rewards, min=0) * 0.5
        target -= penalty.unsqueeze(1)

        # =========================
        # QUANTILE LOSS (Huber)
        # =========================
        td_error = target.unsqueeze(2) - quantiles.unsqueeze(1)

        huber = torch.where(
            td_error.abs() < 1.0,
            0.5 * td_error.pow(2),
            td_error.abs() - 0.5
        )

        tau = torch.linspace(0.0, 1.0, self.n_quantiles, device=self.device).view(1, -1)

        loss = (torch.abs(tau.unsqueeze(2) - (td_error.detach() < 0).float()) * huber).mean()

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