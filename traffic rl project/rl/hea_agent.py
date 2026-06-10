import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from rl.hea_selector import HEASelector


class HEAAgent:
    """
    Hierarchical Emergency-Aware RL Agent

    Combines:
    ✔ Selector network (learns mode)
    ✔ Emergency policy (Dueling-DQN-V2)
    ✔ Normal policy (APA / RCD)
    """

    def __init__(self, state_dim, device, emergency_agent, normal_agent):

        self.device = device

        self.selector = HEASelector(state_dim).to(device)
        self.optimizer = optim.Adam(self.selector.parameters(), lr=1e-3)

        self.emergency_agent = emergency_agent
        self.normal_agent = normal_agent

        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99

    # =========================
    # ACTION
    # =========================
    def act(self, state):

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.selector(state_t)

        mode = torch.argmax(probs).item()

        if mode == 1:
            action = self.emergency_agent.act(state, epsilon=0.0)
        else:
            action = self.normal_agent.act(state, epsilon=0.0)

        return action, mode

    # =========================
    # MEMORY
    # =========================
    def remember(self, state, mode, reward, next_state, done):
        self.memory.append((state, mode, reward, next_state, done))

        if len(self.memory) > 50000:
            self.memory.pop(0)

    # =========================
    # LEARNING (Selector training)
    # =========================
    def learn(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        modes = torch.LongTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)

        # Forward
        probs = self.selector(states)

        # Log prob of chosen mode
        log_probs = torch.log(probs + 1e-8)
        selected_log_probs = log_probs[range(self.batch_size), modes]

        # =========================
        # 🔥 Reward shaping for selector
        # =========================

        # Encourage EMERGENCY mode when reward is very negative
        target = (rewards < -20).float()  # proxy condition

        # Convert to label (0 or 1)
        target_modes = target.long()

        loss = nn.CrossEntropyLoss()(probs, target_modes)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()