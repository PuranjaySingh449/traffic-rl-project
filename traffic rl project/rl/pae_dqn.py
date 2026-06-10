import torch
import torch.nn as nn
import torch.nn.functional as F


class PAEDuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PAEDuelingDQN, self).__init__()

        # -------- Shared Deep Feature Extractor --------
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        # -------- Attention Layer (focus on important features) --------
        self.attention = nn.Linear(128, 128)

        # -------- Value Stream --------
        self.value = nn.Linear(128, 1)

        # -------- Split Advantage Streams --------
        self.adv_emergency = nn.Linear(128, action_dim)
        self.adv_normal = nn.Linear(128, action_dim)

        # -------- Priority Weights --------
        self.w_e = 2.0   # emergency importance
        self.w_n = 1.0   # normal traffic importance

    def forward(self, x):
        # -------- Deep Feature Extraction --------
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # -------- Attention Mechanism --------
        attn = torch.softmax(self.attention(x), dim=-1)
        x = x * attn

        # -------- Streams --------
        V = self.value(x)

        Ae = self.adv_emergency(x)
        An = self.adv_normal(x)

        # -------- Priority-Aware Q Function --------
        Q = V + (self.w_e * Ae + self.w_n * An)

        return Q