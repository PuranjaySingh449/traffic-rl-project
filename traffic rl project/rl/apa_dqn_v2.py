import torch
import torch.nn as nn
import torch.nn.functional as F


class APADuelingDQN_V2(nn.Module):
    """
    APA-DQN V2 (Improved)

    Improvements:
    ✔ Temperature-scaled attention (sharper focus)
    ✔ Better feature reuse
    ✔ More stable priority learning
    ✔ Keeps dueling structure intact
    """

    def __init__(self, state_dim, action_dim):
        super(APADuelingDQN_V2, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # =========================
        # Feature extractor
        # =========================
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # =========================
        # 🔥 Improved Attention (Temperature-scaled)
        # =========================
        self.attn_fc = nn.Linear(128, state_dim)
        self.temperature = 0.5  # lower = sharper attention

        # =========================
        # Priority Network (Improved stability)
        # =========================
        self.priority_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # =========================
        # Dueling Streams
        # =========================
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    # =========================
    # Forward
    # =========================
    def forward(self, state):

        # Initial features
        x = self.feature(state)

        # =========================
        # 🔥 Attention (sharper)
        # =========================
        attn_logits = self.attn_fc(x) / self.temperature
        attn_weights = F.softmax(attn_logits, dim=-1)

        attended_state = state * attn_weights

        # Recompute features
        x = self.feature(attended_state)

        # =========================
        # Priority weights
        # =========================
        priority_logits = self.priority_net(x)
        priority_weights = F.softmax(priority_logits, dim=-1)

        # =========================
        # Dueling Q
        # =========================
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, priority_weights