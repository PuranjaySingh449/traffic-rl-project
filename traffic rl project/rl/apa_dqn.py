import torch
import torch.nn as nn
import torch.nn.functional as F


class APADuelingDQN(nn.Module):
    """
    Adaptive Priority Attention Dueling DQN

    Components:
    1. Feature extractor
    2. Attention mechanism (lane importance)
    3. Priority weight network (dynamic reward weighting)
    4. Dueling streams (Value + Advantage)
    """

    def __init__(self, state_dim, action_dim):
        super(APADuelingDQN, self).__init__()

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
        # Attention layer (NEW)
        # =========================
        self.attention = nn.Sequential(
            nn.Linear(128, state_dim),
            nn.Softmax(dim=-1)
        )

        # =========================
        # Priority weight network (NEW)
        # Outputs weights for:
        # [emergency, normal, queue]
        # =========================
        self.priority_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)  # ensures weights sum to 1
        )

        # =========================
        # Dueling Streams
        # =========================

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    # =========================
    # Forward pass
    # =========================
    def forward(self, state):

        # Extract features
        x = self.feature(state)

        # =========================
        # Attention mechanism
        # =========================
        attn_weights = self.attention(x)  # shape: (batch, state_dim)

        # Apply attention to original state
        attended_state = state * attn_weights

        # Re-extract features from attended state
        x = self.feature(attended_state)

        # =========================
        # Priority weights (NEW)
        # =========================
        priority_weights = self.priority_net(x)
        # shape: (batch, 3)
        # [w_emergency, w_normal, w_queue]

        # =========================
        # Dueling Q computation
        # =========================
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, priority_weights