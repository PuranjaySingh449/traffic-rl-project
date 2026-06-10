import torch
import torch.nn as nn
import torch.nn.functional as F


class RCDDuelingDQN(nn.Module):
    """
    Risk-Constrained Distributional Dueling DQN (RCD-DQN)

    ✔ Dueling architecture
    ✔ Quantile regression (QR-DQN)
    ✔ Outputs distribution over returns
    """

    def __init__(self, state_dim, action_dim, n_quantiles=51):
        super(RCDDuelingDQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

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
        # Value stream (distributional)
        # =========================
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_quantiles)
        )

        # =========================
        # Advantage stream (distributional)
        # =========================
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * n_quantiles)
        )

    def forward(self, state):
        """
        Output:
        quantiles: [batch, action_dim, n_quantiles]
        """

        x = self.feature(state)

        # Value: [batch, n_quantiles]
        value = self.value_stream(x)

        # Advantage: [batch, action_dim * n_quantiles]
        advantage = self.advantage_stream(x)
        advantage = advantage.view(-1, self.action_dim, self.n_quantiles)

        # Expand value to match action dim
        value = value.unsqueeze(1)  # [batch, 1, n_quantiles]

        # Dueling combine
        quantiles = value + advantage - advantage.mean(dim=1, keepdim=True)

        return quantiles