import torch
import torch.nn as nn
import torch.nn.functional as F


class HEASelector(nn.Module):
    """
    High-Level Selector Network (π_high)

    Decides:
    0 → NORMAL mode
    1 → EMERGENCY mode
    """

    def __init__(self, state_dim):
        super(HEASelector, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 modes
        )

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return probs