# ============================================================
# VALUE vs ADVANTAGE EXTRACTION (ROBUST VERSION)
# Works for Dueling + APA-DQN-V2
# ============================================================

import torch
import numpy as np


def _get_streams(model):
    """
    Detect correct layer names dynamically
    """

    # Feature layer
    if hasattr(model, "feature"):
        feature_layer = model.feature
    elif hasattr(model, "feature_layer"):
        feature_layer = model.feature_layer
    else:
        raise AttributeError("Feature layer not found")

    # Value stream
    if hasattr(model, "value"):
        value_layer = model.value
    elif hasattr(model, "value_stream"):
        value_layer = model.value_stream
    elif hasattr(model, "value_head"):
        value_layer = model.value_head
    else:
        raise AttributeError("Value layer not found")

    # Advantage stream
    if hasattr(model, "advantage"):
        adv_layer = model.advantage
    elif hasattr(model, "advantage_stream"):
        adv_layer = model.advantage_stream
    elif hasattr(model, "advantage_head"):
        adv_layer = model.advantage_head
    else:
        raise AttributeError("Advantage layer not found")

    return feature_layer, value_layer, adv_layer


def get_value_advantage(model, state_np):
    """
    Returns:
        value: float
        advantage: numpy array [num_actions]
    """

    model.eval()
    device = next(model.parameters()).device

    state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)

    # ---------------------------
    # Get correct layers
    # ---------------------------
    feature_layer, value_layer, adv_layer = _get_streams(model)

    with torch.no_grad():

        features = feature_layer(state)

        value = value_layer(features)          # [1,1]
        advantage = adv_layer(features)        # [1,action_dim]

        # Normalize advantage (dueling trick)
        advantage = advantage - advantage.mean(dim=1, keepdim=True)

    return (
        value.item(),
        advantage.squeeze(0).cpu().numpy()
    )