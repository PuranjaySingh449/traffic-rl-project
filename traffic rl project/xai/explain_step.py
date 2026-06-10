# ============================================================
# EXPLAIN SINGLE TIMESTEP
# ============================================================

import torch
from xai.integrated_gradients import integrated_gradients

def explain_timestep(model, state_np):
    """
    state_np: numpy array [state_dim]
    """
    state = torch.tensor(state_np, dtype=torch.float32)
    baseline = torch.zeros_like(state)

    with torch.no_grad():
        q_vals = model(state.unsqueeze(0))
        action = torch.argmax(q_vals, dim=1).item()

    ig = integrated_gradients(
        model=model,
        state=state,
        baseline=baseline,
        target_action=action
    )

    return ig.numpy(), action
