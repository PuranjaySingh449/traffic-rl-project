# ============================================================
# COUNTERFACTUAL EXPLANATIONS FOR DUELING DQN
# ============================================================

import numpy as np
import torch

def get_action(model, state_np):
    state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_vals = model(state)
        return torch.argmax(q_vals, dim=1).item()


def generate_counterfactual(model,
                             state_np,
                             feature_idx,
                             delta,
                             max_steps=5):
    """
    Try minimal changes to one feature until action flips
    """
    original_action = get_action(model, state_np)
    cf_state = state_np.copy()

    for step in range(1, max_steps + 1):
        cf_state[feature_idx] += step * delta
        new_action = get_action(model, cf_state)

        if new_action != original_action:
            return {
                "feature_idx": feature_idx,
                "change": step * delta,
                "original_action": original_action,
                "new_action": new_action
            }

    return None
