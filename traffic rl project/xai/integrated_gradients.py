# ============================================================
# INTEGRATED GRADIENTS (FINAL ROBUST VERSION)
# Supports DQN, Dueling, APA-DQN-V2
# ============================================================

import torch


def _get_q_values(model, x):
    """
    Handles both:
    - Standard models → return Q directly
    - Dueling models → return (value, advantage)
    """
    out = model(x)

    # If model returns (value, advantage)
    if isinstance(out, tuple):
        value, advantage = out
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    return out


def integrated_gradients(
    model,
    state,
    baseline=None,
    target_action=None,
    steps=50
):
    """
    model: PyTorch model
    state: torch.Tensor [state_dim]
    baseline: torch.Tensor [state_dim]
    target_action: int (optional)
    """

    model.eval()

    # ---------------------------
    # Device handling
    # ---------------------------
    device = next(model.parameters()).device
    state = state.to(device).float()

    # ---------------------------
    # Baseline
    # ---------------------------
    if baseline is None:
        baseline = torch.zeros_like(state).to(device)
    else:
        baseline = baseline.to(device).float()

    # ---------------------------
    # Target action
    # ---------------------------
    with torch.no_grad():
        if target_action is None:
            q_vals = _get_q_values(model, state.unsqueeze(0))
            target_action = torch.argmax(q_vals, dim=1).item()

    # ---------------------------
    # Integrated Gradients
    # ---------------------------
    total_gradients = torch.zeros_like(state).to(device)

    for alpha in torch.linspace(0, 1, steps, device=device):

        interpolated = baseline + alpha * (state - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)

        # Forward
        q_values = _get_q_values(model, interpolated.unsqueeze(0))
        q_target = q_values[0, target_action]

        # Backward
        model.zero_grad()
        if interpolated.grad is not None:
            interpolated.grad.zero_()

        q_target.backward()

        total_gradients += interpolated.grad.detach()

    # ---------------------------
    # Final IG
    # ---------------------------
    avg_gradients = total_gradients / steps
    ig = (state - baseline) * avg_gradients

    return ig.detach().cpu()