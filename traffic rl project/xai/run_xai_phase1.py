# ============================================================
# XAI PHASE 1 – INTEGRATED GRADIENTS FEATURE IMPORTANCE
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl.dqn_dueling import DuelingQNetwork
from xai.integrated_gradients import integrated_gradients
from xai.feature_config import FEATURE_NAMES

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_PATH = "dueling_dqn_v2.pth"
STATE_DIM = 7
ACTION_DIM = 3

RESULT_DIR = "xai/results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------
# PHASE FUNCTION
# ------------------------------------------------------------
def run_phase1():

    print("Running Phase 1: Integrated Gradients")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    model = DuelingQNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    states = np.load("logs/eval_states.npy")

    feature_importance = np.zeros(len(FEATURE_NAMES))

    # --------------------------------------------------------
    # COMPUTE INTEGRATED GRADIENTS
    # --------------------------------------------------------
    for state in states[:200]:

        s = torch.tensor(state, dtype=torch.float32)

        # baseline reference state
        baseline = torch.zeros_like(s)

        # determine policy action
        with torch.no_grad():
            q_values = model(s.unsqueeze(0))
            target_action = torch.argmax(q_values, dim=1).item()

        # compute IG
        attributions = integrated_gradients(
            model,
            s,
            baseline,
            target_action
        )

        # convert tensor → numpy safely
        if torch.is_tensor(attributions):
            attributions = attributions.detach().cpu().numpy()

        feature_importance += np.abs(attributions)

    # normalize importance
    feature_importance = feature_importance / np.sum(feature_importance)

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------
    df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": feature_importance
    })

    df.to_csv(os.path.join(RESULT_DIR, "feature_importance.csv"), index=False)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(8,5))

    plt.barh(FEATURE_NAMES, feature_importance)

    plt.xlabel("Importance Score")
    plt.title("Feature Importance (Integrated Gradients)")

    plt.tight_layout()

    plt.savefig(os.path.join(RESULT_DIR, "feature_importance.png"))
    plt.close()

    print("Phase 1 complete")

    return FEATURE_NAMES, feature_importance


# ------------------------------------------------------------
# RUN STANDALONE
# ------------------------------------------------------------
if __name__ == "__main__":
    run_phase1()