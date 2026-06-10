# ============================================================
# XAI PHASE 2 – VALUE vs ADVANTAGE DECOMPOSITION
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl.dqn_dueling import DuelingQNetwork

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
def run_phase2():

    print("Running Phase 2: Value vs Advantage Analysis")

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
    emergency_flags = np.load("logs/eval_emergency_flags.npy")

    values = []
    advantages = []
    impacts = []

    values_normal, values_emergency = [], []
    adv_normal, adv_emergency = [], []

    # --------------------------------------------------------
    # VALUE / ADVANTAGE EXTRACTION
    # --------------------------------------------------------
    with torch.no_grad():
        for s_np, is_emg in zip(states, emergency_flags):

            s = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0)

            features = model.feature(s)

            V = model.value(features)            # state value
            A = model.advantage(features)        # action advantages
            A = A - A.mean(dim=1, keepdim=True)

            v = V.item()
            a = torch.max(torch.abs(A)).item()

            values.append(v)
            advantages.append(a)

            # impact metric
            impact = abs(v) + a
            impacts.append(impact)

            if is_emg:
                values_emergency.append(v)
                adv_emergency.append(a)
            else:
                values_normal.append(v)
                adv_normal.append(a)

    timesteps = np.arange(len(values))

    # --------------------------------------------------------
    # SAVE CSV (FOR PAPER)
    # --------------------------------------------------------
    df = pd.DataFrame({
        "timestep": timesteps,
        "value": values,
        "advantage": advantages,
        "impact": impacts
    })

    df.to_csv(os.path.join(RESULT_DIR, "value_advantage.csv"), index=False)

    # --------------------------------------------------------
    # VALUE COMPARISON PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.boxplot([values_normal, values_emergency], labels=["Normal", "Emergency"])
    plt.ylabel("State Value V(s)")
    plt.title("Value Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"value_comparison.png"))
    plt.close()

    # --------------------------------------------------------
    # ADVANTAGE COMPARISON PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.boxplot([adv_normal, adv_emergency], labels=["Normal", "Emergency"])
    plt.ylabel("Max |Advantage|")
    plt.title("Advantage Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"advantage_comparison.png"))
    plt.close()

    print("Phase 2 complete")

    return timesteps, values, advantages, impacts


# ------------------------------------------------------------
# RUN STANDALONE
# ------------------------------------------------------------
if __name__ == "__main__":
    run_phase2()