# ============================================================
# XAI PHASE 3 – COUNTERFACTUAL DECISION FLIPS
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl.dqn_dueling import DuelingQNetwork
from xai.feature_config import FEATURE_NAMES

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_PATH = "dueling_dqn_v2.pth"
STATE_DIM = 7
ACTION_DIM = 3

EMERGENCY_FEATURES = [4]
DELTA = 1.0
MAX_STEPS = 5

RESULT_DIR = "xai/results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------
# PHASE FUNCTION
# ------------------------------------------------------------
def run_phase3():

    print("Running Phase 3: Counterfactual Analysis")

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

    def get_action(state):
        with torch.no_grad():
            q = model(torch.tensor(state).float().unsqueeze(0))
        return torch.argmax(q, dim=1).item()

    # --------------------------------------------------------
    # COUNTERFACTUAL SEARCH
    # --------------------------------------------------------
    flip_counts = {}
    feature_change = []
    action_change = []

    for state, is_emg in zip(states, emergency_flags):

        if not is_emg:
            continue

        base_action = get_action(state)

        for idx in EMERGENCY_FEATURES:

            cf = state.copy()

            for step in range(1, MAX_STEPS + 1):

                cf[idx] += step * DELTA
                new_action = get_action(cf)

                if new_action != base_action:

                    fname = FEATURE_NAMES[idx]
                    flip_counts[fname] = flip_counts.get(fname, 0) + 1

                    feature_change.append(step * DELTA)
                    action_change.append(abs(new_action - base_action))

                    break

    # --------------------------------------------------------
    # SAVE COUNTERFACTUAL CSV
    # --------------------------------------------------------
    df = pd.DataFrame({
        "feature_change": feature_change,
        "action_change": action_change
    })

    df.to_csv(os.path.join(RESULT_DIR, "counterfactual.csv"), index=False)

    # --------------------------------------------------------
    # COUNTERFACTUAL PLOT
    # --------------------------------------------------------
    plt.figure(figsize=(8,5))

    if flip_counts:
        plt.bar(flip_counts.keys(), flip_counts.values())

    plt.ylabel("Decision Flip Count")
    plt.title("Counterfactual Sensitivity (Emergency Features)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULT_DIR, "counterfactual_feature_sensitivity.png"))
    plt.close()

    # ========================================================
    # POLICY RESPONSE WHEN EMERGENCY VEHICLE APPEARS
    # ========================================================

    print("Generating emergency response analysis...")

    emergency_probs = []
    normal_probs = []

    with torch.no_grad():

        for state, is_emg in zip(states, emergency_flags):

            s = torch.tensor(state).float().unsqueeze(0)

            q_values = model(s)

            probs = torch.softmax(q_values, dim=1).numpy()[0]

            green_prob = probs[0]

            if is_emg:
                emergency_probs.append(green_prob)
            else:
                normal_probs.append(green_prob)

    # --------------------------------------------------------
    # SAVE POLICY RESPONSE CSV (LONG FORMAT)
    # --------------------------------------------------------

    traffic_type = (["Normal"] * len(normal_probs)) + (["Emergency"] * len(emergency_probs))
    green_probs = normal_probs + emergency_probs

    df_policy = pd.DataFrame({
        "traffic_state": traffic_type,
        "green_phase_probability": green_probs
    })

    df_policy.to_csv(os.path.join(RESULT_DIR, "policy_response.csv"), index=False)

    # --------------------------------------------------------
    # POLICY RESPONSE PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.boxplot(
        [normal_probs, emergency_probs],
        tick_labels=["Normal Traffic", "Emergency Present"]
    )

    plt.ylabel("Probability of Green Phase")
    plt.title("Policy Response to Emergency Vehicles")

    plt.tight_layout()

    plt.savefig(os.path.join(RESULT_DIR, "policy_response_emergency.png"))
    plt.close()

    # ========================================================
    # EXTRA FIGURE (STRONG FOR PAPER)
    # Emergency Sensitivity Curve
    # ========================================================

    emergency_indicator = []
    green_probability = []

    with torch.no_grad():

        for state in states:

            s = torch.tensor(state).float().unsqueeze(0)

            q = model(s)
            probs = torch.softmax(q, dim=1).numpy()[0]

            emergency_indicator.append(state[4])
            green_probability.append(probs[0])

    plt.figure(figsize=(6,4))

    plt.scatter(emergency_indicator, green_probability, alpha=0.4)

    plt.xlabel("Emergency Feature Value")
    plt.ylabel("Green Phase Probability")
    plt.title("Policy Sensitivity to Emergency Vehicle Feature")

    plt.tight_layout()

    plt.savefig(os.path.join(RESULT_DIR, "emergency_feature_sensitivity.png"))
    plt.close()

    print("Phase 3 complete")

    return feature_change, action_change


# ------------------------------------------------------------
# RUN STANDALONE
# ------------------------------------------------------------
if __name__ == "__main__":
    run_phase3()