import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from rl.fixed_time import evaluate_fixed_time_once

MODEL_NAME = "fixed_time"
SAVE_DIR = f"eval_results/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_CSV = os.path.join(SAVE_DIR, "results.csv")
ROBUST_CSV = os.path.join(SAVE_DIR, "robustness.csv")
SENS_CSV = os.path.join(SAVE_DIR, "sensitivity.csv")

SCENARIOS = [
    "baseline",
    "low_demand",
    "high_demand",
    "emergency_stress",
    "unseen"
]

SCENARIO_SEEDS = {
    "baseline": [1,2,3],
    "low_demand": [4,5,6],
    "high_demand": [7,8,9],
    "emergency_stress": [10,11,12],
    "unseen": [13,14,15],
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate_fn(seed, scenario):
    return evaluate_fixed_time_once(seed, scenario)

# ================= MAIN =================
ALL_RESULTS = {}

for scenario in SCENARIOS:

    em, no, qu = [], [], []

    for seed in SCENARIO_SEEDS[scenario]:

        set_seed(seed)
        out = evaluate_fn(seed, scenario)

        em.append(out["emergency"])
        no.append(out["normal"])
        qu.append(out["queue"])

    ALL_RESULTS[scenario] = {
        "emergency": (np.mean(em), np.std(em)),
        "normal": (np.mean(no), np.std(no)),
        "queue": (np.mean(qu), np.std(qu)),
    }

# ================= SAVE =================
rows = []

for s, d in ALL_RESULTS.items():
    rows.append([s, d["emergency"][0], d["normal"][0], d["queue"][0]])

with open(RESULT_CSV, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print("Saved:", RESULT_CSV)

# ================= ROBUSTNESS =================
def degradation(base, pert):
    return 0 if base == 0 else ((pert - base)/base)*100

robust_rows = []

for scenario in SCENARIOS:

    base_vals, pert_vals = [], []

    for seed in SCENARIO_SEEDS[scenario]:

        b = evaluate_fn(seed, scenario)
        p = evaluate_fn(seed+100, scenario)

        base_vals.append(b["emergency"])
        pert_vals.append(p["emergency"])

    b_m, p_m = np.mean(base_vals), np.mean(pert_vals)

    robust_rows.append([scenario, b_m, p_m, degradation(b_m, p_m)])

with open(ROBUST_CSV, "w", newline="") as f:
    csv.writer(f).writerows(robust_rows)

print("Saved:", ROBUST_CSV)

# ================= SENSITIVITY =================
sens_rows = []

for scenario in ["emergency_low","emergency_medium","emergency_high"]:

    vals = []

    for seed in [1,2,3]:
        out = evaluate_fn(seed, scenario)
        vals.append(out["emergency"])

    sens_rows.append([scenario, np.mean(vals)])

with open(SENS_CSV, "w", newline="") as f:
    csv.writer(f).writerows(sens_rows)

print("Saved:", SENS_CSV)

# ================= PLOTS =================
for metric in ["emergency","normal","queue"]:

    means = [ALL_RESULTS[s][metric][0] for s in SCENARIOS]

    plt.figure()
    plt.bar(SCENARIOS, means)
    plt.xticks(rotation=20)
    plt.title(f"{MODEL_NAME} {metric}")

    path = os.path.join(SAVE_DIR, f"{metric}.png")
    plt.savefig(path)
    plt.close()

    print("Saved:", path)

print("\n✅ FIXED-TIME FULL evaluation complete")