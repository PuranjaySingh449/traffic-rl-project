import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from rl.evaluate_ppo import evaluate_ppo_once

# =========================
# CONFIG
# =========================
MODEL_NAME = "ppo"
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
    "baseline": [1, 2, 3],
    "low_demand": [4, 5, 6],
    "high_demand": [7, 8, 9],
    "emergency_stress": [10, 11, 12],
    "unseen": [13, 14, 15],
}

# =========================
# SEED FUNCTION
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_fn(seed, scenario):
    return evaluate_ppo_once(seed, scenario)

# =========================
# MAIN EVALUATION
# =========================
ALL_RESULTS = {}

for scenario in SCENARIOS:

    emergency_vals = []
    normal_vals = []
    queue_vals = []

    for seed in SCENARIO_SEEDS[scenario]:

        set_seed(seed)

        out = evaluate_fn(seed, scenario)

        emergency_vals.append(out["emergency"])
        normal_vals.append(out["normal"])
        queue_vals.append(out["queue"])

    ALL_RESULTS[scenario] = {
        "emergency": (np.mean(emergency_vals), np.std(emergency_vals)),
        "normal": (np.mean(normal_vals), np.std(normal_vals)),
        "queue": (np.mean(queue_vals), np.std(queue_vals)),
    }

# =========================
# SAVE RESULTS
# =========================
rows = []

for scenario, data in ALL_RESULTS.items():
    rows.append([
        scenario,
        data["emergency"][0],
        data["normal"][0],
        data["queue"][0],
    ])

with open(RESULT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Scenario", "Emergency", "Normal", "Queue"])
    writer.writerows(rows)

print("Saved:", RESULT_CSV)

# =========================
# ROBUSTNESS
# =========================
def degradation_percent(base, perturbed):
    if base == 0:
        return 0
    return ((perturbed - base) / base) * 100

robust_rows = []

for scenario in SCENARIOS:

    base_vals = []
    pert_vals = []

    for seed in SCENARIO_SEEDS[scenario]:

        base = evaluate_fn(seed, scenario)
        pert = evaluate_fn(seed + 100, scenario)

        base_vals.append(base["emergency"])
        pert_vals.append(pert["emergency"])

    base_mean = np.mean(base_vals)
    pert_mean = np.mean(pert_vals)

    robust_rows.append([
        scenario,
        base_mean,
        pert_mean,
        degradation_percent(base_mean, pert_mean)
    ])

with open(ROBUST_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Scenario", "Base", "Perturbed", "Degradation%"])
    writer.writerows(robust_rows)

print("Saved:", ROBUST_CSV)

# =========================
# SENSITIVITY
# =========================
SENSITIVITY_SCENARIOS = [
    "emergency_low",
    "emergency_medium",
    "emergency_high"
]

sens_rows = []

for scenario in SENSITIVITY_SCENARIOS:

    vals = []

    for seed in [1, 2, 3]:
        out = evaluate_fn(seed, scenario)
        vals.append(out["emergency"])

    sens_rows.append([scenario, np.mean(vals)])

with open(SENS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Scenario", "EmergencyDelay"])
    writer.writerows(sens_rows)

print("Saved:", SENS_CSV)

# =========================
# PLOTS
# =========================
for metric in ["emergency", "normal", "queue"]:

    means = [ALL_RESULTS[s][metric][0] for s in SCENARIOS]

    plt.figure()
    plt.bar(SCENARIOS, means)
    plt.xticks(rotation=20)
    plt.ylabel(metric)
    plt.title(f"{MODEL_NAME} - {metric}")

    path = os.path.join(SAVE_DIR, f"{metric}.png")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("Saved:", path)

print(f"\n✅ {MODEL_NAME.upper()} FULL evaluation complete")