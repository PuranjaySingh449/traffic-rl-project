import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
# Import evaluators
# =========================
from rl.evaluate_dqn import evaluate_dqn_once
from rl.evaluate_ppo import evaluate_ppo_once
from rl.fixed_time import evaluate_fixed_time_once
from rl.evaluate_dqn_dueling import evaluate_dqn_dueling_once
from rl.evaluate_dqn_dueling_v2 import evaluate_dqn_dueling_v2_once

# =========================
# Seed control
# =========================
def set_global_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Robustness helpers
# =========================
def degradation_percent(base, perturbed):

    if base == 0:
        return 0

    return ((perturbed - base) / base) * 100


def evaluate_with_noise(eval_fn, seed, scenario, strength=0.1):

    pert_seed = seed + int(1000 * strength)

    out = eval_fn(pert_seed, scenario)

    noise = np.random.uniform(-strength, strength)

    return {
        "emergency": out.get("emergency", 0) * (1 + noise),
        "normal": out.get("normal", 0) * (1 + noise),
        "queue": out.get("queue", 0) * (1 + noise)
    }


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_DIR = os.path.join(BASE_DIR, "eval_seeds_png")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULT_CSV = os.path.join(SAVE_DIR, "comparison_results.csv")
SEED_CSV = os.path.join(SAVE_DIR, "seed_results.csv")
ROBUST_CSV = os.path.join(SAVE_DIR, "robustness_results.csv")
SENSITIVITY_CSV = os.path.join(SAVE_DIR, "emergency_sensitivity.csv")
NORMALIZED_CSV = os.path.join(SAVE_DIR, "normalized_results.csv")


# =========================
# Seeds
# =========================
SEEDS = list(range(10))


# =========================
# Main Scenarios
# =========================
SCENARIOS = [
    "baseline",
    "low_demand",
    "high_demand",
    "emergency_stress",
    "unseen"
]


# =========================
# Emergency Sensitivity
# =========================
EMERGENCY_SENSITIVITY = [
    "emergency_low",
    "emergency_medium",
    "emergency_high"
]


# =========================
# Controllers
# =========================
CONTROLLERS = {

    "Fixed-Time": evaluate_fixed_time_once,
    "DQN": evaluate_dqn_once,
    "Dueling-DQN": evaluate_dqn_dueling_once,
    "Dueling-DQN-V2": evaluate_dqn_dueling_v2_once,
    "PPO": evaluate_ppo_once,
}


# =========================
# Plot helper
# =========================
def plot_metric(results, metric, ylabel):

    for scenario, data in results.items():

        names = list(data.keys())

        means = [data[n][metric][0] for n in names]
        stds = [data[n][metric][1] for n in names]

        x = np.arange(len(names))

        plt.figure(figsize=(7,4))

        plt.bar(x, means, yerr=stds, capsize=6)

        plt.xticks(x, names, rotation=15)

        plt.ylabel(ylabel)
        plt.title(f"{metric} — {scenario}")

        plt.tight_layout()

        path = os.path.join(SAVE_DIR, f"{scenario}_{metric}.png")

        plt.savefig(path)
        plt.close()

        print("Saved:", path)


# =========================
# Main evaluation
# =========================
ALL_RESULTS = []
SEED_RESULTS = []

PLOT_DATA = {}

for scenario in SCENARIOS:

    print(f"\n========== {scenario.upper()} ==========")

    scenario_plot = {}

    for name, eval_fn in CONTROLLERS.items():

        emergency_vals = []
        normal_vals = []
        queue_vals = []

        for seed in SEEDS:

            set_global_seed(seed)

            out = eval_fn(seed, scenario)

            E = out.get("emergency", 0)
            N = out.get("normal", 0)
            Q = out.get("queue", 0)

            emergency_vals.append(E)
            normal_vals.append(N)
            queue_vals.append(Q)

            SEED_RESULTS.append([scenario, name, seed, E, N, Q])

        em_m, em_s = np.mean(emergency_vals), np.std(emergency_vals)
        no_m, no_s = np.mean(normal_vals), np.std(normal_vals)
        q_m, q_s = np.mean(queue_vals), np.std(queue_vals)

        scenario_plot[name] = {
            "emergency": (em_m, em_s),
            "normal": (no_m, no_s),
            "queue": (q_m, q_s)
        }

        ALL_RESULTS.append([
            scenario, name,
            em_m, em_s,
            no_m, no_s,
            q_m, q_s
        ])

        print(name, "E:", em_m, "N:", no_m, "Q:", q_m)

    PLOT_DATA[scenario] = scenario_plot


# =========================
# Save main CSV
# =========================
with open(RESULT_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Scenario","Controller",
        "EmergencyMean","EmergencyStd",
        "NormalMean","NormalStd",
        "QueueMean","QueueStd"
    ])

    writer.writerows(ALL_RESULTS)

print("Saved:", RESULT_CSV)


# =========================
# Save seed CSV
# =========================
with open(SEED_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Scenario","Controller","Seed",
        "Emergency","Normal","Queue"
    ])

    writer.writerows(SEED_RESULTS)

print("Saved:", SEED_CSV)


# =========================
# Robustness experiment
# =========================
ROBUST_RESULTS = []

for scenario in SCENARIOS:

    for name, eval_fn in CONTROLLERS.items():

        base = []
        pert = []

        for seed in SEEDS:

            b = eval_fn(seed, scenario)
            p = evaluate_with_noise(eval_fn, seed, scenario)

            base.append(b["emergency"])
            pert.append(p["emergency"])

        b_m = np.mean(base)
        p_m = np.mean(pert)

        deg = degradation_percent(b_m, p_m)

        ROBUST_RESULTS.append([scenario, name, b_m, p_m, deg])

with open(ROBUST_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Scenario","Controller",
        "BaseEmergency","PertEmergency","DegradationPercent"
    ])

    writer.writerows(ROBUST_RESULTS)

print("Saved:", ROBUST_CSV)


# =========================
# Emergency sensitivity
# =========================
SENSITIVITY_RESULTS = []

for scenario in EMERGENCY_SENSITIVITY:

    for name, eval_fn in CONTROLLERS.items():

        delays = []

        for seed in SEEDS:

            out = eval_fn(seed, scenario)

            delays.append(out["emergency"])

        mean_delay = np.mean(delays)

        SENSITIVITY_RESULTS.append([scenario, name, mean_delay])

with open(SENSITIVITY_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Scenario","Controller","EmergencyDelay"
    ])

    writer.writerows(SENSITIVITY_RESULTS)

print("Saved:", SENSITIVITY_CSV)


# =========================
# Normalized improvement
# =========================
NORMALIZED_RESULTS = []

for scenario, data in PLOT_DATA.items():

    baseline = data["Fixed-Time"]["emergency"][0]

    for controller, metrics in data.items():

        delay = metrics["emergency"][0]

        improvement = ((baseline - delay) / baseline) * 100

        NORMALIZED_RESULTS.append([scenario, controller, improvement])

with open(NORMALIZED_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Scenario","Controller","EmergencyImprovementPercent"
    ])

    writer.writerows(NORMALIZED_RESULTS)

print("Saved:", NORMALIZED_CSV)


# =========================
# Plots
# =========================
plot_metric(PLOT_DATA,"emergency","Emergency Wait")
plot_metric(PLOT_DATA,"normal","Normal Wait")
plot_metric(PLOT_DATA,"queue","Queue Length")

print("\nEvaluation complete.")