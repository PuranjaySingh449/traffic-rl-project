import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = r"D:\traffic rl project\traffic rl project\eval_results"

models = {
    "APA-DQN-V2": os.path.join(BASE, "apa_dqn_v2"),
    "DQN": os.path.join(BASE, "dqn"),
    "Dueling-DQN": os.path.join(BASE, "dueling_dqn"),
    "PAE-DQN": os.path.join(BASE, "pae_dqn")
}

data = []
labels = []

for model_name, path in models.items():
    files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
    scores = []

    for file in files:
        df = pd.read_csv(os.path.join(path, file))

        # Select metric
        if "reward" in df.columns:
            series = df["reward"]
        elif "score" in df.columns:
            series = df["score"]
        else:
            series = df.iloc[:, -1]

        # ---------------------------------------
        # FINAL FIX: STABLE REGION AVERAGE
        # ---------------------------------------
        if len(series) >= 100:
            value = series.tail(100).mean()
        elif len(series) >= 50:
            value = series.tail(50).mean()
        else:
            value = series.mean()

        scores.append(value)

    data.append(scores)
    labels.append(model_name)

# ---------------------------------------
# PLOT
# ---------------------------------------
plt.figure(figsize=(6, 4))

box = plt.boxplot(data, labels=labels, patch_artist=True)

for patch in box['boxes']:
    patch.set_facecolor('lightgray')

plt.title("Robustness Across Random Seeds")
plt.ylabel("Average Performance Score")

plt.xticks(rotation=20)
plt.grid(axis='y')
plt.tight_layout()

plt.savefig("robustness.png", dpi=300, bbox_inches='tight')
plt.show()