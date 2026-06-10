import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your XAI modules
from xai.run_xai_phase1 import run_phase1
from xai.run_xai_phase2 import run_phase2
from xai.run_xai_phase3 import run_phase3

RESULT_DIR = "xai/results"
os.makedirs(RESULT_DIR, exist_ok=True)

print("Running XAI phases...")

# --------------------------------------------------
# Phase 1: Integrated Gradients
# --------------------------------------------------

features, importances = run_phase1()

df = pd.DataFrame({
    "feature": features,
    "importance": importances
})

csv_path = os.path.join(RESULT_DIR, "feature_importance.csv")
df.to_csv(csv_path, index=False)

df = df.sort_values("importance")

plt.figure(figsize=(8,5))
plt.barh(df["feature"], df["importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance (Integrated Gradients)")
plt.tight_layout()

plt.savefig(os.path.join(RESULT_DIR,"feature_importance.png"), dpi=300)
plt.close()

print("Feature importance plot created")


# --------------------------------------------------
# Phase 2: Value / Advantage + Critical Timesteps
# --------------------------------------------------

timesteps, values, advantages, impacts = run_phase2()

df_va = pd.DataFrame({
    "timestep": timesteps,
    "value": values,
    "advantage": advantages
})

df_va.to_csv(os.path.join(RESULT_DIR,"value_advantage.csv"), index=False)

plt.figure(figsize=(8,5))

plt.plot(timesteps, values, label="Value(s)")
plt.plot(timesteps, advantages, label="Advantage(s,a)")

plt.xlabel("Timestep")
plt.ylabel("Score")
plt.title("Value vs Advantage Analysis")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"value_vs_advantage.png"), dpi=300)
plt.close()

print("Value vs Advantage plot created")


# Critical timesteps

df_ct = pd.DataFrame({
    "timestep": timesteps,
    "impact": impacts
})

df_ct.to_csv(os.path.join(RESULT_DIR,"critical_timesteps.csv"), index=False)

plt.figure(figsize=(8,5))

plt.bar(timesteps, impacts)

plt.xlabel("Timestep")
plt.ylabel("Decision Impact")
plt.title("Critical Decision Timesteps")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"critical_timesteps.png"), dpi=300)
plt.close()

print("Critical timestep plot created")


# --------------------------------------------------
# Phase 3: Counterfactual explanations
# --------------------------------------------------

feature_change, action_change = run_phase3()

df_cf = pd.DataFrame({
    "feature_change": feature_change,
    "action_change": action_change
})

df_cf.to_csv(os.path.join(RESULT_DIR,"counterfactual.csv"), index=False)

plt.figure(figsize=(8,5))

plt.plot(feature_change, action_change)

plt.xlabel("Feature Change")
plt.ylabel("Action Change")
plt.title("Counterfactual Explanation")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"counterfactual_analysis.png"), dpi=300)
plt.close()

print("Counterfactual plot created")


print("\nAll REAL XAI plots generated.")