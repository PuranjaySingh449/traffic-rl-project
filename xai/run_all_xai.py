import os
import subprocess
import sys

print("\n===============================")
print("RUNNING FULL XAI PIPELINE")
print("===============================\n")

# Ensure results folder exists
os.makedirs("xai/results", exist_ok=True)

# -----------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------
steps = [
    ("Phase 1 - Integrated Gradients", "python -m xai.run_xai_phase1"),
    ("Phase 2 - Value / Advantage", "python -m xai.run_xai_phase2"),
    ("Phase 3 - Counterfactual", "python -m xai.run_xai_phase3"),
    ("Generate XAI Plots", "python -m xai.generate_xai_reports"),
]

# -----------------------------------------------------------
# Execute steps
# -----------------------------------------------------------
for name, command in steps:

    print(f"\n---- {name} ----\n")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\nERROR during {name}")
        sys.exit(1)

print("\n===============================")
print("XAI PIPELINE COMPLETE")
print("===============================\n")

print("Outputs saved in:\n")
print("xai/results/")