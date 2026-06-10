import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# YOUR MODEL RESULTS (PUT YOUR FINAL VALUES HERE)
# ============================================================

models = ["DQN", "Dueling", "PPO", "PAE", "APA"]

# Replace with your actual averaged values
emergency = [275, 77, 430, 480, 120]
normal    = [3.9, 13.5, 19.1, 0.08, 0.07]
queue     = [11.8, 9.7, 14.0, 4.3, 3.4]

# ============================================================
# 1. MAIN MODEL COMPARISON
# ============================================================

x = np.arange(len(models))

plt.figure()
plt.bar(x - 0.2, emergency, width=0.2, label="Emergency")
plt.bar(x, normal, width=0.2, label="Normal")
plt.bar(x + 0.2, queue, width=0.2, label="Queue")

plt.xticks(x, models)
plt.title("Model Performance Comparison")
plt.ylabel("Metric Value")
plt.legend()

plt.savefig("xai/results/model_comparison.png")
plt.close()


# ============================================================
# 2. SCENARIO-WISE (APA ONLY)
# ============================================================

scenarios = ["baseline", "low", "high", "stress", "unseen"]

apa_emergency = [20, 5, 120, 230, 25]
apa_queue     = [1.3, 0.6, 3.4, 1.5, 1.45]

x = np.arange(len(scenarios))

plt.figure()
plt.plot(x, apa_emergency, marker='o', label="Emergency")
plt.plot(x, apa_queue, marker='o', label="Queue")

plt.xticks(x, scenarios)
plt.title("APA-DQN-V2 Scenario Performance")
plt.legend()

plt.savefig("xai/results/scenario_comparison.png")
plt.close()


# ============================================================
# 3. TRADE-OFF GRAPH (VERY IMPORTANT)
# ============================================================

plt.figure()

plt.scatter(emergency, queue)

for i, model in enumerate(models):
    plt.text(emergency[i], queue[i], model)

plt.xlabel("Emergency Delay")
plt.ylabel("Queue Length")
plt.title("Trade-off Analysis")

plt.savefig("xai/results/tradeoff.png")
plt.close()


print("✅ All paper graphs generated in xai/results/")