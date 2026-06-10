import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

controllers = ["Fixed-Time", "DQN", "PPO"]

emergency_wait = [1378.0, 0.0, 0.0]
avg_normal_wait = [3.50, 0.04, 0.09]
avg_queue = [6.60, 1.17, 1.48]

x = np.arange(len(controllers))
width = 0.25

# -------------------------
# Emergency waiting time
# -------------------------
plt.figure()
plt.bar(controllers, emergency_wait)
plt.ylabel("Emergency Waiting Time (s)")
plt.title("Emergency Vehicle Delay Comparison")
plt.grid(axis="y")
plt.savefig("plots/emergency_wait_comparison.png")
plt.show()

# -------------------------
# Normal waiting time
# -------------------------
plt.figure()
plt.bar(controllers, avg_normal_wait)
plt.ylabel("Avg Normal Waiting Time (s)")
plt.title("Normal Vehicle Delay Comparison")
plt.grid(axis="y")
plt.savefig("plots/normal_wait_comparison.png")
plt.show()

# -------------------------
# Queue length
# -------------------------
plt.figure()
plt.bar(controllers, avg_queue)
plt.ylabel("Average Queue Length")
plt.title("Queue Length Comparison")
plt.grid(axis="y")
plt.savefig("plots/queue_length_comparison.png")
plt.show()

print("✅ All comparison plots saved in /plots")
