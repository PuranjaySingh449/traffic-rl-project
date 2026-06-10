import numpy as np
import matplotlib.pyplot as plt

fixed = np.load("results_fixed.npy", allow_pickle=True).item()
ppo   = np.load("results_ppo.npy", allow_pickle=True).item()
dqn   = np.load("results_dqn.npy", allow_pickle=True).item()

labels = ["Fixed-Time", "PPO", "DQN"]

emergency = [
    fixed["emergency_wait"],
    ppo["emergency_wait"],
    dqn["emergency_wait"],
]

normal = [
    fixed["avg_normal_wait"],
    ppo["avg_normal_wait"],
    dqn["avg_normal_wait"],
]

queue = [
    fixed["avg_queue"],
    ppo["avg_queue"],
    dqn["avg_queue"],
]

# --------------------
# Plot 1: Emergency
# --------------------
plt.figure()
plt.bar(labels, emergency)
plt.title("Emergency Vehicle Waiting Time Comparison")
plt.ylabel("Waiting Time (s)")
plt.grid(True)
plt.show()

# --------------------
# Plot 2: Normal vehicles
# --------------------
plt.figure()
plt.bar(labels, normal)
plt.title("Average Normal Vehicle Waiting Time")
plt.ylabel("Waiting Time (s)")
plt.grid(True)
plt.show()

# --------------------
# Plot 3: Queue length
# --------------------
plt.figure()
plt.bar(labels, queue)
plt.title("Average Queue Length Comparison")
plt.ylabel("Vehicles")
plt.grid(True)
plt.show()
