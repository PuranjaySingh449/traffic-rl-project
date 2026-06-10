from rl.env import TrafficSignalEnv
import numpy as np

env = TrafficSignalEnv()
state = env.reset()

done = False
while not done:
    action = np.random.choice([0, 1, 2])
    state, reward, done, _ = env.step(action)

print("Environment test completed.")
