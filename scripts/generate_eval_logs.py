# ============================================================
# GENERATE EVALUATION LOGS FOR XAI (FINAL, ENV-CORRECT)
# ============================================================

import os
import numpy as np
import torch

from rl.dqn_dueling import DuelingQNetwork
from rl.env import TrafficSignalEnv

# ------------------------------------------------------------
# CONFIG (MUST MATCH TRAINING)
# ------------------------------------------------------------
MODEL_PATH = "dueling_dqn_v2.pth"

STATE_DIM = 7
ACTION_DIM = 3

NUM_EPISODES = 5
MAX_STEPS = 1000

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
model = DuelingQNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ------------------------------------------------------------
# ENV (EVAL MODE)
# ------------------------------------------------------------
env = TrafficSignalEnv(
    scenario="baseline",   # or "unseen" later
    seed=0
)

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------
states = []
emergency_flags = []

# ------------------------------------------------------------
# ROLLOUT
# ------------------------------------------------------------
for ep in range(NUM_EPISODES):
    state = env.reset()

    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = torch.argmax(model(state_tensor), dim=1).item()

        next_state, reward, done, _ = env.step(action)

        # -----------------------------
        # LOGGING (POST-HOC SAFE)
        # -----------------------------
        states.append(state)
        emergency_flags.append(int(state[4] > 0))  # emergency_present

        state = next_state
        if done:
            break

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
states = np.array(states, dtype=np.float32)
emergency_flags = np.array(emergency_flags, dtype=np.int32)

np.save("logs/eval_states.npy", states)
np.save("logs/eval_emergency_flags.npy", emergency_flags)

print(f"✅ Saved {len(states)} evaluation states for XAI")
