from xai.load_models import load_apa, load_dueling
from xai.integrated_gradients import integrated_gradients
from xai.value_advantage import get_value_advantage

import torch
import numpy as np

# ---------------------------
# STATE (FIXED)
# ---------------------------
state = torch.rand(7)

apa = load_apa()
duel = load_dueling()

# ---------------------------
# Integrated Gradients
# ---------------------------
ig_apa = integrated_gradients(apa.q_net, state)
ig_duel = integrated_gradients(duel.q_net, state)

print("\n=== Integrated Gradients ===")
print("APA:", ig_apa)
print("Dueling:", ig_duel)

# ---------------------------
# Value Advantage
# ---------------------------
state_np = state.numpy()

v_a, a_a = get_value_advantage(apa.q_net, state_np)
v_d, a_d = get_value_advantage(duel.q_net, state_np)

print("\n=== Value Advantage ===")
print("APA Advantage:", a_a)
print("Dueling Advantage:", a_d)