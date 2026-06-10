import numpy as np
from xai.load_models import load_apa, load_dueling

EMERGENCY_INDEX = 0  # adjust if needed

def test_emergency_sensitivity(state):

    s_no = state.copy()
    s_no[EMERGENCY_INDEX] = 0

    s_yes = state.copy()
    s_yes[EMERGENCY_INDEX] = 1

    apa = load_apa()
    duel = load_dueling()

    q_no_apa = apa.q_net(
        torch.FloatTensor(s_no).unsqueeze(0)
    ).detach().numpy()

    q_yes_apa = apa.q_net(
        torch.FloatTensor(s_yes).unsqueeze(0)
    ).detach().numpy()

    q_no_duel = duel.q_net(
        torch.FloatTensor(s_no).unsqueeze(0)
    ).detach().numpy()

    q_yes_duel = duel.q_net(
        torch.FloatTensor(s_yes).unsqueeze(0)
    ).detach().numpy()

    return q_no_apa, q_yes_apa, q_no_duel, q_yes_duel