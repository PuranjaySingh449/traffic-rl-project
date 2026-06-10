import torch
from rl.dqn_dueling import DuelingDQNAgent
from rl.apa_agent_v2 import APAAgentV2
from rl.apa_dqn_v2 import APADuelingDQN_V2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dueling():
    agent = DuelingDQNAgent(7, 3, DEVICE)
    agent.q_net.load_state_dict(
        torch.load("models/dqn_dueling_tls.pt", map_location=DEVICE)
    )
    agent.q_net.eval()
    return agent

def load_apa():
    agent = APAAgentV2(7, 3, DEVICE, model_class=APADuelingDQN_V2)
    agent.q_net.load_state_dict(
        torch.load("models/apa_dqn_v2.pth", map_location=DEVICE)
    )
    agent.q_net.eval()
    return agent