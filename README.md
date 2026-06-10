# Traffic Signal Control with Reinforcement Learning

Adaptive traffic-signal control for a single intersection, trained and evaluated in the
[SUMO](https://www.eclipse.org/sumo/) microscopic traffic simulator. The project compares
several reinforcement-learning agents (and classical baselines) under normal traffic and
under **emergency-vehicle** scenarios, and includes an explainability (XAI) suite to analyse
agent decisions.

## Overview

A learning agent controls one traffic light. At each step it observes the local traffic
state and chooses how to act on the signal.

- **State (7-dim):** `[queue_NS, queue_EW, avg_wait_NS, avg_wait_EW, emergency_present, emergency_distance, current_phase]`
- **Actions (3):** `0` = keep current phase · `1` = switch to next phase · `2` = extend current phase duration
- **Reward:** weighted combination of queue length, vehicle waiting time, phase-switching
  penalty, and a strong bonus for clearing emergency vehicles (reward weights `alpha=10, beta=1, gamma=0.5, delta=1` in `rl/env.py`).

Multiple traffic scenarios are supported via SUMO route files (baseline, low/high demand,
unseen, and emergency stress variants) — see `env.ROUTE_MAP` in `rl/env.py`.

## Agents & baselines

| Method | Files |
|--------|-------|
| Fixed-time controller (baseline) | `baselines/run_fixed_time.py`, `rl/fixed_time.py` |
| Rule-based controller (baseline) | `baselines/run_rule_based.py` |
| DQN | `rl/dqn.py`, `rl/train_dqn.py` |
| Dueling DQN (+ v2) | `rl/dqn_dueling.py`, `rl/train_dqn_dueling.py`, `rl/train_dqn_dueling_v2.py` |
| PPO | `rl/ppo.py`, `rl/train_ppo.py` |
| APA-DQN (+ v2) | `rl/apa_dqn.py`, `rl/apa_agent.py`, `rl/train_apa_dqn.py` |
| PAE-DQN | `rl/pae_dqn.py`, `rl/train_pae_dqn.py` |
| RCD-DQN | `rl/rcd_dqn.py`, `rl/rcd_agent.py`, `rl/train_rcd_dqn.py` |
| HEA selector | `rl/hea_selector.py`, `rl/hea_agent.py`, `rl/train_hea.py` |

## Repository structure

```
.
├── rl/              # Environments, agents, training & evaluation scripts
├── baselines/       # Fixed-time and rule-based controllers
├── sumo/            # SUMO network, config, and route files (scenarios)
├── xai/             # Explainability: integrated gradients, counterfactuals, etc.
├── scripts/         # Helper scripts (e.g. generate evaluation logs)
├── utils/           # Seeding and shared utilities
├── models/          # Trained model checkpoints (.pth / .pt)
├── logs/            # Training reward curves and evaluation arrays (.npy / .csv)
├── eval_results/    # Per-method evaluation outputs (CSV + plots)
├── plots/           # Generated comparison figures
└── setup.py
```

## Requirements

- Python 3.12
- [SUMO](https://www.eclipse.org/sumo/) installed, with the `SUMO_HOME` environment variable
  set and `sumo` / `sumo-gui` on your `PATH`
- Python packages: `traci`, `numpy`, `torch`, `matplotlib`, `pandas`
  (the `traci` Python bindings ship with SUMO)

## Setup

```bash
# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# install the project (editable)
pip install -e .
pip install torch numpy matplotlib pandas
```

Make sure SUMO is installed and `SUMO_HOME` is set, e.g. on Windows:

```powershell
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

## Usage

Train an agent (examples — run from the repo root):

```bash
python rl/train_dqn.py
python rl/train_ppo.py
python rl/train_apa_dqn.py
```

Evaluate a trained agent and produce metrics/plots:

```bash
python rl/evaluate_dqn.py
python rl/eval_dqn_full.py
```

Compare methods:

```bash
python rl/plot_comparison.py
```

Run the explainability analysis:

```bash
python xai/run_all_xai.py
```

> Training and evaluation launch a SUMO simulation through `traci`. Use the `sumo` binary
> for headless runs or `sumo-gui` to watch the simulation; this is configured via the
> `sumo_binary` argument when constructing `TrafficSignalEnv` in `rl/env.py`.

## Results

Evaluation outputs are organised per method under `eval_results/` (normal traffic, emergency,
queue, robustness and sensitivity), with aggregate comparison figures in `plots/`.

## License

No license has been specified for this project.
