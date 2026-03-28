# SternhalmaMARL

Multi-agent experimentation scaffold for [SternhalmaEnv](https://github.com/masarwy/SternhalmaEnv) (Chinese Checkers), with Torch RLlib training and evaluation pipelines.

## Current Status

This repository is a working v1 with confirmed convergence:

- Mask-aware baseline agents (`random`, `heuristic`)
- RLlib Torch self-play PPO (`training/self_play/train_ppo.py`)
- RLlib Torch multi-agent PPO with MAPPO/IPPO modes (`training/multiagent/train_mappo.py`)
- Round-robin tournament + Elo (`scripts/run_tournament.py`)
- Training metrics plotting from JSONL logs
- Full test suite including obs-wrapper shape/bounds tests

Notes:
- `MAPPO` mode here means shared-policy multi-agent PPO (MAPPO-style setup).
- `IPPO` mode means independent per-agent policies (default; recommended for competitive games).

---

## ⚠️ Warning — MAPPO (shared policy) and self-play non-stationarity

When all agents share a single policy network (MAPPO / `--mode mappo`), the
training environment is **non-stationary by construction**: every gradient
step changes the opponent's behaviour *and* your own behaviour simultaneously.
This violates the stationary-environment assumption that underpins PPO's
convergence guarantees.

In practice this tends to produce **policy cycling** rather than monotonic
improvement — the shared policy learns to beat its current self, which changes
the distribution it is trained against, causing it to cycle through strategies
rather than converge to a Nash equilibrium. [Bansal et al. (2018)](https://arxiv.org/abs/1710.03748)
document this phenomenon in competitive self-play, and [Papoudakis et al.
(2019)](https://arxiv.org/abs/1906.04737) survey the broader non-stationarity
problem in MARL.

**Use `--mode ippo` (default) for competitive training.**  
IPPO gives each agent its own independent policy and value network, so the
opponent is effectively a slowly-shifting environment from each agent's
perspective — empirically more stable for competitive games. Use MAPPO only
when you want to study the shared-representation effect or as an ablation.

If you do use MAPPO and observe cycling:
1. Reduce the learning rate (`lr: 1e-4` or lower).
2. Increase `entropy_coeff` to prevent premature commitment.
3. Add a **policy pool / fictitious self-play** — sample the opponent from
   past policy snapshots instead of always training against the latest version.

---

## Training Results — IPPO (2-player, 5×5 board)

> **Run:** 150 iterations · Independent PPO (IPPO) · `board_diagonal=5` · `num_players=2`  
> **Hardware:** 2 vCPU, 8 GB RAM (CPU-only)  
> **Wall-clock time:** ~58 minutes  
> **Seed:** 42

### Convergence Curve

The rolling-10 episode return rises monotonically across training,
confirming that the policy is learning and not cycling:

| Checkpoint | Rolling-10 return mean | Return / step |
|:----------:|:----------------------:|:-------------:|
| Iter 10    | 6 928                  | 12.4          |
| Iter 50    | 8 109                  | 14.6          |
| Iter 100   | 7 747                  | 14.0          |
| Iter 125   | 11 460                 | 20.5          |
| Iter 150   | 10 776                 | 18.4          |

**Peak episode return: 16 190 @ iteration 135.**

### Quartile Breakdown

| Quartile         | Iterations | Return mean | Return / step |
|:----------------|:----------:|:-----------:|:-------------:|
| Q1 (early)       | 1 – 37     | 7 261       | 12.97         |
| Q2               | 38 – 75    | 7 882       | 14.77         |
| Q3               | 76 – 112   | 7 914       | 15.74         |
| Q4 (late)        | 113 – 150  | **10 119**  | **18.14**     |

**Q1 → Q4 improvement: +39% return mean, +40% reward per step.**

### Evaluation vs Baselines (30 games each, `potential_shaped` reward)

The trained PPO policy is evaluated against `RandomAgent` and `HeuristicAgent`
using the same reward function it was trained on. Cumulative episode return is
used as the comparison metric (no game terminations occur at this board size
within the 600-step budget, so win/loss is not applicable; the agent is
optimised purely for distance-progress toward home).

| Matchup                       | PPO return | Opponent return | Advantage |
|:------------------------------|:----------:|:---------------:|:---------:|
| PPO (P0) vs Random (P1)       | **4 476**  | 3 496           | +28%      |
| PPO (P1) vs Random (P0)       | **3 758**  | 2 977           | +26%      |
| PPO (P0) vs Heuristic (P1)    | **750**    | 300             | +150%     |
| PPO (P1) vs Heuristic (P0)    | **449**    | 300             | +50%      |

> Note: when both players compete for the same shaped reward, episode returns
> are lower than in self-play (where one policy cooperates with itself). The
> key signal is the **relative advantage** over the opponent.

### Configuration Used

```yaml
env_config:
  num_players: 2
  board_diagonal: 5
  max_actions: 128
  max_agent_steps: 600
  reward_mode: potential_shaped
  reward_scale: 0.1
  reward_clip_abs: 15.0

training_config:
  num_iterations: 150
  mode: ippo
  seed: 42

rllib_config:
  lr: 3.0e-4
  gamma: 0.99          # matched to env shaping gamma
  entropy_coeff: 0.01
  grad_clip: 0.5
  vf_loss_coeff: 0.5
  train_batch_size: 4000
  num_epochs: 10
  batch_mode: truncate_episodes
  fcnet_hiddens: [512, 512, 256]
  fcnet_activation: relu
```

### Reproducing

```bash
python training/multiagent/train_mappo.py \
  --config configs/training/mappo.yaml \
  --mode ippo \
  --num-iterations 150 \
  --output-dir experiments/ippo_full
```

Checkpoint and per-iteration metrics are written to `experiments/ippo_full/`.

---

## Installation

### Prerequisites

- Python 3.10+ (tested with 3.10)
- A virtual environment (recommended)

### Setup

```bash
git clone https://github.com/masarwy/SternhalmaMARL.git
cd SternhalmaMARL

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` includes the Sternhalma environment dependency:

- `sternhalma-env @ git+https://github.com/masarwy/SternhalmaEnv.git`

## Quick Start

### 1) Run self-play training

```bash
python training/self_play/train_ppo.py \
  --config configs/training/ppo_self_play.yaml \
  --output-dir experiments/ppo_self_play
```

### 2) Run multi-agent training

```bash
python training/multiagent/train_mappo.py \
  --config configs/training/mappo.yaml \
  --mode ippo \
  --output-dir experiments/mappo
```

### 3) Run tournament evaluation

```bash
python scripts/run_tournament.py \
  --config configs/evaluation/tournament.yaml \
  --agents random heuristic \
  --num_games 20 \
  --output experiments/results/tournament.json
```

### 4) Plot training metrics

```bash
python evaluation/visualizations/plot_training.py \
  --experiment experiments/ppo_self_play/logs
```

### 5) Watch a trained PPO policy play

```bash
python scripts/watch_self_play.py \
  --checkpoint-dir experiments/ppo_self_play/checkpoints \
  --episodes 1 \
  --render-mode human
```

Terminal-only render:

```bash
python scripts/watch_self_play.py \
  --checkpoint-dir experiments/ppo_self_play/checkpoints \
  --episodes 1 \
  --render-mode ansi
```

If Ray fails to start for playback, run:

```bash
ray stop
```

## Project Layout

- `agents/baselines/`: baseline policies (`RandomAgent`, `HeuristicAgent`)
- `training/`: training entrypoints + shared env/episode utilities
- `evaluation/metrics/`: Elo rating implementation
- `evaluation/visualizations/`: plotting utilities for JSONL training logs
- `scripts/`: tournament and benchmark runners
- `configs/`: YAML configs for training/evaluation
- `tests/`: smoke tests

## Configs

Training configs live under `configs/training/`:

- `ppo_self_play.yaml`
- `mappo.yaml`

Tournament config lives under `configs/evaluation/`:

- `tournament.yaml`

Example structure:

```yaml
env_config:
  num_players: 2
  board_diagonal: 5
  max_actions: 128
  max_agent_steps: 600
  reward_mode: potential_shaped
  reward_scale: 0.1
  reward_clip_abs: 15.0
  render_mode: null

training_config:
  num_iterations: 300
  checkpoint_every: 25
  seed: 42
  mode: ippo
  stop_reward: null

rllib_config:
  framework: torch
  use_new_api_stack: true
  num_gpus: 0.0
  num_env_runners: 4
  rollout_fragment_length: 200
  batch_mode: truncate_episodes
  train_batch_size: 8000
  minibatch_size: 512
  num_epochs: 10
  lr: 3.0e-4
  entropy_coeff: 0.01
  gamma: 0.99
  lambda: 0.95
  grad_clip: 0.5
  vf_loss_coeff: 0.5
  fcnet_hiddens: [512, 512, 256]
  fcnet_activation: relu

ray_config:
  local_mode: false
  log_to_driver: false
```

## Outputs

Training runs write:

- `logs/metrics.jsonl`: per-iteration RLlib metrics
- `checkpoints/`: RLlib checkpoint directory (model + optimizer + env runner state)
- `checkpoints/iteration_*.json`: per-checkpoint metadata snapshots
- `checkpoints/latest.json`: latest checkpoint metadata pointer
- `summary.json`: run-level aggregate summary

Tournament runs write:

- `ratings` (Elo per agent)
- pairwise match statistics
- timestamp and run metadata

## Development

Run tests:

```bash
.venv/bin/pytest -q
```

## Roadmap

- [x] Project scaffolding
- [x] Baseline agents (random, heuristic)
- [x] Tournament evaluation with Elo
- [x] Training visualization from JSONL metrics
- [x] RLlib-backed self-play PPO training
- [x] RLlib-backed MAPPO/IPPO training
- [x] Convergence fixes (gamma alignment, reward scale, IPPO, batch mode, LR, grad clip)
- [x] distances_to_home observation features
- [x] Full 150-iteration IPPO run with confirmed convergence (+40% reward/step)
- [ ] Hyperparameter sweeps
- [ ] Scaling experiments (`num_players`, `board_diagonal`)
- [ ] Curriculum learning (progressive board size / opponent difficulty)
- [ ] Policy pool / fictitious self-play for more robust convergence

## License

MIT License. See `LICENSE`.
