# SternhalmaMARL

Multi-agent experimentation scaffold for [SternhalmaEnv](https://github.com/masarwy/SternhalmaEnv) (Chinese Checkers), with Torch RLlib training and evaluation pipelines.

## Current Status

This repository is now a working v0:

- Mask-aware baseline agents (`random`, `heuristic`)
- RLlib Torch self-play PPO (`training/self_play/train_ppo.py`)
- RLlib Torch multi-agent PPO with MAPPO/IPPO modes (`training/multiagent/train_mappo.py`)
- Round-robin tournament + Elo (`scripts/run_tournament.py`)
- Training metrics plotting from JSONL logs
- Smoke tests for agents and training entrypoints

Notes:
- `MAPPO` mode here means shared-policy multi-agent PPO (MAPPO-style setup).
- `IPPO` mode means independent per-agent policies.

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
  max_agent_steps: 3000
  reward_mode: potential_shaped
  reward_scale: 0.01
  reward_clip_abs: 5.0
  render_mode: null

training_config:
  num_iterations: 100
  checkpoint_every: 10
  seed: 42
  stop_reward: null

rllib_config:
  framework: torch
  use_new_api_stack: true
  num_gpus: 0.0
  num_env_runners: 2
  rollout_fragment_length: 400
  train_batch_size: 8000
  minibatch_size: 512
  num_epochs: 5
  lr: 1e-5
  entropy_coeff: 0.003
  gamma: 0.99
  lambda: 0.95

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
- [ ] Hyperparameter sweeps
- [ ] Scaling experiments (`num_players`, `board_diagonal`)
- [ ] Curriculum learning

## License

MIT License. See `LICENSE`.
