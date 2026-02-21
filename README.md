# SternhalmaMARL

Multi-agent experimentation scaffold for [SternhalmaEnv](https://github.com/masarwy/SternhalmaEnv) (Chinese Checkers), with runnable baseline training/evaluation pipelines.

## Current Status

This repository is now a working v0:

- Mask-aware baseline agents (`random`, `heuristic`)
- Self-play training runner (`training/self_play/train_ppo.py`)
- Multi-agent training runner (`training/multiagent/train_mappo.py`)
- Round-robin tournament + Elo (`scripts/run_tournament.py`)
- Training metrics plotting from JSONL logs
- Smoke tests for agents and training entrypoints

Note: the current training entrypoints are baseline-policy runners (not learned RLlib PPO/MAPPO yet).

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
  render_mode: null

training_config:
  num_episodes: 40
  max_steps_per_episode: 300
  checkpoint_every: 10
  seed: 42
```

## Outputs

Training runs write:

- `logs/metrics.jsonl`: per-episode metrics
- `checkpoints/latest.json`: latest checkpoint metadata
- `summary.json`: run-level aggregate summary

Tournament runs write:

- `ratings` (Elo per agent)
- pairwise match statistics
- timestamp and run metadata

## Development

Run tests:

```bash
pytest -q
```

## Roadmap

- [x] Project scaffolding
- [x] Baseline agents (random, heuristic)
- [x] Tournament evaluation with Elo
- [x] Training visualization from JSONL metrics
- [ ] RLlib-backed self-play PPO training
- [ ] RLlib-backed MAPPO/IPPO training
- [ ] Hyperparameter sweeps
- [ ] Scaling experiments (`num_players`, `board_diagonal`)
- [ ] Curriculum learning

## License

MIT License. See `LICENSE`.
