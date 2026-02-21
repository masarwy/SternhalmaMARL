# SternhalmaMARL

Multi-Agent Reinforcement Learning experiments on [SternhalmaEnv](https://github.com/masarwy/SternhalmaEnv) (Chinese Checkers) using RLlib.

## Overview

This project trains and benchmarks various MARL algorithms on the Sternhalma environment:

- **Self-Play PPO**: Single shared policy trained via self-play  
- **MAPPO**: Multi-Agent PPO with centralized critic  
- **IPPO**: Independent PPO agents  
- **Baseline agents**: Random and heuristic policies for comparison  

## Features

- RLlib integration with PettingZoo  
- Action masking support for invalid moves  
- Self-play training pipeline  
- Multi-agent cooperative/competitive training  
- Tournament evaluation system with Elo ratings  
- Training visualization and metrics tracking  

## Installation

### Prerequisites

- Python 3.8+  
- [SternhalmaEnv](https://github.com/masarwy/SternhalmaEnv) installed or installable from Git  

### Setup

Clone the repository and create a virtual environment:

    git clone https://github.com/masarwy/SternhalmaMARL.git
    cd SternhalmaMARL

    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies:

    pip install -e .
    pip install git+https://github.com/masarwy/SternhalmaEnv.git

## Quick Start

### Train Self-Play PPO

    python training/self_play/train_ppo.py --config configs/training/ppo_self_play.yaml

### Run Tournament Evaluation

    python scripts/run_tournament.py --agents random heuristic ppo_checkpoint

### Visualize Training Results

    python evaluation/visualizations/plot_training.py --experiment experiments/logs/ppo_self_play

## Configuration

Training hyperparameters are defined in YAML files under `configs/training/`:

- `ppo_self_play.yaml`: Self-play PPO configuration  
- `mappo.yaml`: Multi-Agent PPO configuration  

Example config structure:

    env_config:
      num_players: 2
      board_diagonal: 5
      max_actions: 512

    training_config:
      num_workers: 4
      train_batch_size: 4000
      sgd_minibatch_size: 128
      num_sgd_iter: 10
      lr: 5e-5

## Algorithms

### Self-Play PPO

Single shared policy plays against itself. Best for symmetric games like Sternhalma.

    python training/self_play/train_ppo.py

### MAPPO (Multi-Agent PPO)

Centralized critic with decentralized actors. Good for cooperative or mixed scenarios.

    python training/multiagent/train_mappo.py

### Baseline Agents

- **Random Agent**: Uniformly samples from valid moves.  
- **Heuristic Agent**: Greedy policy moving pieces toward target home.  

These are implemented under `agents/baselines/`.

## Evaluation

### Tournament Mode

Run round-robin tournament between trained agents:

    python scripts/run_tournament.py \
      --agents random heuristic ppo_100k ppo_500k mappo_100k \
      --num_games 50 \
      --output experiments/results/tournament_2026_02_21.json

### Metrics

- Elo rating  
- Win rate  
- Average episode length  
- Convergence curves  

## Development

### Run Tests

    pytest tests/ -v

### Format Code

    black .
    isort .

### Type Checking

    mypy agents/ training/ evaluation/

## Experiments & Results

Training logs and checkpoints are saved to `experiments/`:

- `checkpoints/`: Model weights  
- `logs/`: TensorBoard logs, CSV metrics  
- `results/`: Tournament JSON, Elo rankings, plots  

View TensorBoard:

    tensorboard --logdir experiments/logs

## Roadmap

- [x] Project scaffolding  
- [ ] Self-play PPO implementation  
- [ ] MAPPO implementation  
- [ ] Baseline agents (random, heuristic)  
- [ ] Tournament evaluation system  
- [ ] Elo rating calculations  
- [ ] Training visualization dashboard  
- [ ] Hyperparameter tuning experiments  
- [ ] Scaling experiments (num_players, board_diagonal)  
- [ ] Curriculum learning (progressively harder opponents)  

## License

MIT License – see `LICENSE`.

