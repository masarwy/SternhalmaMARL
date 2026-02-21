"""Convenience benchmark for baseline agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tournament import run_tournament
from training.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline agents.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/tournament.yaml"),
        help="Tournament config path.",
    )
    parser.add_argument("--num_games", type=int, default=30, help="Games per pair.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    result = run_tournament(
        agents=["random", "heuristic"],
        num_games=args.num_games,
        env_config=cfg.get("env_config", {}),
        seed=int(cfg.get("tournament_config", {}).get("seed", 42)),
        max_steps_per_episode=cfg.get("tournament_config", {}).get("max_steps_per_episode"),
    )
    print(json.dumps(result["ratings"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
