"""Minimal self-play training runner (mask-aware baseline policy)."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baselines.random_agent import RandomAgent
from training.utils import append_jsonl, ensure_dir, load_yaml, make_env, save_json, set_global_seed
from training.utils import run_aec_episode


DEFAULT_CONFIG: dict[str, Any] = {
    "env_config": {"num_players": 2, "board_diagonal": 5, "max_actions": 128},
    "training_config": {
        "num_episodes": 50,
        "max_steps_per_episode": 300,
        "checkpoint_every": 25,
        "seed": 42,
    },
    "output": {"root_dir": "experiments", "run_name": "ppo_self_play"},
}


def run_training(config: dict[str, Any], output_dir: Path, seed: int = 42) -> dict[str, Any]:
    env_config = config.get("env_config", {})
    train_cfg = config.get("training_config", {})

    num_episodes = int(train_cfg.get("num_episodes", 50))
    max_steps = train_cfg.get("max_steps_per_episode")
    checkpoint_every = int(train_cfg.get("checkpoint_every", 25))

    logs_dir = ensure_dir(output_dir / "logs")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = logs_dir / "metrics.jsonl"

    if metrics_path.exists():
        metrics_path.unlink()

    set_global_seed(seed)
    env = make_env(env_config)
    policy = RandomAgent(seed=seed)

    episode_means: list[float] = []
    episode_lengths: list[int] = []

    def policy_fn(agent_id: str, observation: Any, info: dict[str, Any], action_space: Any) -> int | None:
        del agent_id
        return policy.act(observation, info, action_space)

    for episode in range(1, num_episodes + 1):
        episode_out = run_aec_episode(
            env=env,
            policy_fn=policy_fn,
            seed=seed + episode,
            max_steps=max_steps,
        )
        rewards = episode_out["rewards"]
        mean_return = float(statistics.fmean(rewards.values())) if rewards else 0.0
        episode_steps = int(episode_out["episode_steps"])

        episode_means.append(mean_return)
        episode_lengths.append(episode_steps)

        record = {
            "episode": episode,
            "episode_return_mean": mean_return,
            "episode_steps": episode_steps,
            "episode_rewards": rewards,
        }
        append_jsonl(metrics_path, record)

        if episode % checkpoint_every == 0 or episode == num_episodes:
            ckpt_payload = {
                "algorithm": "self_play_baseline",
                "episode": episode,
                "seed": seed,
                "mean_return_so_far": float(statistics.fmean(episode_means)),
                "mean_steps_so_far": float(statistics.fmean(episode_lengths)),
            }
            save_json(checkpoints_dir / f"episode_{episode:06d}.json", ckpt_payload)
            save_json(checkpoints_dir / "latest.json", ckpt_payload)

    env.close()

    summary = {
        "algorithm": "self_play_baseline",
        "episodes": num_episodes,
        "seed": seed,
        "mean_return": float(statistics.fmean(episode_means)) if episode_means else 0.0,
        "mean_episode_steps": float(statistics.fmean(episode_lengths)) if episode_lengths else 0.0,
        "metrics_path": str(metrics_path),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal self-play training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/ppo_self_play.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG | load_yaml(args.config)
    output_cfg = config.get("output", {})
    output_dir = args.output_dir or Path(output_cfg.get("root_dir", "experiments")) / output_cfg.get(
        "run_name", "ppo_self_play"
    )
    train_cfg = config.get("training_config", {})
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    summary = run_training(config=config, output_dir=output_dir, seed=seed)
    print(f"Training finished. Summary written to {output_dir / 'summary.json'}")
    print(summary)


if __name__ == "__main__":
    main()
