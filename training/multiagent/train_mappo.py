"""Minimal multi-agent runner with mixed baseline policies."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baselines.heuristic_agent import HeuristicAgent
from agents.baselines.random_agent import RandomAgent
from training.utils import append_jsonl, ensure_dir, load_yaml, make_env, save_json, set_global_seed
from training.utils import run_aec_episode


DEFAULT_CONFIG: dict[str, Any] = {
    "env_config": {"num_players": 2, "board_diagonal": 5, "max_actions": 128},
    "training_config": {
        "num_episodes": 50,
        "max_steps_per_episode": 300,
        "checkpoint_every": 25,
        "seed": 7,
        "agent_types": ["heuristic", "random"],
    },
    "output": {"root_dir": "experiments", "run_name": "mappo"},
}


def _build_policy(agent_type: str, seed: int) -> RandomAgent:
    if agent_type == "heuristic":
        return HeuristicAgent(seed=seed)
    return RandomAgent(seed=seed)


def run_training(config: dict[str, Any], output_dir: Path, seed: int = 7) -> dict[str, Any]:
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

    possible_agents = list(env.possible_agents)
    requested_types = list(train_cfg.get("agent_types", []))
    if not requested_types:
        requested_types = ["heuristic"] + ["random"] * max(0, len(possible_agents) - 1)

    policy_map: dict[str, RandomAgent] = {}
    for idx, agent_id in enumerate(possible_agents):
        agent_type = requested_types[idx] if idx < len(requested_types) else "random"
        policy_map[agent_id] = _build_policy(agent_type=agent_type, seed=seed + idx)

    episode_means: list[float] = []
    episode_lengths: list[int] = []

    def policy_fn(agent_id: str, observation: Any, info: dict[str, Any], action_space: Any) -> int | None:
        return policy_map[agent_id].act(observation, info, action_space)

    for episode in range(1, num_episodes + 1):
        out = run_aec_episode(
            env=env,
            policy_fn=policy_fn,
            seed=seed + episode,
            max_steps=max_steps,
        )
        rewards = out["rewards"]
        mean_return = float(statistics.fmean(rewards.values())) if rewards else 0.0
        steps = int(out["episode_steps"])

        episode_means.append(mean_return)
        episode_lengths.append(steps)

        append_jsonl(
            metrics_path,
            {
                "episode": episode,
                "episode_return_mean": mean_return,
                "episode_steps": steps,
                "episode_rewards": rewards,
            },
        )

        if episode % checkpoint_every == 0 or episode == num_episodes:
            ckpt_payload = {
                "algorithm": "mappo_baseline_mix",
                "episode": episode,
                "seed": seed,
                "agent_types": requested_types,
                "mean_return_so_far": float(statistics.fmean(episode_means)),
                "mean_steps_so_far": float(statistics.fmean(episode_lengths)),
            }
            save_json(checkpoints_dir / f"episode_{episode:06d}.json", ckpt_payload)
            save_json(checkpoints_dir / "latest.json", ckpt_payload)

    env.close()
    summary = {
        "algorithm": "mappo_baseline_mix",
        "episodes": num_episodes,
        "seed": seed,
        "mean_return": float(statistics.fmean(episode_means)) if episode_means else 0.0,
        "mean_episode_steps": float(statistics.fmean(episode_lengths)) if episode_lengths else 0.0,
        "metrics_path": str(metrics_path),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal multi-agent training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/mappo.yaml"),
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
        "run_name", "mappo"
    )
    train_cfg = config.get("training_config", {})
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 7))
    summary = run_training(config=config, output_dir=output_dir, seed=seed)
    print(f"Training finished. Summary written to {output_dir / 'summary.json'}")
    print(summary)


if __name__ == "__main__":
    main()
