"""RLlib Torch multi-agent PPO trainer (MAPPO/IPPO modes)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rllib_utils import (
    build_ippo_multiagent,
    build_mappo_multiagent,
    build_ppo_config,
    extract_iteration_record,
    get_agent_space_info,
    init_ray,
    register_sternhalma_env,
    resolve_checkpoint_path,
    shutdown_ray,
    write_checkpoint_metadata,
)
from training.utils import append_jsonl, ensure_dir, load_yaml, save_json


DEFAULT_CONFIG: dict[str, Any] = {
    "env_config": {
        "num_players": 2,
        "board_diagonal": 5,
        "max_actions": 128,
        "max_agent_steps": 600,       # shorter episodes reduce advantage variance
        "reward_mode": "potential_shaped",
    },
    "training_config": {
        "num_iterations": 300,
        "checkpoint_every": 25,
        "seed": 7,
        "mode": "ippo",               # IPPO avoids shared-policy non-stationarity
        "stop_reward": None,
    },
    "rllib_config": {
        "framework": "torch",
        "use_new_api_stack": True,
        "num_gpus": 0.0,
        "num_env_runners": 4,
        "num_envs_per_env_runner": 2,
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",  # avoids batch starvation
        "train_batch_size": 8000,
        "minibatch_size": 512,
        "num_epochs": 10,
        "lr": 3e-4,
        "entropy_coeff": 0.01,
        "gamma": 0.99,                 # MUST match env gamma
        "lambda": 0.95,
        "grad_clip": 0.5,
        "vf_loss_coeff": 0.5,
        "fcnet_hiddens": [512, 512, 256],
        "fcnet_activation": "relu",
    },
    "ray_config": {
        "local_mode": False,
        "log_to_driver": False,
    },
    "output": {"root_dir": "experiments", "run_name": "mappo"},
}


def run_training(config: dict[str, Any], output_dir: Path, seed: int = 7) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    env_config = config.get("env_config", {})
    train_cfg = config.get("training_config", {})
    rllib_cfg = config.get("rllib_config", {})
    ray_cfg = config.get("ray_config", {})

    num_iterations = int(train_cfg.get("num_iterations", 20))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 5))
    mode = str(train_cfg.get("mode", "mappo")).lower()
    stop_reward = train_cfg.get("stop_reward")
    if mode not in {"mappo", "ippo"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: mappo, ippo")

    logs_dir = ensure_dir(output_dir / "logs")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = logs_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    started_ray = init_ray(ray_cfg)
    algo = None
    last_record: dict[str, Any] = {}
    best_return = float("-inf")
    latest_checkpoint_path = ""
    completed_iterations = 0

    try:
        env_name = register_sternhalma_env(env_config=env_config, name_prefix="sternhalma_multiagent")
        agent_spaces = get_agent_space_info(env_config)

        if mode == "ippo":
            policies, mapping_fn = build_ippo_multiagent(agent_spaces)
        else:
            policies, mapping_fn = build_mappo_multiagent(agent_spaces)
        print(
            f"[train] Starting {mode.upper()} PPO: iterations={num_iterations}, "
            f"checkpoint_every={checkpoint_every}, reward_mode={env_config.get('reward_mode', 'default')}",
            flush=True,
        )

        ppo_config = build_ppo_config(
            env_name=env_name,
            env_config=env_config,
            train_cfg=train_cfg,
            rllib_cfg=rllib_cfg,
            seed=seed,
        )
        ppo_config = ppo_config.multi_agent(
            policies=policies,
            policy_mapping_fn=mapping_fn,
            policies_to_train=list(policies.keys()),
            count_steps_by="agent_steps",
        )

        algo = ppo_config.build_algo()
        for iteration in range(1, num_iterations + 1):
            print(f"[train] Iteration {iteration}/{num_iterations} started", flush=True)
            iter_start = time.time()
            result = algo.train()
            iter_time = time.time() - iter_start
            record = extract_iteration_record(result=result, iteration=iteration)
            append_jsonl(metrics_path, record)
            last_record = record
            completed_iterations = iteration
            print(
                "[train] Iteration "
                f"{iteration}/{num_iterations} finished in {iter_time:.1f}s | "
                f"episodes={record.get('episodes_this_iter', 0)} | "
                f"return_mean={float(record.get('episode_return_mean', 0.0) or 0.0):.3f} | "
                f"len_mean={float(record.get('episode_len_mean', 0.0) or 0.0):.1f} | "
                f"return_per_step={float(record.get('episode_return_per_step', 0.0) or 0.0):.3f}",
                flush=True,
            )

            mean_return = float(record.get("episode_return_mean", 0.0) or 0.0)
            if mean_return > best_return:
                best_return = mean_return

            if iteration % checkpoint_every == 0 or iteration == num_iterations:
                save_result = algo.save(checkpoint_dir=str(checkpoints_dir))
                checkpoint_path = resolve_checkpoint_path(save_result)
                metadata = write_checkpoint_metadata(
                    checkpoints_dir=checkpoints_dir,
                    iteration=iteration,
                    checkpoint_path=checkpoint_path,
                    extra={
                        "algorithm": f"rllib_ppo_{mode}",
                        "mode": mode,
                        "seed": seed,
                        "episode_return_mean": mean_return,
                    },
                )
                save_json(checkpoints_dir / f"iteration_{iteration:06d}.json", metadata)
                save_json(checkpoints_dir / "latest.json", metadata)
                latest_checkpoint_path = checkpoint_path
                print(f"[train] Checkpoint saved at iteration {iteration}: {checkpoint_path}", flush=True)

            if stop_reward is not None and mean_return >= float(stop_reward):
                print(
                    f"[train] Stop condition reached at iteration {iteration}: "
                    f"episode_return_mean={mean_return:.3f} >= stop_reward={float(stop_reward):.3f}",
                    flush=True,
                )
                break
    finally:
        if algo is not None:
            algo.stop()
        shutdown_ray(started_ray)

    summary = {
        "algorithm": f"rllib_ppo_{mode}",
        "mode": mode,
        "iterations": completed_iterations,
        "seed": seed,
        "best_episode_return_mean": best_return if best_return != float("-inf") else 0.0,
        "last_episode_return_mean": float(last_record.get("episode_return_mean", 0.0) or 0.0),
        "last_episode_len_mean": float(last_record.get("episode_len_mean", 0.0) or 0.0),
        "last_episode_return_per_step": float(last_record.get("episode_return_per_step", 0.0) or 0.0),
        "metrics_path": str(metrics_path),
        "latest_checkpoint_path": latest_checkpoint_path,
    }
    save_json(output_dir / "summary.json", summary)
    print(
        f"[train] Finished {mode.upper()} PPO: iterations={completed_iterations}, "
        f"best_return_mean={summary['best_episode_return_mean']:.3f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLlib multi-agent PPO training.")
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
    parser.add_argument(
        "--mode",
        choices=["mappo", "ippo"],
        default=None,
        help="Multi-agent setup: shared-policy MAPPO-style or independent-policy IPPO.",
    )
    parser.add_argument("--num-iterations", type=int, default=None, help="Optional iteration override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG | load_yaml(args.config)
    training_cfg = dict(config.get("training_config", {}))
    if args.mode is not None:
        training_cfg["mode"] = args.mode
    if args.num_iterations is not None:
        training_cfg["num_iterations"] = int(args.num_iterations)
    config["training_config"] = training_cfg
    output_cfg = config.get("output", {})
    output_dir = args.output_dir or Path(output_cfg.get("root_dir", "experiments")) / output_cfg.get(
        "run_name", "mappo"
    )
    seed = args.seed if args.seed is not None else int(training_cfg.get("seed", 7))
    summary = run_training(config=config, output_dir=output_dir, seed=seed)
    print(f"Training finished. Summary written to {output_dir / 'summary.json'}")
    print(summary)


if __name__ == "__main__":
    main()
