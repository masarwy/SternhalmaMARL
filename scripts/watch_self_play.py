"""Render a trained self-play PPO checkpoint for visual inspection."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ray.rllib.core.columns import Columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

from training.rllib_env import SternhalmaRLlibObsWrapper
from training.rllib_utils import init_ray, shutdown_ray
from training.utils import ensure_dir, load_yaml, make_env

torch, _ = try_import_torch()


def _resolve_checkpoint_dir(raw_path: Path) -> Path:
    path = raw_path.expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_file():
        raise ValueError(f"Expected checkpoint directory, got file: {path}")
    return path


def _load_ctor_config(checkpoint_dir: Path) -> dict[str, Any]:
    ctor_path = checkpoint_dir / "class_and_ctor_args.pkl"
    if not ctor_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata file: {ctor_path}")
    payload = pickle.loads(ctor_path.read_bytes())
    ctor = payload.get("ctor_args_and_kwargs")
    if not isinstance(ctor, tuple) or len(ctor) != 2:
        raise ValueError("Unexpected checkpoint ctor args payload format.")
    ctor_args, _ctor_kwargs = ctor
    if not isinstance(ctor_args, tuple) or not ctor_args:
        raise ValueError("Unexpected checkpoint ctor args payload content.")
    cfg = ctor_args[0]
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint config payload is not a dict.")
    return cfg


def _register_env_from_checkpoint(checkpoint_dir: Path) -> tuple[str, dict[str, Any]]:
    cfg = _load_ctor_config(checkpoint_dir)
    env_name = str(cfg.get("env", "")).strip()
    if not env_name:
        raise ValueError("Checkpoint config does not contain a valid 'env' name.")
    env_config = dict(cfg.get("env_config", {}))
    register_env(env_name, lambda c: PettingZooEnv(SternhalmaRLlibObsWrapper(make_env(c))))
    return env_name, env_config


def _module_device(module: Any):
    if torch is None:
        return None
    try:
        return next(module.parameters()).device
    except Exception:
        return None


def _to_model_tensor(value: Any, device: Any) -> Any:
    array = np.expand_dims(np.asarray(value), axis=0)
    if torch is None:
        return array
    tensor = torch.as_tensor(array, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


def _batch_obs(observation: Any, module: Any) -> dict[str, Any]:
    device = _module_device(module)
    if isinstance(observation, dict):
        batched_obs = {key: _to_model_tensor(value, device) for key, value in observation.items()}
    else:
        batched_obs = _to_model_tensor(observation, device)
    return {Columns.OBS: batched_obs}


def _extract_action_from_module_out(module_out: dict[str, Any], stochastic: bool) -> int:
    if Columns.ACTIONS in module_out:
        actions = module_out[Columns.ACTIONS]
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        return int(np.asarray(actions).reshape(-1)[0])

    logits = module_out.get(Columns.ACTION_DIST_INPUTS)
    if logits is None:
        keys = sorted(module_out.keys())
        raise ValueError(f"Module output missing action keys. Got keys: {keys}")

    if torch is None:
        logits_arr = np.asarray(logits)
        if logits_arr.ndim == 1:
            logits_arr = logits_arr.reshape(1, -1)
        if stochastic:
            probs = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            return int(np.random.choice(np.arange(probs.shape[-1]), p=probs[0]))
        return int(np.argmax(logits_arr[0]))

    logits_t = logits if hasattr(logits, "detach") else torch.as_tensor(logits)
    if logits_t.ndim == 1:
        logits_t = logits_t.unsqueeze(0)
    if stochastic:
        action_t = torch.distributions.Categorical(logits=logits_t).sample()
    else:
        action_t = torch.argmax(logits_t, dim=-1)
    return int(action_t.reshape(-1)[0].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained self-play PPO checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/ppo_self_play.yaml"),
        help="Training config used for defaults (episodes/seed only).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments/ppo_self_play/checkpoints"),
        help="RLlib checkpoint directory produced by training.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for evaluation episodes.")
    parser.add_argument(
        "--render-mode",
        choices=["human", "ansi", "rgb_array", "none"],
        default="human",
        help="Rendering mode. Use 'ansi' for terminal text output.",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="shared_policy",
        help="Policy id to use for action inference.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastic actions (default is deterministic).",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=None,
        help="Optional hard cap on agent steps per episode during playback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    train_cfg = dict(config.get("training_config", {}))
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    _env_name, env_config = _register_env_from_checkpoint(checkpoint_dir)

    if args.render_mode == "none":
        env_config["render_mode"] = None
    else:
        env_config["render_mode"] = args.render_mode

    # Keep algorithm-side temporary output inside this workspace.
    import ray.rllib.algorithms.algorithm as algorithm_module

    algorithm_module.DEFAULT_STORAGE_PATH = str(ensure_dir(PROJECT_ROOT / ".ray_results"))

    started_ray = False
    try:
        started_ray = init_ray({"local_mode": False, "log_to_driver": False})
    except RuntimeError as first_err:
        print(f"[watch] Ray init failed in normal mode: {first_err}")
        print("[watch] Retrying Ray init in local_mode=True ...")
        try:
            started_ray = init_ray({"local_mode": True, "log_to_driver": False})
        except RuntimeError as second_err:
            raise RuntimeError(
                "Unable to start Ray for checkpoint playback. "
                "Try running `ray stop` and rerun this command."
            ) from second_err

    algo = None
    env = None
    try:
        algo = Algorithm.from_checkpoint(str(checkpoint_dir.resolve()))
        module = algo.get_module(args.policy_id) or algo.get_module()
        if module is None:
            raise RuntimeError(
                f"No RLModule found for policy id '{args.policy_id}'. "
                "Check --policy-id and checkpoint compatibility."
            )
        env = SternhalmaRLlibObsWrapper(make_env(env_config))

        for ep in range(1, args.episodes + 1):
            env.reset(seed=seed + ep - 1)
            rewards = {agent: 0.0 for agent in env.possible_agents}
            steps = 0

            print(f"[watch] Episode {ep}/{args.episodes} started")
            while env.agents:
                agent_id = env.agent_selection
                observation, reward, terminated, truncated, info = env.last()
                rewards[agent_id] = rewards.get(agent_id, 0.0) + float(reward)

                if args.render_mode == "ansi":
                    frame = env.render()
                    if isinstance(frame, str):
                        print(frame)
                elif args.render_mode in {"human", "rgb_array"}:
                    env.render()

                if terminated or truncated:
                    action = None
                else:
                    batch = _batch_obs(observation, module)
                    if args.stochastic:
                        module_out = module.forward_exploration(batch)
                    else:
                        module_out = module.forward_inference(batch)
                    action = _extract_action_from_module_out(module_out, stochastic=args.stochastic)

                env.step(action)

                if action is not None:
                    steps += 1
                    if args.max_agent_steps is not None and steps >= args.max_agent_steps:
                        break

            winner = max(rewards.items(), key=lambda kv: kv[1])[0] if rewards else "n/a"
            print(
                f"[watch] Episode {ep} finished | steps={steps} | "
                f"winner={winner} | rewards={rewards}"
            )
    finally:
        if env is not None:
            env.close()
        if algo is not None:
            algo.stop()
        shutdown_ray(started_ray)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[watch] Failed: {exc}")
        raise SystemExit(1)
