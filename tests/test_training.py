from pathlib import Path

import pytest

from training.multiagent.train_mappo import run_training as run_mappo_training
from training.self_play.train_ppo import run_training as run_self_play_training


pytest.importorskip("sternhalma_v0")
pytest.importorskip("ray")


def _run_or_skip_on_ray_runtime_error(fn):
    try:
        return fn()
    except RuntimeError as exc:
        text = str(exc).lower()
        if "failed to start the grpc server" in text or "timed out waiting for file" in text:
            pytest.skip(f"Ray runtime unavailable in current environment: {exc}")
        raise


def test_self_play_training_smoke(tmp_path: Path) -> None:
    config = {
        "env_config": {
            "num_players": 2,
            "board_diagonal": 7,
            "max_actions": 64,
            "max_agent_steps": 200,
            "reward_mode": "potential_shaped",
            "reward_scale": 0.1,
            "reward_clip_abs": 15.0,
        },
        "training_config": {"num_iterations": 1, "checkpoint_every": 1},
        "rllib_config": {
            "framework": "torch",
            "num_gpus": 0.0,
            "num_env_runners": 0,
            "rollout_fragment_length": 50,
            "batch_mode": "truncate_episodes",
            "train_batch_size": 100,
            "minibatch_size": 50,
            "num_epochs": 1,
            "lr": 3e-4,
            "gamma": 0.99,
            "grad_clip": 0.5,
            "vf_loss_coeff": 0.5,
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "ray_config": {"local_mode": False, "log_to_driver": False},
    }
    out_dir = tmp_path / "self_play"
    summary = _run_or_skip_on_ray_runtime_error(
        lambda: run_self_play_training(config=config, output_dir=out_dir, seed=101)
    )

    assert summary["iterations"] == 1
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "logs" / "metrics.jsonl").exists()
    assert (out_dir / "checkpoints" / "latest.json").exists()


def test_mappo_training_smoke(tmp_path: Path) -> None:
    config = {
        "env_config": {
            "num_players": 2,
            "board_diagonal": 7,
            "max_actions": 64,
            "max_agent_steps": 200,
            "reward_mode": "potential_shaped",
        },
        "training_config": {
            "num_iterations": 1,
            "checkpoint_every": 1,
            "mode": "ippo",
        },
        "rllib_config": {
            "framework": "torch",
            "num_gpus": 0.0,
            "num_env_runners": 0,
            "rollout_fragment_length": 50,
            "batch_mode": "truncate_episodes",
            "train_batch_size": 100,
            "minibatch_size": 50,
            "num_epochs": 1,
            "lr": 3e-4,
            "gamma": 0.99,
            "grad_clip": 0.5,
            "vf_loss_coeff": 0.5,
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "ray_config": {"local_mode": False, "log_to_driver": False},
    }
    out_dir = tmp_path / "mappo"
    summary = _run_or_skip_on_ray_runtime_error(lambda: run_mappo_training(config=config, output_dir=out_dir, seed=202))

    assert summary["iterations"] == 1
    assert summary["mode"] == "ippo"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "logs" / "metrics.jsonl").exists()
    assert (out_dir / "checkpoints" / "latest.json").exists()
