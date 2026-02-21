from pathlib import Path

import pytest

from training.multiagent.train_mappo import run_training as run_mappo_training
from training.self_play.train_ppo import run_training as run_self_play_training


pytest.importorskip("sternhalma_v0")


def test_self_play_training_smoke(tmp_path: Path) -> None:
    config = {
        "env_config": {"num_players": 2, "board_diagonal": 5, "max_actions": 64},
        "training_config": {"num_episodes": 3, "max_steps_per_episode": 30, "checkpoint_every": 2},
    }
    out_dir = tmp_path / "self_play"
    summary = run_self_play_training(config=config, output_dir=out_dir, seed=101)

    assert summary["episodes"] == 3
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "logs" / "metrics.jsonl").exists()
    assert (out_dir / "checkpoints" / "latest.json").exists()


def test_mappo_training_smoke(tmp_path: Path) -> None:
    config = {
        "env_config": {"num_players": 2, "board_diagonal": 5, "max_actions": 64},
        "training_config": {
            "num_episodes": 3,
            "max_steps_per_episode": 30,
            "checkpoint_every": 2,
            "agent_types": ["heuristic", "random"],
        },
    }
    out_dir = tmp_path / "mappo"
    summary = run_mappo_training(config=config, output_dir=out_dir, seed=202)

    assert summary["episodes"] == 3
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "logs" / "metrics.jsonl").exists()
    assert (out_dir / "checkpoints" / "latest.json").exists()
