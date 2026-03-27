"""Unit tests for SternhalmaRLlibObsWrapper — no Ray required."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sternhalma_v0")

from training.rllib_env import SternhalmaRLlibObsWrapper
from training.utils import make_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wrapped(board_diagonal: int = 5, num_players: int = 2) -> SternhalmaRLlibObsWrapper:
    env_config = {
        "num_players": num_players,
        "board_diagonal": board_diagonal,
        "max_actions": 64,
        "reward_mode": "potential_shaped",
    }
    return SternhalmaRLlibObsWrapper(make_env(env_config))


# ---------------------------------------------------------------------------
# Observation space tests
# ---------------------------------------------------------------------------

def test_observation_space_contains_distances_features() -> None:
    """Feature dim must be board_flat + 1 (current_player) + dist features."""
    env = _make_wrapped()
    env.reset()
    agent = env.agent_selection
    obs_space = env.observation_space(agent)

    # Board shape for diagonal=5: (h-1) x (w-1)
    # We just check the space is bigger than board_flat + 1
    inner = obs_space.spaces["observations"]
    feature_dim = inner.shape[0]

    # board_diagonal=5 -> board is (14-1) x (10-1) = 13x9 = 117 cells
    # + 1 (current_player) = 118 baseline without distances.
    # With 2 players * 3 pieces = 6 distance features -> 124 total.
    # We assert strictly greater than the no-distance baseline (118).
    assert feature_dim > 118, (
        f"Expected feature_dim > 118 (board+player+distances), got {feature_dim}"
    )
    env.close()


def test_observation_space_bounds() -> None:
    """The observation Box low should be <= 0 (board has negative values)."""
    env = _make_wrapped()
    env.reset()
    agent = env.agent_selection
    inner = env.observation_space(agent).spaces["observations"]
    assert inner.low.min() < 0, "Board has -2 cells; obs low should be negative"
    assert inner.high.max() >= 1.0, "distances_to_home is in [0,1]; high must be >= 1"
    env.close()


def test_observe_shape_matches_space() -> None:
    """observe() must return a vector whose shape matches the declared space."""
    env = _make_wrapped()
    env.reset()
    agent = env.agent_selection
    obs = env.observe(agent)
    declared = env.observation_space(agent)

    feat = obs["observations"]
    mask = obs["action_mask"]

    assert feat.shape == declared.spaces["observations"].shape, (
        f"Feature shape mismatch: {feat.shape} vs {declared.spaces['observations'].shape}"
    )
    assert mask.shape == declared.spaces["action_mask"].shape, (
        f"Mask shape mismatch: {mask.shape} vs {declared.spaces['action_mask'].shape}"
    )
    env.close()


def test_distances_appended_after_board_and_player() -> None:
    """The last N entries of the feature vector should be in [0, 1] (distances)."""
    env = _make_wrapped(board_diagonal=5, num_players=2)
    env.reset()
    agent = env.agent_selection

    # Recover the distance feature slice length from the underlying obs space.
    inner_obs_space = env.env.observation_space(agent)  # DiscreteActionMaskWrapper space
    base_obs_space = inner_obs_space.spaces["observations"]  # raw env space
    dist_dim = int(np.prod(base_obs_space.spaces["distances_to_home"].shape))

    feat = env.observe(agent)["observations"]
    dist_slice = feat[-dist_dim:]

    assert np.all(dist_slice >= 0.0), f"distances contain negative values: {dist_slice}"
    assert np.all(dist_slice <= 1.0), f"distances exceed 1.0: {dist_slice}"
    env.close()


def test_action_mask_dtype_and_values() -> None:
    """action_mask must be float32 with entries in {0.0, 1.0}."""
    env = _make_wrapped()
    env.reset()
    obs = env.observe(env.agent_selection)
    mask = obs["action_mask"]
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0}), f"Unexpected mask values: {np.unique(mask)}"
    env.close()


def test_feature_vector_is_finite() -> None:
    """No NaN or Inf should appear in the observation after reset."""
    env = _make_wrapped()
    env.reset()
    feat = env.observe(env.agent_selection)["observations"]
    assert np.all(np.isfinite(feat)), f"Non-finite values in observation: {feat}"
    env.close()


def test_observation_space_valid_after_step() -> None:
    """observe() must remain within its declared space after a valid step."""
    env = _make_wrapped()
    env.reset()
    agent = env.agent_selection
    valid_moves = env.infos.get(agent, {}).get("valid_moves", [])
    if valid_moves:
        env.step(0)  # index 0 = first valid move via DiscreteActionMaskWrapper
    obs = env.observe(env.agent_selection)
    space = env.observation_space(env.agent_selection)
    assert space.spaces["observations"].contains(obs["observations"]), (
        "observations vector is outside its declared Box after a step"
    )
    env.close()
