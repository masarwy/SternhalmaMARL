"""PettingZoo wrappers tailored for RLlib compatibility."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper


def _extract_obs_payload(raw_obs: Any) -> dict[str, Any] | None:
    if isinstance(raw_obs, np.ndarray) and raw_obs.dtype == object:
        if raw_obs.shape == ():
            raw_obs = raw_obs.item()
        elif raw_obs.size:
            raw_obs = raw_obs.flat[0]
    if isinstance(raw_obs, list) and raw_obs:
        raw_obs = raw_obs[0]
    if isinstance(raw_obs, dict):
        return raw_obs
    return None


class SternhalmaRLlibObsWrapper(BaseWrapper):
    """
    Convert nested/object observations into a flat float32 feature vector
    that RLlib's MLP backbone can consume directly.

    Output observation format::

        {
            "observations": float32 vector
                            [flatten(board), current_player, distances_to_home...],
            "action_mask":  float32 vector  (shape: max_actions,)
        }

    The ``distances_to_home`` features (per-piece hex distances normalised to
    [0, 1]) come from the SternhalmaEnv fix that adds them to ``observe()``.
    Concatenating them here means the MLP gets a compact spatial signal without
    having to re-learn hex geometry from the raw board encoding.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        self._observation_spaces: dict[str, spaces.Dict] = {}

        for agent_id in self.possible_agents:
            wrapped_space = env.observation_space(agent_id)
            if not isinstance(wrapped_space, spaces.Dict):
                raise ValueError("Expected Dict observation space from discrete_action_env wrapper.")

            obs_space = wrapped_space.spaces["observations"]
            mask_space = wrapped_space.spaces["action_mask"]
            if not isinstance(obs_space, spaces.Dict):
                raise ValueError("Expected nested Dict at observation['observations'].")

            board_space = obs_space.spaces["board"]
            if not isinstance(board_space, spaces.Box):
                raise ValueError("Expected board observation as Box space.")

            # Base feature dim: flattened board + current_player scalar.
            base_dim = int(np.prod(board_space.shape)) + 1

            # Extra dim: distances_to_home vector (added in SternhalmaEnv fix).
            dist_dim = 0
            if "distances_to_home" in obs_space.spaces:
                dist_space = obs_space.spaces["distances_to_home"]
                dist_dim = int(np.prod(dist_space.shape))

            feature_dim = base_dim + dist_dim

            # Use the widest safe bounds across board + distance features.
            board_low = float(np.min(board_space.low))
            board_high = float(np.max(board_space.high))
            # distances_to_home is in [0, 1]; board low may be negative, so
            # we keep the existing board_low as the global lower bound.
            obs_low = board_low
            obs_high = max(board_high, 1.0)

            self._observation_spaces[agent_id] = spaces.Dict(
                {
                    "observations": spaces.Box(
                        low=obs_low,
                        high=obs_high,
                        shape=(feature_dim,),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=mask_space.shape,
                        dtype=np.float32,
                    ),
                }
            )

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def last(self, observe: bool = True) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        agent = self.agent_selection
        assert agent is not None
        observation = self.observe(agent) if observe else None
        _, reward, terminated, truncated, info = self.env.last(observe=False)
        return observation, float(reward), terminated, truncated, info

    def observe(self, agent: str):
        wrapped_obs = self.env.observe(agent)
        payload = _extract_obs_payload(wrapped_obs.get("observations"))
        obs_space = self._observation_spaces[agent]
        feature_dim = obs_space.spaces["observations"].shape[0]
        mask_dim = obs_space.spaces["action_mask"].shape[0]

        features = np.zeros((feature_dim,), dtype=np.float32)
        if payload is not None:
            board = np.asarray(payload.get("board"), dtype=np.float32).reshape(-1)
            current_player = float(payload.get("current_player", 0.0))

            # -- board features --
            board_end = min(board.size, feature_dim - 1)
            if board_end > 0:
                features[:board_end] = board[:board_end]
            # current_player goes right after the board
            cp_idx = board_end
            if cp_idx < feature_dim:
                features[cp_idx] = current_player

            # -- distances_to_home features (new in SternhalmaEnv fix) --
            dist_raw = payload.get("distances_to_home")
            if dist_raw is not None:
                dist = np.asarray(dist_raw, dtype=np.float32).reshape(-1)
                dist_start = cp_idx + 1
                usable_dist = min(dist.size, feature_dim - dist_start)
                if usable_dist > 0:
                    features[dist_start: dist_start + usable_dist] = dist[:usable_dist]

        raw_mask = np.asarray(wrapped_obs.get("action_mask"), dtype=np.float32).reshape(-1)
        mask = np.zeros((mask_dim,), dtype=np.float32)
        usable_mask = min(raw_mask.size, mask_dim)
        if usable_mask > 0:
            mask[:usable_mask] = raw_mask[:usable_mask]

        return {"observations": features, "action_mask": mask}
