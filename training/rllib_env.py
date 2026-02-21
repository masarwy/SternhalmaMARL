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
    Convert nested/object observations into fixed numeric tensors:

    {
      "observations": float32 vector [flatten(board), current_player],
      "action_mask": float32 vector
    }
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

            feature_dim = int(np.prod(board_space.shape)) + 1
            board_low = float(np.min(board_space.low))
            board_high = float(np.max(board_space.high))

            self._observation_spaces[agent_id] = spaces.Dict(
                {
                    "observations": spaces.Box(
                        low=board_low,
                        high=board_high,
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
            usable = min(board.size, feature_dim - 1)
            if usable > 0:
                features[:usable] = board[:usable]
            features[-1] = current_player

        raw_mask = np.asarray(wrapped_obs.get("action_mask"), dtype=np.float32).reshape(-1)
        mask = np.zeros((mask_dim,), dtype=np.float32)
        usable_mask = min(raw_mask.size, mask_dim)
        if usable_mask > 0:
            mask[:usable_mask] = raw_mask[:usable_mask]

        return {"observations": features, "action_mask": mask}
