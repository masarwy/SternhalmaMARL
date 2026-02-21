"""Environment wrappers used by training pipelines."""

from __future__ import annotations

from typing import Any

from pettingzoo.utils import BaseWrapper


class EpisodeStepLimitWrapper(BaseWrapper):
    """
    Truncate an AEC episode after a maximum number of non-dead agent steps.
    """

    def __init__(self, env: Any, max_agent_steps: int):
        if max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be > 0.")
        super().__init__(env)
        self.max_agent_steps = int(max_agent_steps)
        self._agent_steps = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._agent_steps = 0
        super().reset(seed=seed, options=options)

    def last(self, observe: bool = True) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Delegate to wrapped env so inner wrappers can transform rewards.
        return self.env.last(observe=observe)

    def step(self, action: Any) -> None:
        agent = getattr(self, "agent_selection", None)
        is_dead_step = (
            agent is not None
            and (self.terminations.get(agent, False) or self.truncations.get(agent, False))
        )
        super().step(action)

        if is_dead_step:
            return
        if not self.agents:
            return

        self._agent_steps += 1
        if self._agent_steps < self.max_agent_steps:
            return

        # Time-limit truncation for all remaining live agents.
        for live_agent in list(self.agents):
            if not self.terminations.get(live_agent, False):
                self.truncations[live_agent] = True
                info = self.infos.get(live_agent, {})
                info["time_limit_reached"] = True
                self.infos[live_agent] = info


class RewardTransformWrapper(BaseWrapper):
    """
    Transform rewards returned by AEC env.last() for training stability.
    Applies: reward := clip(reward * reward_scale, -reward_clip_abs, reward_clip_abs)
    when reward_clip_abs is provided.
    """

    def __init__(
        self,
        env: Any,
        reward_scale: float = 1.0,
        reward_clip_abs: float | None = None,
    ):
        super().__init__(env)
        self.reward_scale = float(reward_scale)
        self.reward_clip_abs = None if reward_clip_abs is None else float(reward_clip_abs)
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be > 0.")
        if self.reward_clip_abs is not None and self.reward_clip_abs <= 0.0:
            raise ValueError("reward_clip_abs must be > 0 when provided.")

    def last(self, observe: bool = True) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().last(observe=observe)
        transformed = float(reward) * self.reward_scale
        if self.reward_clip_abs is not None:
            limit = self.reward_clip_abs
            transformed = max(-limit, min(limit, transformed))
        return observation, transformed, terminated, truncated, info
