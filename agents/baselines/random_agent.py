"""Random baseline with optional action-mask support."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_action_mask(observation: Any) -> np.ndarray | None:
    """Return a 1D boolean action mask when present."""
    if not isinstance(observation, dict):
        return None
    raw_mask = observation.get("action_mask")
    if raw_mask is None:
        return None
    mask = np.asarray(raw_mask).astype(bool)
    if mask.ndim != 1:
        return None
    return mask


class RandomAgent:
    """Uniform policy over legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Any, info: dict[str, Any], action_space: Any) -> int | None:
        mask = extract_action_mask(observation)
        if mask is not None:
            legal_actions = np.flatnonzero(mask)
            if legal_actions.size == 0:
                return None
            return int(self.rng.choice(legal_actions))
        if hasattr(action_space, "sample"):
            return int(action_space.sample())
        return None
