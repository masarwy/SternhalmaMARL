"""A simple geometric heuristic baseline for discrete Sternhalma actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.baselines.random_agent import RandomAgent, extract_action_mask


def _extract_obs_payload(observation: dict[str, Any]) -> dict[str, Any] | None:
    raw = observation.get("observations")
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.shape == ():
            raw = raw.item()
        elif raw.size:
            raw = raw.flat[0]
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, dict):
        return raw
    return None


class HeuristicAgent(RandomAgent):
    """
    Prioritizes moves that advance pieces toward the reflected centroid target.
    Falls back to longest move, then random.
    """

    def act(self, observation: Any, info: dict[str, Any], action_space: Any) -> int | None:
        if not isinstance(observation, dict):
            return super().act(observation, info, action_space)

        valid_moves = info.get("valid_moves", []) if isinstance(info, dict) else []
        if not valid_moves:
            return super().act(observation, info, action_space)

        mask = extract_action_mask(observation)
        payload = _extract_obs_payload(observation)
        board: np.ndarray | None = None
        current_player: int | None = None
        if payload is not None:
            board_data = payload.get("board")
            if board_data is not None:
                board = np.asarray(board_data)
            raw_player = payload.get("current_player")
            if raw_player is not None:
                current_player = int(raw_player)

        best_action = None
        best_score = float("-inf")

        center = None
        goal = None
        if board is not None and current_player is not None and board.ndim == 2:
            piece_value = current_player + 1
            piece_coords = np.argwhere(board == piece_value)
            if piece_coords.size > 0:
                center = np.array([(board.shape[0] - 1) / 2.0, (board.shape[1] - 1) / 2.0])
                centroid = piece_coords.mean(axis=0)
                goal = 2.0 * center - centroid

        for idx, move in enumerate(valid_moves):
            if mask is not None and (idx >= mask.size or not mask[idx]):
                continue
            if not (isinstance(move, (list, tuple)) and len(move) == 2):
                continue
            start = np.asarray(move[0], dtype=float)
            end = np.asarray(move[1], dtype=float)
            jump_length = float(np.linalg.norm(end - start))

            if goal is not None:
                progress = float(np.linalg.norm(start - goal) - np.linalg.norm(end - goal))
                score = progress + 0.05 * jump_length
            else:
                score = jump_length
            if score > best_score:
                best_score = score
                best_action = idx

        if best_action is not None:
            return int(best_action)
        return super().act(observation, info, action_space)
