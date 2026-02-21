"""Shared training/evaluation utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from training.env_wrappers import EpisodeStepLimitWrapper, RewardTransformWrapper

try:
    import yaml
except ImportError:  # pragma: no cover - exercised through CLI only
    yaml = None  # type: ignore[assignment]

try:
    import sternhalma_v0
except ImportError:  # pragma: no cover - tested with importorskip in unit tests
    sternhalma_v0 = None  # type: ignore[assignment]


PolicyFn = Callable[[str, Any, dict[str, Any], Any], int | None]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """
    Minimal YAML subset parser for this repository's config files.
    Supports nested mappings/lists and primitive scalars.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    cleaned: list[tuple[int, str]] = []
    for line in raw_lines:
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        cleaned.append((indent, content.lstrip(" ")))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    idx = 0
    while idx < len(cleaned):
        indent, token = cleaned[idx]
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if token.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Invalid list item in {path}: {token}")
            item_text = token[2:].strip()
            if ":" in item_text and not item_text.startswith(("'", '"')):
                key, rest = item_text.split(":", 1)
                item: dict[str, Any] = {}
                item[key.strip()] = _parse_scalar(rest.strip()) if rest.strip() else {}
                parent.append(item)
                if not rest.strip():
                    stack.append((indent, item[key.strip()]))
            else:
                parent.append(_parse_scalar(item_text))
            idx += 1
            continue

        if ":" not in token:
            raise ValueError(f"Invalid mapping in {path}: {token}")

        key, rest = token.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        if rest:
            if not isinstance(parent, dict):
                raise ValueError(f"Expected mapping parent in {path} for token: {token}")
            parent[key] = _parse_scalar(rest)
            idx += 1
            continue

        next_container: Any = {}
        if idx + 1 < len(cleaned):
            next_indent, next_token = cleaned[idx + 1]
            if next_indent > indent and next_token.startswith("- "):
                next_container = []
        if not isinstance(parent, dict):
            raise ValueError(f"Expected mapping parent in {path} for token: {token}")
        parent[key] = next_container
        stack.append((indent, next_container))
        idx += 1

    return root


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        data = _load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, found {type(data).__name__}")
    return data


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_jsonable(payload), sort_keys=True))
        handle.write("\n")


def make_env(env_config: dict[str, Any]):
    if sternhalma_v0 is None:
        raise RuntimeError(
            "sternhalma_v0 module is not installed. "
            "Install SternhalmaEnv before running training/evaluation."
        )
    config = dict(env_config)
    max_actions = int(config.pop("max_actions", 256))
    max_agent_steps = config.pop("max_agent_steps", None)
    reward_scale = float(config.pop("reward_scale", 1.0))
    reward_clip_abs_raw = config.pop("reward_clip_abs", None)
    reward_clip_abs = None if reward_clip_abs_raw is None else float(reward_clip_abs_raw)
    config.setdefault("num_players", 2)
    config.setdefault("board_diagonal", 5)
    config.setdefault("render_mode", None)
    env = sternhalma_v0.discrete_action_env(max_actions=max_actions, **config)
    if reward_scale != 1.0 or reward_clip_abs is not None:
        env = RewardTransformWrapper(
            env,
            reward_scale=reward_scale,
            reward_clip_abs=reward_clip_abs,
        )
    if max_agent_steps is not None:
        env = EpisodeStepLimitWrapper(env, max_agent_steps=int(max_agent_steps))
    return env


def run_aec_episode(
    env: Any,
    policy_fn: PolicyFn,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    env.reset(seed=seed)
    rewards = {agent: 0.0 for agent in env.possible_agents}
    steps = 0

    while env.agents:
        agent_id = env.agent_selection
        observation, reward, terminated, truncated, info = env.last()
        rewards[agent_id] = rewards.get(agent_id, 0.0) + float(reward)

        if terminated or truncated:
            break

        action = policy_fn(agent_id, observation, info, env.action_space(agent_id))

        try:
            env.step(action)
        except ValueError as exc:
            if "only valid action is None" not in str(exc):
                raise
            break
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return {"rewards": rewards, "episode_steps": steps}
