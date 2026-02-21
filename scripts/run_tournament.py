"""Round-robin tournament runner with Elo ratings."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baselines.heuristic_agent import HeuristicAgent
from agents.baselines.random_agent import RandomAgent
from evaluation.metrics.elo_rating import EloTracker
from training.utils import load_yaml, make_env, run_aec_episode


@dataclass
class MatchStats:
    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0


def _load_checkpoint_agent_type(spec: str) -> str:
    path = Path(spec)
    if not path.exists():
        return "random"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "random"
    algo = str(payload.get("algorithm", "")).lower()
    if "heuristic" in algo:
        return "heuristic"
    return "random"


def _build_agent(spec: str, seed: int):
    if spec == "heuristic":
        return HeuristicAgent(seed=seed)
    if spec == "random":
        return RandomAgent(seed=seed)
    inferred = _load_checkpoint_agent_type(spec)
    return HeuristicAgent(seed=seed) if inferred == "heuristic" else RandomAgent(seed=seed)


def _decide_winner(rewards: dict[str, float], players: list[str]) -> str | None:
    p0, p1 = players
    r0 = float(rewards.get(p0, 0.0))
    r1 = float(rewards.get(p1, 0.0))
    if r0 == r1:
        return None
    return p0 if r0 > r1 else p1


def run_tournament(
    agents: list[str],
    num_games: int,
    env_config: dict[str, Any],
    seed: int = 42,
    max_steps_per_episode: int | None = None,
) -> dict[str, Any]:
    if len(agents) < 2:
        raise ValueError("At least two agents are required for a tournament.")
    if int(env_config.get("num_players", 2)) != 2:
        raise ValueError("Tournament runner currently supports num_players=2.")

    elo = EloTracker()
    pair_stats: dict[str, MatchStats] = {}

    for idx_a, idx_b in itertools.combinations(range(len(agents)), 2):
        name_a = agents[idx_a]
        name_b = agents[idx_b]
        key = f"{name_a}_vs_{name_b}"
        stats = MatchStats()

        for game_idx in range(num_games):
            swap = game_idx % 2 == 1
            first, second = (name_b, name_a) if swap else (name_a, name_b)

            env = make_env(env_config)
            game_agents = {
                env.possible_agents[0]: _build_agent(first, seed=seed + game_idx + idx_a),
                env.possible_agents[1]: _build_agent(second, seed=seed + game_idx + idx_b),
            }

            def policy_fn(agent_id: str, observation: Any, info: dict[str, Any], action_space: Any) -> int | None:
                return game_agents[agent_id].act(observation, info, action_space)

            out = run_aec_episode(
                env=env,
                policy_fn=policy_fn,
                seed=seed + (idx_a * 1000) + (idx_b * 100) + game_idx,
                max_steps=max_steps_per_episode,
            )
            env.close()

            winner_slot = _decide_winner(out["rewards"], list(game_agents.keys()))

            stats.games += 1
            if winner_slot is None:
                stats.draws += 1
                elo.record_game(name_a, name_b, 0.5)
            else:
                winner_global = first if winner_slot == env.possible_agents[0] else second
                if winner_global == name_a:
                    stats.wins_a += 1
                    elo.record_game(name_a, name_b, 1.0)
                else:
                    stats.wins_b += 1
                    elo.record_game(name_a, name_b, 0.0)

        pair_stats[key] = stats

    ratings = dict(sorted(elo.ratings.items(), key=lambda kv: kv[1], reverse=True))
    matches = {
        pair: {
            "games": stats.games,
            "wins_first_agent": stats.wins_a,
            "wins_second_agent": stats.wins_b,
            "draws": stats.draws,
        }
        for pair, stats in pair_stats.items()
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agents": agents,
        "ratings": ratings,
        "matches": matches,
        "num_games_per_pair": num_games,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Sternhalma round-robin tournament.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/tournament.yaml"),
        help="Tournament config path.",
    )
    parser.add_argument("--agents", nargs="+", default=None, help="Agent names or checkpoint JSON paths.")
    parser.add_argument("--num_games", type=int, default=None, help="Games per pair override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    env_config = dict(config.get("env_config", {}))
    tournament_cfg = dict(config.get("tournament_config", {}))

    agents = args.agents or list(tournament_cfg.get("agents", ["random", "heuristic"]))
    num_games = args.num_games if args.num_games is not None else int(tournament_cfg.get("num_games", 20))
    seed = args.seed if args.seed is not None else int(tournament_cfg.get("seed", 42))
    max_steps = tournament_cfg.get("max_steps_per_episode")

    result = run_tournament(
        agents=agents,
        num_games=num_games,
        env_config=env_config,
        seed=seed,
        max_steps_per_episode=max_steps,
    )

    output = args.output or Path("experiments/results") / f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote tournament results to {output}")
    print(json.dumps(result["ratings"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
