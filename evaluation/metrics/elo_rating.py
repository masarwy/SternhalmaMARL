"""Elo rating utilities for head-to-head evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute the expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloTracker:
    """Mutable Elo tracker with standard update rules."""

    initial_rating: float = 1000.0
    k_factor: float = 32.0
    ratings: dict[str, float] = field(default_factory=dict)

    def ensure(self, agent_name: str) -> float:
        if agent_name not in self.ratings:
            self.ratings[agent_name] = float(self.initial_rating)
        return self.ratings[agent_name]

    def record_game(self, agent_a: str, agent_b: str, score_a: float) -> None:
        """Record one game where score_a is 1.0 win, 0.5 draw, 0.0 loss."""
        score_a = float(score_a)
        score_b = 1.0 - score_a

        rating_a = self.ensure(agent_a)
        rating_b = self.ensure(agent_b)

        exp_a = expected_score(rating_a, rating_b)
        exp_b = 1.0 - exp_a

        self.ratings[agent_a] = rating_a + self.k_factor * (score_a - exp_a)
        self.ratings[agent_b] = rating_b + self.k_factor * (score_b - exp_b)
