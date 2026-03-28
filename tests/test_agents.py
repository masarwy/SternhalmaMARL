import numpy as np

from agents.baselines.heuristic_agent import HeuristicAgent
from agents.baselines.random_agent import RandomAgent


class DummyActionSpace:
    def sample(self) -> int:
        return 3


def test_random_agent_respects_mask() -> None:
    agent = RandomAgent(seed=123)
    observation = {"action_mask": np.array([0, 1, 0, 1, 0], dtype=np.int8)}
    actions = {agent.act(observation, info={}, action_space=DummyActionSpace()) for _ in range(30)}
    assert actions.issubset({1, 3})


def test_heuristic_agent_prefers_longer_jump_without_board_payload() -> None:
    agent = HeuristicAgent(seed=999)
    observation = {"action_mask": np.array([1, 1, 1], dtype=np.int8)}
    info = {
        "valid_moves": [
            ((6, 8), (6, 7)),  # short move
            ((6, 8), (4, 6)),  # longer jump
            ((5, 7), (4, 7)),  # short move
        ]
    }
    action = agent.act(observation, info=info, action_space=DummyActionSpace())
    assert action == 1


def test_heuristic_agent_scores_multi_hop_by_final_position() -> None:
    """A 3-cell chain jump should score by move[0]->move[-1], not move[0]->move[1]."""
    agent = HeuristicAgent(seed=999)
    observation = {"action_mask": np.array([1, 1], dtype=np.int8)}
    info = {
        "valid_moves": [
            # Single step: net displacement 1
            [(6, 8), (6, 7)],
            # Multi-hop: move[0]->(6,6)->(6,4): net displacement 4 via move[-1]
            [(6, 8), (6, 6), (6, 4)],
        ]
    }
    action = agent.act(observation, info=info, action_space=DummyActionSpace())
    # Multi-hop has larger displacement and should win
    assert action == 1, f"Expected multi-hop (action 1), got {action}"
