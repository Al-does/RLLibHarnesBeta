from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent
from envs.gol.model import controlled_kernels, gol_model


class GolRewardTask:
    """Four controlled joint edge laws with unit reward on arrival in E."""

    requires_belief = False

    def __init__(
        self, *, model: HMMModel, variant: int = 3,
        speed: str = "half", coarse: bool = False,
    ) -> None:
        expected = gol_model(variant, speed, coarse)
        for name in ("transition_matrix", "emission_matrix", "edge_transition_matrices"):
            actual, reference = getattr(model, name), getattr(expected, name)
            if actual is None or actual.shape != reference.shape or not np.allclose(
                actual, reference, atol=1e-12, rtol=0.0,
            ):
                raise ValueError(f"model {name} does not match the selected gol variant/speed/coarse")
        for name in ("state_labels", "token_labels"):
            if getattr(model, name) != getattr(expected, name):
                raise ValueError(f"model {name} does not match the selected gol ordering")
        self._model = model
        self.kernels = controlled_kernels(variant, speed, coarse)
        self.transition_matrices = self.kernels.sum(axis=1)
        self.transition_matrices.setflags(write=False)
        self.reward_state = 1 if coarse else 2
        self.action_space = gym.spaces.Discrete(4)
        self.action_observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def _action(self, action: int) -> int:
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("action must be an integer from 0 to 3")
        return int(action)

    def resolve_action(self, action: int, state: int, model: HMMModel) -> ActionDecision:
        if model is not self._model:
            raise ValueError("task must be used with the model it was constructed for")
        selected = self._action(action)
        return ActionDecision(
            requested_action=selected,
            executed_action=selected,
            transition_matrix=self.transition_matrices[selected],
            edge_transition_matrices=self.kernels[selected],
        )

    def reward(
        self, event: TransitionEvent, decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        reward = float(event.state_after == self.reward_state)
        return reward, {"reward_state_reward": reward}

    def encode_action(self, executed_action: int) -> np.ndarray:
        return np.eye(4, dtype=np.float32)[self._action(executed_action)]
