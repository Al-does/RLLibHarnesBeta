from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.hmm import ActionDecision, HMMModel, TransitionEvent, compose_hmm_factors
from envs.strata.model import controlled_kernels, strata_model


class StrataRewardTask:
    requires_belief = False

    def __init__(
        self,
        *,
        model: HMMModel,
        reward_state: int = 0,
        rewarded_factors: tuple[int, ...] = (0,),
        alpha: float = 0.97,
        t0: float = 0.38,
        t1: float = 0.54,
        strength: float = 0.15,
    ) -> None:
        self.factor_kernels = controlled_kernels(alpha, t0, t1, strength)
        self.factor_kernels.setflags(write=False)
        self.n_factors = 0
        size = model.n_states
        while size > 1 and size % 3 == 0:
            self.n_factors += 1
            size //= 3
        if size != 1 or self.n_factors == 0:
            raise ValueError("model must be a Cartesian product of Strata factors")
        expected = compose_hmm_factors(
            [strata_model(alpha, t0, t1)] * self.n_factors
        )
        for name in (
            "initial_distribution",
            "transition_matrix",
            "emission_matrix",
            "edge_transition_matrices",
        ):
            actual = getattr(model, name)
            reference = getattr(expected, name)
            if (
                actual is None
                or actual.shape != reference.shape
                or not np.allclose(actual, reference, atol=1e-12, rtol=0.0)
            ):
                raise ValueError(
                    f"model {name} does not match independent Strata factors"
                )
        self._model = model
        if isinstance(reward_state, (bool, np.bool_)) or not isinstance(
            reward_state, (int, np.integer)
        ) or reward_state not in (0, 1, 2):
            raise ValueError("reward_state must be 0, 1, or 2")
        self.reward_state = int(reward_state)
        factors = tuple(rewarded_factors)
        if (
            not factors
            or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < self.n_factors
                for index in factors
            )
            or len(set(factors)) != len(factors)
        ):
            raise ValueError(
                "rewarded_factors must be distinct valid factor indices"
            )
        self.rewarded_factors = tuple(int(index) for index in factors)
        self.action_space = gym.spaces.Discrete(3 ** self.n_factors)
        self.action_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * self.n_factors,),
            dtype=np.float32,
        )

    def reset(self) -> None:
        pass

    def _factor_actions(self, action: int) -> tuple[int, ...]:
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(
            action
        ):
            raise ValueError("action is outside the Cartesian Strata action space")
        return np.unravel_index(int(action), (3,) * self.n_factors)

    def resolve_action(
        self,
        action: int,
        state: int,
        model: HMMModel,
    ) -> ActionDecision:
        del state
        if model is not self._model:
            raise ValueError("task must be used with the model it was constructed for")
        actions = self._factor_actions(action)
        edges = self.factor_kernels[actions[0]]
        for factor_action in actions[1:]:
            edges = np.stack([
                np.kron(left, right)
                for left in edges
                for right in self.factor_kernels[factor_action]
            ])
        return ActionDecision(
            requested_action=int(action),
            executed_action=int(action),
            transition_matrix=edges.sum(axis=0),
            edge_transition_matrices=edges,
        )

    def reward(
        self,
        event: TransitionEvent,
        decision: ActionDecision,
    ) -> tuple[float, dict[str, float]]:
        del decision
        states = np.unravel_index(event.state_after, (3,) * self.n_factors)
        components = {
            f"factor_{index}_reward": float(states[index] == self.reward_state)
            for index in self.rewarded_factors
        }
        reward = float(np.mean(tuple(components.values())))
        return reward, {**components, "reward_state_reward": reward}

    def encode_action(self, executed_action: int) -> np.ndarray:
        actions = self._factor_actions(executed_action)
        return np.eye(3, dtype=np.float32)[list(actions)].reshape(-1)
