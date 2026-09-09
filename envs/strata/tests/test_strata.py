import json

import numpy as np
import pytest

from envs.hmm import HMMEnv, HMMModel, TransitionEvent, compose_hmm_factors
from envs.strata.model import controlled_kernels, strata_model
from envs.strata.tasks.reward_state import StrataRewardTask


def strata_config(n_factors=2, **task_kwargs):
    parameters = {
        key: task_kwargs[key]
        for key in ("alpha", "t0", "t1")
        if key in task_kwargs
    }
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {
                "factors": [
                    {
                        "factory": "envs.strata.model:strata_model",
                        "kwargs": parameters,
                    }
                    for _ in range(n_factors)
                ]
            },
        },
        "task": {
            "class": "envs.strata.tasks.reward_state:StrataRewardTask",
            "kwargs": task_kwargs,
        },
        "observation": {"belief": True},
        "diagnostics": {
            "state": True,
            "belief": True,
            "raw_belief": True,
            "tokens": True,
            "rewards": True,
            "transitions": True,
        },
    }


def test_strata_equation_ten():
    alpha, t0, t1 = 0.98, 0.30, 0.80
    b = (1.0 - alpha) / 2.0
    model = strata_model(alpha, t0, t1)
    expected = np.array([
        [
            [t0 * alpha, 0.0, 0.0],
            [0.0, t1 * alpha, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [(1.0 - t0) * alpha, b, b],
            [b, (1.0 - t1) * alpha, b],
            [b, b, alpha],
        ],
    ])
    np.testing.assert_array_equal(model.edge_transition_matrices, expected)
    np.testing.assert_allclose(model.transition_matrix, expected.sum(axis=0))
    np.testing.assert_allclose(model.emission_matrix, expected.sum(axis=2).T)
    np.testing.assert_allclose(model.transition_matrix.sum(axis=0), 1.0)
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        model.emission_matrix,
        [[t0 * alpha, 1.0 - t0 * alpha], [t1 * alpha, 1.0 - t1 * alpha], [0.0, 1.0]],
    )


@pytest.mark.parametrize("strength", [0.0, 0.15, 1.0])
def test_control_preserves_emissions_and_uniform_stationarity(strength):
    model = strata_model()
    kernels = controlled_kernels(strength=strength)
    assert kernels.shape == (3, 2, 3, 3)
    plus = np.roll(np.eye(3), 1, axis=1)
    for action, permutation in enumerate([np.eye(3), plus, plus.T]):
        control = (1.0 - strength) * np.eye(3) + strength * permutation
        np.testing.assert_allclose(
            kernels[action],
            model.edge_transition_matrices @ control,
        )
        transition = kernels[action].sum(axis=0)
        assert np.all(kernels[action] >= 0.0)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            kernels[action].sum(axis=2).T,
            model.emission_matrix,
        )
        np.testing.assert_allclose(
            np.full(3, 1.0 / 3.0) @ transition,
            np.full(3, 1.0 / 3.0),
        )
    np.testing.assert_allclose(kernels[0], model.edge_transition_matrices)


@pytest.mark.parametrize("name", ["alpha", "t0", "t1", "strength"])
@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf, True])
def test_invalid_probability_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        controlled_kernels(**{name: value})


@pytest.mark.parametrize(
    "alpha,t0,t1",
    [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ],
)
def test_probability_parameter_boundaries(alpha, t0, t1):
    model = strata_model(alpha, t0, t1)
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.emission_matrix.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "n_factors,rewarded_factors",
    [(1, (0,)), (2, (0,)), (2, (0, 1)), (3, (0, 2))],
)
@pytest.mark.parametrize("reward_state", [0, 1, 2])
def test_joint_action_conditioned_bayes_and_arrival_rewards(
    n_factors,
    rewarded_factors,
    reward_state,
):
    alpha, t0, t1, strength = 0.82, 0.31, 0.63, 0.27
    config = strata_config(
        n_factors,
        rewarded_factors=rewarded_factors,
        reward_state=reward_state,
        alpha=alpha,
        t0=t0,
        t1=t1,
        strength=strength,
    )
    json.dumps(config)
    env = HMMEnv(config)
    model = env.model
    assert model.n_states == 3 ** n_factors
    assert model.n_tokens == 2 ** n_factors
    assert env.action_space.n == 3 ** n_factors
    assert env.task.action_observation_space.shape == (3 * n_factors,)
    observation, info = env.reset(seed=97)
    assert env.observation_space.contains(observation)
    belief = (
        model.initial_distribution
        @ model.edge_transition_matrices[info["raw_token_current"]]
    )
    belief /= belief.sum()
    np.testing.assert_allclose(info["belief_current"], belief)
    factor_kernels = controlled_kernels(alpha, t0, t1, strength)
    for action in range(env.action_space.n):
        factor_actions = np.unravel_index(action, (3,) * n_factors)
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert not terminated and not truncated
        token_tuple = np.unravel_index(
            info["raw_token_current"],
            (2,) * n_factors,
        )
        kernel = np.ones((1, 1))
        transition = np.ones((1, 1))
        for factor_action, token in zip(factor_actions, token_tuple):
            kernel = np.kron(
                kernel,
                factor_kernels[factor_action, token],
            )
            transition = np.kron(
                transition,
                factor_kernels[factor_action].sum(axis=0),
            )
        np.testing.assert_allclose(
            info["executed_transition_matrix"],
            transition,
        )
        np.testing.assert_allclose(
            info["executed_edge_transition_matrices"][
                info["raw_token_current"]
            ],
            kernel,
        )
        belief = belief @ kernel
        belief /= belief.sum()
        np.testing.assert_allclose(info["belief_current"], belief, atol=1e-12)
        states = np.unravel_index(
            info["state_after"],
            (3,) * n_factors,
        )
        expected = (
            sum(
                states[index] == reward_state
                for index in rewarded_factors
            )
            / len(rewarded_factors)
        )
        assert reward == expected
        np.testing.assert_array_equal(
            env.task.encode_action(action),
            np.eye(3)[list(factor_actions)].reshape(-1),
        )


@pytest.mark.parametrize(
    "rewarded_factors,expected",
    [((0,), 1.0), ((1,), 0.0), ((0, 1), 0.5)],
)
def test_one_factor_reward_is_unscaled_and_multiple_rewards_are_mean(
    rewarded_factors,
    expected,
):
    model = compose_hmm_factors([strata_model(), strata_model()])
    task = StrataRewardTask(
        model=model,
        reward_state=2,
        rewarded_factors=rewarded_factors,
    )
    event = TransitionEvent(0, 0, 6, 0, 0)
    reward, _ = task.reward(event, task.resolve_action(0, 0, model))
    assert reward == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward_state": -1},
        {"reward_state": 3},
        {"reward_state": 0.5},
        {"reward_state": True},
        {"rewarded_factors": ()},
        {"rewarded_factors": (0, 0)},
        {"rewarded_factors": (2,)},
        {"rewarded_factors": (-1,)},
        {"rewarded_factors": (0.5,)},
        {"rewarded_factors": (True,)},
        {"alpha": 0.7},
        {"t0": 0.2},
        {"t1": 0.2},
    ],
)
def test_task_rejects_invalid_parameters_and_model_mismatch(kwargs):
    model = compose_hmm_factors([strata_model(), strata_model()])
    with pytest.raises(ValueError):
        StrataRewardTask(model=model, **kwargs)


@pytest.mark.parametrize(
    "action",
    [-1, 9, 1.5, True, "1", [1], np.array([1])],
)
def test_task_rejects_invalid_actions(action):
    model = compose_hmm_factors([strata_model(), strata_model()])
    task = StrataRewardTask(model=model)
    with pytest.raises(ValueError, match="action"):
        task.resolve_action(action, 0, model)
    with pytest.raises(ValueError, match="action"):
        task.encode_action(action)


def test_task_rejects_wrong_models():
    model = compose_hmm_factors([strata_model(), strata_model()])
    task = StrataRewardTask(model=model)
    with pytest.raises(ValueError, match="constructed for"):
        task.resolve_action(
            0,
            0,
            compose_hmm_factors([strata_model(), strata_model()]),
        )
    state_model = HMMModel(
        model.initial_distribution,
        model.transition_matrix,
        model.emission_matrix,
    )
    with pytest.raises(ValueError, match="edge_transition_matrices"):
        StrataRewardTask(model=state_model)
    with pytest.raises(ValueError, match="Cartesian product"):
        StrataRewardTask(
            model=HMMModel(np.ones(2) / 2, np.eye(2), np.eye(2))
        )
