from dataclasses import replace
from itertools import product

import gymnasium as gym
import numpy as np
import pytest

from envs.gol import ACTIONS, AGGREGATION, STATES, GolRewardTask, controlled_kernels, gol_model
from envs.hmm import DiagnosticsConfig, HMMEnv, HMMModel, condition_edge


CASES = [(variant, speed, coarse) for variant in (2, 3) for speed in ("half", "quarter") for coarse in (False, True) if variant == 2 or not coarse]
EDGE_TOKENS = np.array([[0, -1, 1, -1], [-1, 0, 1, -1], [-1, -1, 0, 1], [0, 1, 1, 0]])


def config(variant=3, speed="half", coarse=False, initial_distribution=None):
    kwargs = dict(variant=variant, speed=speed, coarse=coarse)
    return {
        "model": {"factory": "envs.gol:gol_model", "kwargs": {**kwargs, "initial_distribution": initial_distribution}},
        "task": {"class": "envs.gol:GolRewardTask", "kwargs": kwargs},
        "reset_emission": False,
        "episode_length": None,
        "delay": 0,
        "diagnostics": DiagnosticsConfig.full(),
    }


@pytest.mark.parametrize("variant,speed,coarse", CASES)
def test_frozen_kernels_prior_and_task(variant, speed, coarse):
    model = gol_model(variant, speed, coarse)
    kernels = controlled_kernels(variant, speed, coarse)
    n = 3 if coarse else 4
    assert isinstance(model, HMMModel)
    assert kernels.shape == (4, 2, n, n)
    assert STATES == ("M1", "M2", "E", "S")
    assert ACTIONS == ("a1", "a2", "aE", "aS")
    np.testing.assert_allclose(kernels.sum(axis=(1, 3)), 1)
    assert np.min(kernels) >= 0
    h, l = (0.2, 0.05) if speed == "half" else (0.1, 0.025)
    fine = np.zeros((4, 2, 4, 4))
    for a, e, u, v, z in zip(range(4), [.3, .3, .49, .465], [.145, .145, .17, .03], [.4, .4, .02, .07], [.155, .155, .32, .435]):
        q1 = h if a == 0 else l
        q2 = h if a == (0 if variant == 2 else 1) else l
        fine[a, 0] = [[1-q1, 0, 0, 0], [0, 1-q2, 0, 0], [0, 0, e, 0], [u, 0, 0, z]]
        fine[a, 1] = [[0, 0, q1, 0], [0, 0, q2, 0], [0, 0, 0, 1-e], [0, v, e, 0]]
    expected = (fine @ AGGREGATION)[:, :, [0, 2, 3], :] if coarse else fine
    np.testing.assert_array_equal(kernels, expected)
    prior = np.array([.227891803466, .413925928744, .195402408172, .162779859618] if speed == "half" else [.277609138938, .504228844193, .119015895820, .099146121049])
    np.testing.assert_allclose(model.initial_distribution, prior @ AGGREGATION if coarse else prior, atol=1e-12)
    np.testing.assert_allclose(model.initial_distribution @ kernels.sum(axis=1).mean(axis=0), model.initial_distribution, atol=1e-14)
    np.testing.assert_array_equal(model.edge_transition_matrices, kernels[0])
    task = GolRewardTask(model=model, variant=variant, speed=speed, coarse=coarse)
    assert isinstance(task.action_space, gym.spaces.Discrete) and task.action_space.n == 4
    assert task.action_observation_space.shape == (4,)
    for a in range(4):
        decision = task.resolve_action(a, 0, model)
        np.testing.assert_array_equal(decision.edge_transition_matrices, kernels[a])
        np.testing.assert_array_equal(decision.transition_matrix, kernels[a].sum(axis=0))
        np.testing.assert_array_equal(task.encode_action(a), np.eye(4)[a])


@pytest.mark.parametrize("speed", ["half", "quarter"])
def test_exact_quotient_and_filter(speed):
    fine, coarse = controlled_kernels(2, speed), controlled_kernels(2, speed, True)
    np.testing.assert_allclose(fine @ AGGREGATION, AGGREGATION @ coarse, atol=1e-14)
    assert coarse[3, 0, 2, 0] == .03 and coarse[3, 1, 2, 0] == .07
    b = np.array([.1, .3, .4, .2])
    c = b @ AGGREGATION
    for a, x in list(product(range(4), range(2))) * 10:
        b, c = condition_edge(b, fine[a], x), condition_edge(c, coarse[a], x)
        np.testing.assert_allclose(b @ AGGREGATION, c, atol=1e-14)
        np.testing.assert_allclose(b @ fine.sum(axis=1)[:, :, 2].T, c @ coarse.sum(axis=1)[:, :, 1].T)


@pytest.mark.parametrize("variant,speed", list(product((2, 3), ("half", "quarter"))))
def test_filter_matches_enumerated_hidden_paths(variant, speed):
    kernels = controlled_kernels(variant, speed)
    transition = kernels.sum(axis=1)
    prior = gol_model(variant, speed).initial_distribution
    actions, tokens = (0, 3, 2, 1), (0, 1, 1, 0)
    truth = np.zeros(4)
    for path in product(range(4), repeat=5):
        weight = prior[path[0]]
        for t, (a, x) in enumerate(zip(actions, tokens)):
            i, j = path[t:t+2]
            weight *= transition[a, i, j] * (EDGE_TOKENS[i, j] == x)
        truth[path[-1]] += weight
    belief = prior
    for a, x in zip(actions, tokens):
        belief = condition_edge(belief, kernels[a], x)
    np.testing.assert_allclose(belief, truth / truth.sum(), atol=1e-14)


@pytest.mark.parametrize("variant,speed,coarse", CASES)
def test_environment_joint_edges_destination_reward_token_only_belief(variant, speed, coarse):
    prior = [0, 0, 1] if coarse else [0, 0, 0, 1]
    env = HMMEnv(config(variant, speed, coarse, prior))
    obs, info = env.reset(seed=21)
    np.testing.assert_array_equal(obs, np.zeros(6))
    assert info["raw_token_current"] is None
    np.testing.assert_array_equal(info["belief_current"], prior)
    belief = np.array(prior)
    kernels = controlled_kernels(variant, speed, coarse)
    reward_state = 1 if coarse else 2
    for step in range(1200):
        a = step % 4
        obs, reward, terminated, truncated, info = env.step(a)
        x, i, j = info["raw_token_after"], info["state_before"], info["state_after"]
        assert kernels[a, x, i, j] > 0
        assert reward == float(j == reward_state)
        assert not terminated and not truncated
        belief = condition_edge(belief, kernels[a], x)
        np.testing.assert_allclose(info["belief_current"], belief)
        np.testing.assert_allclose(info["raw_belief_current"], belief)
        np.testing.assert_array_equal(obs, np.r_[np.eye(2)[x], np.eye(4)[a]])
    assert 0 < belief[reward_state] < 1


def test_model_task_mismatch_validation_and_prior_override():
    model = gol_model(initial_distribution=[1, 0, 0, 0])
    GolRewardTask(model=model)
    for bad in [gol_model(2), gol_model(speed="quarter"), gol_model(2, coarse=True), replace(model, state_labels=("S", "M1", "M2", "E")), replace(model, edge_transition_matrices=None)]:
        with pytest.raises(ValueError, match="model"):
            GolRewardTask(model=bad)
    task = GolRewardTask(model=model)
    for action in [-1, 4, True, 1.0, "1"]:
        with pytest.raises(ValueError, match="action"):
            task.resolve_action(action, 0, model)
        with pytest.raises(ValueError, match="action"):
            task.encode_action(action)


@pytest.mark.parametrize("kwargs", [{"variant": 1}, {"variant": 2.0}, {"speed": "full"}, {"coarse": True}, {"coarse": 1}])
def test_invalid_model_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        gol_model(**kwargs)
    with pytest.raises((ValueError, TypeError)):
        controlled_kernels(**kwargs)


@pytest.mark.parametrize("prior", [[1, 0], [0, 0, 0, 0], [-1, 1, 1, 0], [np.nan, 0, 0, 1]])
def test_invalid_prior(prior):
    with pytest.raises(ValueError):
        gol_model(initial_distribution=prior)
