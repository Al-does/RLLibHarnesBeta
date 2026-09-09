from __future__ import annotations

import numpy as np

from envs.hmm import HMMModel


def _unit_parameter(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must lie in [0, 1]")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def strata_model(
    alpha: float = 0.98,
    t0: float = 0.30,
    t1: float = 0.80,
) -> HMMModel:
    alpha = _unit_parameter(alpha, "alpha")
    t0 = _unit_parameter(t0, "t0")
    t1 = _unit_parameter(t1, "t1")
    b = (1.0 - alpha) / 2.0
    edges = np.array([
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
    return HMMModel(
        initial_distribution=np.full(3, 1.0 / 3.0),
        transition_matrix=edges.sum(axis=0),
        emission_matrix=edges.sum(axis=2).T,
        state_labels=("0", "1", "2"),
        token_labels=("0", "1"),
        edge_transition_matrices=edges,
    )


def controlled_kernels(
    alpha: float = 0.98,
    t0: float = 0.30,
    t1: float = 0.80,
    strength: float = 0.15,
) -> np.ndarray:
    strength = _unit_parameter(strength, "strength")
    edges = strata_model(alpha, t0, t1).edge_transition_matrices
    identity = np.eye(3)
    plus = np.roll(identity, 1, axis=1)
    minus = plus.T
    controls = np.stack([
        identity,
        (1.0 - strength) * identity + strength * plus,
        (1.0 - strength) * identity + strength * minus,
    ])
    return edges[None, ...] @ controls[:, None, ...]
