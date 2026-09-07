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


def wing_model(alpha: float = 0.94, x: float = 0.4) -> HMMModel:
    alpha = _unit_parameter(alpha, "alpha")
    x = _unit_parameter(x, "x")
    b = (1.0 - alpha) / 2.0
    edges = np.array([
        [[0.0, b, 0.0], [0.0, x * alpha, b / 2.0], [b, 0.0, 0.0]],
        [[alpha, 0.0, b], [b, (1.0 - x) * alpha, b / 2.0], [0.0, b, alpha]],
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
    alpha: float = 0.94,
    x: float = 0.4,
    strength: float = 0.15,
) -> np.ndarray:
    strength = _unit_parameter(strength, "strength")
    edges = wing_model(alpha, x).edge_transition_matrices
    identity = np.eye(3)
    plus = np.roll(identity, 1, axis=1)
    minus = plus.T
    controls = np.stack([
        identity,
        (1.0 - strength) * identity + strength * plus,
        (1.0 - strength) * identity + strength * minus,
    ])
    return edges[None, ...] @ controls[:, None, ...]
