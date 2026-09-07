"""Validated probability data for finite discrete HMMs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _probability_array(
    value: np.ndarray,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if (array < 0.0).any():
        raise ValueError(f"{name} must be non-negative")
    return array


def _validate_distribution(value: np.ndarray, *, name: str) -> None:
    if not np.isclose(value.sum(), 1.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")


def _validate_row_stochastic(value: np.ndarray, *, name: str) -> None:
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError(f"each row of {name} must sum to one")


def validate_edge_transition_matrices(
    value: np.ndarray,
    transition_matrix: np.ndarray,
    n_tokens: int,
) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    expected = (n_tokens, *transition_matrix.shape)
    if matrices.shape != expected:
        raise ValueError(f"edge_transition_matrices must have shape {expected}")
    if not np.isfinite(matrices).all() or (matrices < 0.0).any():
        raise ValueError("edge_transition_matrices must be finite and non-negative")
    if not np.allclose(
        matrices.sum(axis=0), transition_matrix, atol=1e-12, rtol=0.0
    ):
        raise ValueError("edge_transition_matrices must sum to transition_matrix")
    return matrices


@dataclass(frozen=True, slots=True)
class HMMModel:
    """Finite HMM definition independent of tasks, actions, and rewards."""

    initial_distribution: np.ndarray
    transition_matrix: np.ndarray
    emission_matrix: np.ndarray
    state_labels: tuple[str, ...] | None = None
    token_labels: tuple[str, ...] | None = None
    edge_transition_matrices: np.ndarray | None = None

    def __post_init__(self) -> None:
        initial = _probability_array(
            self.initial_distribution,
            name="initial_distribution",
            ndim=1,
        )
        transition = _probability_array(
            self.transition_matrix,
            name="transition_matrix",
            ndim=2,
        )
        emission = _probability_array(
            self.emission_matrix,
            name="emission_matrix",
            ndim=2,
        )

        n_states = initial.shape[0]
        if n_states == 0:
            raise ValueError("an HMM must contain at least one state")
        if transition.shape != (n_states, n_states):
            raise ValueError(
                "transition_matrix must have shape "
                f"({n_states}, {n_states}), got {transition.shape}"
            )
        if emission.shape[0] != n_states or emission.shape[1] == 0:
            raise ValueError(
                "emission_matrix must have shape (n_states, n_tokens) with "
                "at least one token"
            )

        _validate_distribution(initial, name="initial_distribution")
        _validate_row_stochastic(transition, name="transition_matrix")
        _validate_row_stochastic(emission, name="emission_matrix")

        if self.state_labels is not None and len(self.state_labels) != n_states:
            raise ValueError("state_labels must match the number of states")
        if (
            self.token_labels is not None
            and len(self.token_labels) != emission.shape[1]
        ):
            raise ValueError("token_labels must match the number of tokens")

        if self.edge_transition_matrices is not None:
            edges = validate_edge_transition_matrices(
                self.edge_transition_matrices, transition, emission.shape[1]
            ).copy()
            if not np.allclose(
                edges.sum(axis=2).T, emission, atol=1e-12, rtol=0.0
            ):
                raise ValueError(
                    "emission_matrix must equal edge_transition_matrices "
                    "summed over destinations and transposed"
                )
            edges.setflags(write=False)
            object.__setattr__(self, "edge_transition_matrices", edges)

        initial.setflags(write=False)
        transition.setflags(write=False)
        emission.setflags(write=False)
        object.__setattr__(self, "initial_distribution", initial)
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "emission_matrix", emission)

    @property
    def n_states(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.emission_matrix.shape[1])


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Return a normalized stationary row distribution."""

    matrix = _probability_array(
        transition_matrix,
        name="transition_matrix",
        ndim=2,
    )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    _validate_row_stochastic(matrix, name="transition_matrix")

    values, vectors = np.linalg.eig(matrix.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    stationary = np.real(vectors[:, index])
    if stationary.sum() < 0.0:
        stationary = -stationary
    stationary = np.maximum(stationary, 0.0)
    total = stationary.sum()
    if total <= 0.0:
        raise ValueError("could not recover a stationary distribution")
    return stationary / total
