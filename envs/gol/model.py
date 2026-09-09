from __future__ import annotations

import numpy as np

from envs.hmm import HMMModel
from envs.hmm.model import stationary_distribution


STATES = ("M1", "M2", "E", "S")
COARSE_STATES = ("M", "E", "S")
ACTIONS = ("a1", "a2", "aE", "aS")
AGGREGATION = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
AGGREGATION.setflags(write=False)


def controlled_kernels(
    variant: int = 3, speed: str = "half", coarse: bool = False,
) -> np.ndarray:
    """Return frozen v1 joint probabilities in action/token/source/destination order."""
    if isinstance(variant, (bool, np.bool_)) or not isinstance(variant, (int, np.integer)) or variant not in (2, 3):
        raise ValueError("variant must be 2 or 3")
    if speed not in ("half", "quarter"):
        raise ValueError("speed must be 'half' or 'quarter'")
    if not isinstance(coarse, bool):
        raise TypeError("coarse must be a bool")
    if coarse and variant != 2:
        raise ValueError("variant 3 has no exact coarse quotient")
    h, l = (0.2, 0.05) if speed == "half" else (0.1, 0.025)
    q1 = np.array([h, l, l, l])
    q2 = q1 if variant == 2 else np.array([l, h, l, l])
    e = np.array([.300, .300, .490, .465])
    kernels = np.zeros((4, 2, 4, 4), dtype=np.float64)
    kernels[:, 0, 0, 0] = 1 - q1
    kernels[:, 1, 0, 2] = q1
    kernels[:, 0, 1, 1] = 1 - q2
    kernels[:, 1, 1, 2] = q2
    kernels[:, 0, 2, 2] = e
    kernels[:, 1, 2, 3] = 1 - e
    kernels[:, 0, 3, 0] = [.145, .145, .170, .030]
    kernels[:, 1, 3, 1] = [.400, .400, .020, .070]
    kernels[:, 1, 3, 2] = e
    kernels[:, 0, 3, 3] = [.155, .155, .320, .435]
    if coarse:
        kernels = (kernels @ AGGREGATION)[:, :, [0, 2, 3], :].copy()
    kernels.setflags(write=False)
    return kernels


def gol_model(
    variant: int = 3,
    speed: str = "half",
    coarse: bool = False,
    initial_distribution: np.ndarray | None = None,
) -> HMMModel:
    """Build the a1 baseline with a uniform-action stationary reset prior.

    Compose with GolRewardTask and HMMEnv configured with reset_emission=False,
    delay=0, episode_length=None for the frozen continuing action/token process.
    A supplied prior uses the selected fine or coarse state order.
    """
    kernels = controlled_kernels(variant, speed, coarse)
    if initial_distribution is None:
        fine_kernels = controlled_kernels(variant, speed)
        initial_distribution = stationary_distribution(fine_kernels.sum(axis=1).mean(axis=0))
        if coarse:
            initial_distribution = initial_distribution @ AGGREGATION
    edges = kernels[0]
    return HMMModel(
        initial_distribution=initial_distribution,
        transition_matrix=edges.sum(axis=0),
        emission_matrix=edges.sum(axis=2).T,
        edge_transition_matrices=edges,
        state_labels=COARSE_STATES if coarse else STATES,
        token_labels=("0", "1"),
    )
