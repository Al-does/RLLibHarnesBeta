# `envs/` — reusable environments and domain logic

Each `envs/<name>/` package implements a reusable environment or benchmark
domain. Environments do not belong inside individual experiments.

## Contract

- Implement the Gymnasium API.
- Accept validated behavior through `env_config` supplied by
  `AlgorithmConfig.environment(...)`.
- Keep defaults explicit and deterministic under a supplied seed.
- Expose correct observation and action spaces.
- Keep reusable filters, wrappers, solvers, and analytic baselines with their
  domain.

An environment must not import the harness, experiments, learners, losses, or
analysis pipeline.

## Configuration

Environment configuration describes simulation behavior, not training:

- dynamics and observation variants;
- episode horizon;
- action constraints;
- optional diagnostic instrumentation.

Algorithm, model, loss, training budget, checkpoint, result-path, phase, and
gate settings do not belong in an environment config.

Avoid top-level special cases in the harness. For example, input scrambling is
an environment config or wrapper selected by an experiment, not a Boolean
field that generic checkpoint or training code interprets.

## Diagnostics

Evaluation may need true latent state, beliefs, or other privileged values.
Expose these through documented public accessors or `info` fields, optionally
behind a diagnostic config when computation is expensive.

Diagnostic data must not silently enter the policy observation. Generic
analysis must not read private attributes such as `_s` or `_filter`.

## Boundaries and tests

Environment packages may contain environment-focused tests and reusable
domain solver tests. Move model, RLModule, Learner, probe-pipeline, and training
tests to their owning packages.

Training loops and supervised workflows do not belong under `envs/`. Keep
those in experiment leaves; promote a generic supervised helper only after
another experiment demonstrates reuse.

## Gol public API and lifecycle

`envs.gol` exports `gol_model(variant=3, speed="half", coarse=False,
initial_distribution=None)`, `controlled_kernels(variant=3, speed="half",
coarse=False)`, and `GolRewardTask(model=..., variant=3, speed="half",
coarse=False)`. Variants 2/3 and half/quarter speeds are frozen; only variant 2
has an exact three-state quotient. Kernels use action/token/source/destination
axes. The model baseline is action a1; its default prior is stationary under
uniform independent actions, not the baseline action. The task validates the
supplied baseline and labels while allowing prior overrides.

Configure `HMMEnv` with model factory `envs.gol:gol_model`, task class
`envs.gol:GolRewardTask`, matching variant/speed/coarse kwargs, and
`reset_emission=False`, `episode_length=None`, `delay=0`. Reset samples only
the prior state, returns zero-padded token/action history, and reports absent
tokens as `None`; filters start at the prior. Each step jointly samples a
token and destination, rewards destination E, and filters on the token only.
Default policy features are token one-hot(2) and previous-action one-hot(4),
without reward, belief, or hidden state. `STATES`, `COARSE_STATES`, `ACTIONS`,
and the fine-to-coarse `AGGREGATION` matrix are exported.

The generic lifecycle defaults remain reset emission enabled and horizon 1024.
An unbounded horizon never truncates and cannot randomize its first length.
No-reset-emission mode currently requires delay zero for either emission kind.

Targeted verification: `uv run pytest -q envs/gol/tests envs/hmm/tests
 envs/wing/tests envs/mess3/tests envs/cassandra_machine/tests`.
