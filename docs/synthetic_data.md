# Synthetic Data

`healthpipe` includes synthetic-data tooling for teams that need realistic but
lower-risk datasets for development, testing, analytics prototyping, or demos.

The public entry point is `synthesize(...)`, backed by
`healthpipe.synthetic.generator`.

## Recommended Flow

The intended sequence is:

1. ingest source data
2. de-identify the dataset
3. generate synthetic records from the de-identified dataset
4. validate privacy risk and utility before reuse

Example:

```python
import asyncio
import healthpipe as hp


async def main() -> None:
    dataset = await hp.ingest([hp.CSVSource(path="patients.csv")])
    deidentified = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift=True,
        date_shift_salt="secret-salt",
    )
    synthetic = await hp.synthesize(
        deidentified,
        n_patients=5_000,
        method="gaussian_copula",
        validate=True,
    )
    utility = hp.evaluate_utility(synthetic, deidentified)
    print(utility.fidelity)


asyncio.run(main())
```

## Available Components

The synthetic-data stack is split across:

- `healthpipe.synthetic.generator`
- `healthpipe.synthetic.validator`
- `healthpipe.synthetic.utility`

Public objects include:

- `SyntheticGenerator`
- `synthesize`
- `ReidentificationValidator`
- `evaluate_utility`
- `UtilityReport`

## Generation Modes

The package is designed to support structured record generation from a
de-identified input dataset. The README and tests center on
`method="gaussian_copula"`, and the package also exposes optional synthetic
extras for richer generators.

Install extras as needed:

```bash
pip install healthpipe[synthetic]
```

## Validation

Synthetic output is only useful if it is both:

- statistically similar enough for the intended task
- unlikely to reproduce or expose original patient identities

The built-in validation tooling focuses on those two axes:

- `ReidentificationValidator` checks re-identification style risk
- `evaluate_utility(...)` compares synthetic and source distributions

Treat generation and validation as a pair. Do not ship synthetic output without
running both.

## What Synthetic Data Is Good For

- local application development
- test fixtures with plausible distributions
- analytics notebooks and dashboards
- product demos that should not use live patient records

## What It Is Not

- a guarantee that all privacy risk is eliminated
- a substitute for de-identification of the source dataset
- a drop-in replacement for every model-training workload

The safest posture is still:

1. de-identify first
2. synthesize second
3. validate third
4. document intended use

## Practical Guidance

- start from a representative de-identified seed dataset
- inspect fidelity metrics before downstream adoption
- set `validate=True` in generation workflows
- keep audit artifacts if synthetic data is shared across teams
