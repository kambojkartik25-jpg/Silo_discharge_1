# silo_blend

Minimal, deterministic Python project for 3-silo malt discharge simulation and valve-time optimization.

## Features
- Beverloo mass flow model per silo.
- Layer-based Gaussian discharge sampler (bottom to top).
- SLSQP valve-time optimizer with optional fixed total mass constraint.
- Monte Carlo randomized search baseline for comparison.
- CLI runner, example script, sample CSV data, and unit tests.

## Environment
- Python 3.14.2
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Layout
- `src/silo_blend/models.py`: dataclasses and `normal_cdf`.
- `src/silo_blend/simulate.py`: discharge simulation and blending.
- `src/silo_blend/optimize.py`: SLSQP optimizer.
- `src/silo_blend/montecarlo.py`: Monte Carlo optimizer.
- `src/silo_blend/cli.py`: command-line entrypoint.
- `examples/run_example.py`: runnable example.
- `data/sample_*.csv`: sample suppliers, silos, and layers.
- `tests/`: unit tests.

## Run Example

```bash
python examples/run_example.py
```

## Run CLI
Mode A (fixed total mass):

```bash
python -m silo_blend.cli --mode A --samples 500 --steps 800
```

Mode B (free mass):

```bash
python -m silo_blend.cli --mode B --samples 500 --steps 800
```

CLI writes `results.json` in the current working directory.

## Run Tests

```bash
pytest
```

## Dev Formatting

```bash
black src tests examples
```

## Notes
- All units are SI (kg, m, s).
- Gaussian sampler uses `sigma_m` as local mixing width around the moving discharge front.
- Monte Carlo is deterministic for a fixed `seed`.
