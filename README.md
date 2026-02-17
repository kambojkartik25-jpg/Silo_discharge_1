# silo_blend

Minimal, deterministic Python project for 3-silo malt discharge simulation, valve-time optimization, and HTTP API integration.

## Features
- Beverloo mass flow model per silo.
- Layer-based Gaussian discharge sampler (bottom to top).
- SLSQP valve-time optimizer with optional fixed total mass constraint.
- Monte Carlo randomized search baseline for comparison.
- Bayesian optimization (`gp_minimize`) support.
- FastAPI backend for `/simulate` and `/optimize` endpoints.
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
- `src/silo_blend/bayes_opt.py`: Bayesian optimization.
- `src/silo_blend/compare_optimizers.py`: optimizer benchmark harness.
- `src/silo_blend/cli.py`: command-line entrypoint.
- `src/silo_blend/api/`: FastAPI app, schemas, and services.
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

## API Usage
Start server:

```bash
python -m silo_blend.api
```

or:

```bash
python serve.py
```

OpenAPI docs:
- `http://localhost:8000/docs`
- `http://localhost:8000/openapi.json`

Health check:

```bash
curl http://localhost:8000/health
```

Optimize example:

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "silos": [
    {"silo_id":"S1","capacity_kg":4000,"body_diameter_m":3.0,"outlet_diameter_m":0.2},
    {"silo_id":"S2","capacity_kg":4000,"body_diameter_m":3.2,"outlet_diameter_m":0.2},
    {"silo_id":"S3","capacity_kg":4000,"body_diameter_m":3.1,"outlet_diameter_m":0.21}
  ],
  "layers": [
    {"silo_id":"S1","layer_index":1,"lot_id":"L1001","supplier":"BBM","segment_mass_kg":1200},
    {"silo_id":"S1","layer_index":2,"lot_id":"L1002","supplier":"COFCO","segment_mass_kg":900},
    {"silo_id":"S1","layer_index":3,"lot_id":"L1003","supplier":"Malteurop","segment_mass_kg":700},
    {"silo_id":"S2","layer_index":1,"lot_id":"L1001","supplier":"BBM","segment_mass_kg":1400},
    {"silo_id":"S2","layer_index":2,"lot_id":"L1003","supplier":"Malteurop","segment_mass_kg":1000},
    {"silo_id":"S2","layer_index":3,"lot_id":"L1002","supplier":"COFCO","segment_mass_kg":600},
    {"silo_id":"S3","layer_index":1,"lot_id":"L1002","supplier":"COFCO","segment_mass_kg":700},
    {"silo_id":"S3","layer_index":2,"lot_id":"L1003","supplier":"Malteurop","segment_mass_kg":700}
  ],
  "suppliers": [
    {"supplier":"BBM","moisture_pct":4.2,"fine_extract_db_pct":82.0,"wort_pH":5.98,"diastatic_power_WK":342.1,"total_protein_pct":10.12,"wort_colour_EBC":3.8},
    {"supplier":"COFCO","moisture_pct":4.4,"fine_extract_db_pct":81.8,"wort_pH":5.93,"diastatic_power_WK":317.4,"total_protein_pct":11.1,"wort_colour_EBC":4.0},
    {"supplier":"Malteurop","moisture_pct":4.3,"fine_extract_db_pct":81.2,"wort_pH":5.97,"diastatic_power_WK":336.9,"total_protein_pct":10.5,"wort_colour_EBC":3.8}
  ],
  "material": {"rho_bulk_kg_m3":610.0,"grain_diameter_m":0.004},
  "beverloo": {"C":0.58,"k":1.4,"g_m_s2":9.81},
  "sigma_m": 0.12,
  "steps": 200,
  "auto_adjust": false,
  "target_params": {"moisture_pct":4.28,"wort_pH":5.96},
  "weights": {"moisture_pct":1.0,"wort_pH":5.0},
  "mode": "A",
  "fixed_total_mass_kg": 2500,
  "optimizer": "slsqp",
  "optimizer_settings": {"maxiter": 100, "ftol": 1e-8},
  "return_debug": true
}
JSON
```

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
- Monte Carlo and Bayesian optimization are deterministic for a fixed seed/random_state.
