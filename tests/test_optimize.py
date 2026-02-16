from pathlib import Path

import pandas as pd

from silo_blend.models import BeverlooParams, Material
from silo_blend.optimize import optimize_valve_times


def test_optimizer_fixed_mass_in_bounds() -> None:
    root = Path(__file__).resolve().parents[1]
    df_silos = pd.read_csv(root / "data" / "sample_silos.csv")
    df_layers = pd.read_csv(root / "data" / "sample_layers.csv")
    df_suppliers = pd.read_csv(root / "data" / "sample_suppliers.csv")

    material = Material(rho_bulk_kg_m3=610.0, grain_diameter_m=0.004)
    bev = BeverlooParams()

    target = {
        "moisture_pct": 4.28,
        "fine_extract_db_pct": 81.75,
        "wort_pH": 5.96,
    }

    out = optimize_valve_times(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target,
        weights=None,
        steps=200,
        fixed_total_mass_kg=1200.0,
        cache=True,
    )

    assert out["success"]
    for sid, m in out["best_masses_kg"].items():
        assert m >= -1e-9
        assert m <= out["available_mass_kg"][sid] + 1e-9
