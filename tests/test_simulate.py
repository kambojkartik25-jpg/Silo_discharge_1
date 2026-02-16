from pathlib import Path

import pandas as pd

from silo_blend.models import BeverlooParams, Material
from silo_blend.simulate import run_three_silo_blend


def test_run_three_silo_blend_returns_expected_shapes() -> None:
    root = Path(__file__).resolve().parents[1]
    df_silos = pd.read_csv(root / "data" / "sample_silos.csv")
    df_layers = pd.read_csv(root / "data" / "sample_layers.csv")
    df_suppliers = pd.read_csv(root / "data" / "sample_suppliers.csv")

    df_discharge = pd.DataFrame(
        [
            {"silo_id": "S1", "discharge_mass_kg": 1000.0},
            {"silo_id": "S2", "discharge_mass_kg": 900.0},
            {"silo_id": "S3", "discharge_mass_kg": 600.0},
        ]
    )

    material = Material(rho_bulk_kg_m3=610.0, grain_diameter_m=0.004)
    bev = BeverlooParams()

    out = run_three_silo_blend(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        df_discharge=df_discharge,
        material=material,
        bev=bev,
        sigma_m=0.12,
        steps=100,
        auto_adjust=False,
    )

    expected = {"moisture_pct", "fine_extract_db_pct", "wort_pH", "diastatic_power_WK"}
    assert expected.issubset(set(out["total_blended_params"].keys()))

    expected_mass = float(df_discharge["discharge_mass_kg"].sum())
    assert abs(out["total_discharged_mass_kg"] - expected_mass) < 1e-6
