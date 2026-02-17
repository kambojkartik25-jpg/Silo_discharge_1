from pathlib import Path

import pandas as pd

from silo_blend.bayes_opt import optimize_valve_times_bayes
from silo_blend.models import BeverlooParams, Material
from silo_blend.optimize import available_mass_per_silo, compute_mdot_per_silo


def test_bayes_mode_a_fixed_mass_feasible() -> None:
    root = Path(__file__).resolve().parents[1]
    df_silos = pd.read_csv(root / "data" / "sample_silos.csv")
    df_layers = pd.read_csv(root / "data" / "sample_layers.csv")
    df_suppliers = pd.read_csv(root / "data" / "sample_suppliers.csv")

    material = Material(rho_bulk_kg_m3=610.0, grain_diameter_m=0.004)
    bev = BeverlooParams()
    target = {"moisture_pct": 4.28, "wort_pH": 5.96}

    fixed_total = 1200.0
    out = optimize_valve_times_bayes(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target,
        steps=120,
        mode="A",
        fixed_total_mass_kg=fixed_total,
        n_calls=16,
        n_initial_points=6,
        random_state=7,
        cache=True,
    )

    assert "bayes" in out["optimizer_name"]
    assert out["success"]
    assert abs(sum(out["best_masses_kg"].values()) - fixed_total) < 1e-5

    mdot = compute_mdot_per_silo(df_silos, material, bev)
    avail = available_mass_per_silo(df_layers)
    for sid, t in out["best_times_s"].items():
        assert t >= -1e-9
        assert out["best_masses_kg"][sid] <= avail[sid] + 1e-9
        assert t <= (avail[sid] / mdot[sid]) + 1e-9


def test_bayes_mode_b_bounds() -> None:
    root = Path(__file__).resolve().parents[1]
    df_silos = pd.read_csv(root / "data" / "sample_silos.csv")
    df_layers = pd.read_csv(root / "data" / "sample_layers.csv")
    df_suppliers = pd.read_csv(root / "data" / "sample_suppliers.csv")

    material = Material(rho_bulk_kg_m3=610.0, grain_diameter_m=0.004)
    bev = BeverlooParams()
    target = {"moisture_pct": 4.28, "wort_pH": 5.96}

    out = optimize_valve_times_bayes(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target,
        steps=120,
        mode="B",
        n_calls=15,
        n_initial_points=5,
        random_state=9,
        cache=True,
    )

    assert out["success"]
    mdot = compute_mdot_per_silo(df_silos, material, bev)
    avail = available_mass_per_silo(df_layers)
    for sid, t in out["best_times_s"].items():
        assert t >= -1e-9
        assert t <= (avail[sid] / mdot[sid]) + 1e-9
        assert out["best_masses_kg"][sid] >= -1e-9
        assert out["best_masses_kg"][sid] <= avail[sid] + 1e-9
