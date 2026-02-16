from __future__ import annotations

from pathlib import Path

import pandas as pd

from silo_blend.models import BeverlooParams, Material
from silo_blend.montecarlo import monte_carlo_optimize_valve_times
from silo_blend.optimize import optimize_valve_times


def _report(title: str, res: dict, target: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  success={res['success']} error={res['final_error']:.8f}")
    for sid, t in res["best_times_s"].items():
        print(f"  {sid}: time={t:.2f}s ({t/60:.2f} min), mass={res['best_masses_kg'][sid]:.2f} kg")
    pred = res["best_result"]["total_blended_params"]
    print("  total blended vs target:")
    for p, tv in target.items():
        pv = pred.get(p, float("nan"))
        print(f"    {p}: pred={pv:.5f}, target={tv:.5f}, delta={pv-tv:+.5f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    df_silos = pd.read_csv(data_dir / "sample_silos.csv")
    df_layers = pd.read_csv(data_dir / "sample_layers.csv")
    df_suppliers = pd.read_csv(data_dir / "sample_suppliers.csv")

    material = Material(rho_bulk_kg_m3=610.0, grain_diameter_m=0.004)
    bev = BeverlooParams(C=0.58, k=1.4, g_m_s2=9.81)

    target_params = {
        "moisture_pct": 4.28,
        "fine_extract_db_pct": 81.75,
        "wort_pH": 5.96,
        "diastatic_power_WK": 334.0,
        "total_protein_pct": 10.55,
    }
    weights = {
        "moisture_pct": 1.0,
        "fine_extract_db_pct": 1.0,
        "wort_pH": 5.0,
        "diastatic_power_WK": 0.05,
        "total_protein_pct": 1.5,
    }

    res_a = optimize_valve_times(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target_params,
        weights=weights,
        steps=1000,
        auto_adjust=False,
        fixed_total_mass_kg=2500.0,
        cache=True,
    )
    _report("Mode A - SLSQP", res_a, target_params)

    res_mc = monte_carlo_optimize_valve_times(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target_params,
        weights=weights,
        steps=800,
        auto_adjust=False,
        fixed_total_mass_kg=2500.0,
        n_samples=500,
        seed=123,
    )
    _report("Mode A - Monte Carlo (n=500)", res_mc, target_params)


if __name__ == "__main__":
    main()
