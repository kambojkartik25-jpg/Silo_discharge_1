from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .models import BeverlooParams, Material
from .montecarlo import monte_carlo_optimize_valve_times
from .optimize import optimize_valve_times


def _load_sample_data(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_silos = pd.read_csv(base / "sample_silos.csv")
    df_layers = pd.read_csv(base / "sample_layers.csv")
    df_suppliers = pd.read_csv(base / "sample_suppliers.csv")
    return df_silos, df_layers, df_suppliers


def _short_report(name: str, res: dict, target_params: dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"  success={res['success']} err={res['final_error']:.8f}")
    for sid, t in res["best_times_s"].items():
        print(f"  {sid}: {t:.2f}s ({t/60.0:.2f} min), mass={res['best_masses_kg'][sid]:.2f} kg")
    pred = res["best_result"]["total_blended_params"]
    for p, tv in target_params.items():
        pv = pred.get(p, float("nan"))
        print(f"  {p}: pred={pv:.5f} target={tv:.5f} delta={pv-tv:+.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize 3-silo valve open times")
    parser.add_argument("--mode", choices=["A", "B"], default="A")
    parser.add_argument("--samples", type=int, default=500, help="Monte Carlo samples")
    parser.add_argument("--steps", type=int, default=800, help="Simulation time steps")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[2] / "data"
    df_silos, df_layers, df_suppliers = _load_sample_data(data_dir)

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

    fixed_total = 2500.0 if args.mode == "A" else None

    t0 = time.perf_counter()
    res_slsqp = optimize_valve_times(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target_params,
        weights=weights,
        steps=args.steps,
        auto_adjust=False,
        fixed_total_mass_kg=fixed_total,
        cache=True,
    )
    t1 = time.perf_counter()

    res_mc = monte_carlo_optimize_valve_times(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        material=material,
        bev=bev,
        sigma_m=0.12,
        target_params=target_params,
        weights=weights,
        steps=args.steps,
        auto_adjust=False,
        fixed_total_mass_kg=fixed_total,
        n_samples=args.samples,
        seed=123,
    )
    t2 = time.perf_counter()

    print(f"Mode: {args.mode} ({'fixed mass' if args.mode == 'A' else 'free mass'})")
    _short_report("SLSQP", res_slsqp, target_params)
    print(f"  runtime={t1 - t0:.3f}s")
    _short_report("Monte Carlo", res_mc, target_params)
    print(f"  runtime={t2 - t1:.3f}s")

    results = {
        "mode": args.mode,
        "slsqp": {
            "final_error": res_slsqp["final_error"],
            "best_times_s": res_slsqp["best_times_s"],
            "best_masses_kg": res_slsqp["best_masses_kg"],
            "runtime_s": t1 - t0,
        },
        "montecarlo": {
            "final_error": res_mc["final_error"],
            "best_times_s": res_mc["best_times_s"],
            "best_masses_kg": res_mc["best_masses_kg"],
            "runtime_s": t2 - t1,
            "n_samples": args.samples,
        },
    }

    out_path = Path.cwd() / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
