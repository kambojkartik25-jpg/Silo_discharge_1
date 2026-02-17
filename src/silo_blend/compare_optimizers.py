from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .bayes_opt import optimize_valve_times_bayes
from .models import BeverlooParams, Material
from .montecarlo import monte_carlo_optimize_valve_times
from .optimize import optimize_valve_times


def compare_optimizers(
    optimizers: list[str],
    *,
    mode: str,
    df_silos: pd.DataFrame,
    df_layers: pd.DataFrame,
    df_suppliers: pd.DataFrame,
    material: Material,
    bev: BeverlooParams,
    sigma_m: float,
    target_params: dict[str, float],
    weights: dict[str, float] | None = None,
    steps: int = 400,
    auto_adjust: bool = False,
    fixed_total_mass_kg: float | None = None,
    n_calls: int = 60,
    n_initial_points: int = 15,
    random_state: int = 42,
    n_samples: int = 500,
    seed: int = 123,
    write_outputs: bool = False,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Run multiple optimizers on same inputs and return a comparison table."""

    weights = weights or {}
    rows: list[dict[str, Any]] = []

    for name in optimizers:
        t0 = time.perf_counter()
        lower = name.lower()

        if lower in {"slsqp", "optimize", "slsqp_opt"}:
            result = optimize_valve_times(
                df_silos=df_silos,
                df_layers=df_layers,
                df_suppliers=df_suppliers,
                material=material,
                bev=bev,
                sigma_m=sigma_m,
                target_params=target_params,
                weights=weights,
                steps=steps,
                auto_adjust=auto_adjust,
                fixed_total_mass_kg=fixed_total_mass_kg if mode == "A" else None,
                cache=True,
            )
            opt_name = "slsqp"
        elif lower in {"montecarlo", "mc"}:
            result = monte_carlo_optimize_valve_times(
                df_silos=df_silos,
                df_layers=df_layers,
                df_suppliers=df_suppliers,
                material=material,
                bev=bev,
                sigma_m=sigma_m,
                target_params=target_params,
                weights=weights,
                steps=steps,
                auto_adjust=auto_adjust,
                fixed_total_mass_kg=fixed_total_mass_kg if mode == "A" else None,
                n_samples=n_samples,
                seed=seed,
            )
            opt_name = "montecarlo"
        elif lower in {"bayes", "bayes_opt", "bayesian"}:
            result = optimize_valve_times_bayes(
                df_silos=df_silos,
                df_layers=df_layers,
                df_suppliers=df_suppliers,
                material=material,
                bev=bev,
                sigma_m=sigma_m,
                target_params=target_params,
                weights=weights,
                steps=steps,
                auto_adjust=auto_adjust,
                mode=mode,
                fixed_total_mass_kg=fixed_total_mass_kg if mode == "A" else None,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                random_state=random_state,
                cache=True,
            )
            opt_name = "bayes_opt_gp_minimize"
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

        t1 = time.perf_counter()
        total_mass = float(sum(result["best_masses_kg"].values()))
        row: dict[str, Any] = {
            "optimizer_name": opt_name,
            "success": bool(result.get("success", False)),
            "final_error": float(result.get("final_error", float("inf"))),
            "total_mass": total_mass,
            "wall_time_s": t1 - t0,
            "best_times_s": result.get("best_times_s", {}),
            "best_masses_kg": result.get("best_masses_kg", {}),
        }
        predicted = result["best_result"]["total_blended_params"]
        for k, v in predicted.items():
            row[f"pred_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["final_error", "wall_time_s"]).reset_index(drop=True)

    if write_outputs:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "optimizer_comparison.csv", index=False)
        (out / "optimizer_comparison.json").write_text(
            json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
        )

    return df
