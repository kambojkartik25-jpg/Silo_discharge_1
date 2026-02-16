from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import BeverlooParams, Material
from .optimize import available_mass_per_silo, compute_mdot_per_silo
from .simulate import run_three_silo_blend


def monte_carlo_optimize_valve_times(
    df_silos: pd.DataFrame,
    df_layers: pd.DataFrame,
    df_suppliers: pd.DataFrame,
    material: Material,
    bev: BeverlooParams,
    sigma_m: float,
    target_params: dict[str, float],
    weights: dict[str, float] | None = None,
    steps: int = 1200,
    auto_adjust: bool = False,
    fixed_total_mass_kg: float | None = None,
    n_samples: int = 2000,
    seed: int = 123,
) -> dict[str, Any]:
    """Randomized baseline optimizer for valve times."""

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not target_params:
        raise ValueError("target_params must not be empty")

    weights = weights or {}
    rng = np.random.default_rng(seed)

    silo_ids = df_silos["silo_id"].astype(str).tolist()
    mdot_map = compute_mdot_per_silo(df_silos, material, bev)
    avail_map = available_mass_per_silo(df_layers)

    mdot_vec = np.array([mdot_map[s] for s in silo_ids], dtype=float)
    avail_vec = np.array([float(avail_map.get(s, 0.0)) for s in silo_ids], dtype=float)
    ub_times = avail_vec / mdot_vec

    penalty = 1e12
    best_error = float("inf")
    best_masses = np.zeros_like(avail_vec)
    best_times = np.zeros_like(ub_times)
    best_result: dict[str, Any] | None = None

    def score(masses: np.ndarray) -> tuple[float, dict[str, Any]]:
        df_discharge = pd.DataFrame({"silo_id": silo_ids, "discharge_mass_kg": masses})
        result = run_three_silo_blend(
            df_silos=df_silos,
            df_layers=df_layers,
            df_suppliers=df_suppliers,
            df_discharge=df_discharge,
            material=material,
            bev=bev,
            sigma_m=sigma_m,
            steps=steps,
            auto_adjust=auto_adjust,
        )
        pred = result["total_blended_params"]
        err = 0.0
        for p, target in target_params.items():
            w = float(weights.get(p, 1.0))
            v = pred.get(p, np.nan)
            if v is None or not np.isfinite(v):
                err += penalty * w
            else:
                err += w * (float(v) - float(target)) ** 2
        return float(err), result

    for _ in range(n_samples):
        if fixed_total_mass_kg is not None:
            alpha = np.ones(len(silo_ids), dtype=float)
            split = rng.dirichlet(alpha)
            masses = split * fixed_total_mass_kg
            masses = np.minimum(masses, avail_vec)

            # If clipping reduced total mass, redistribute remainder among available silos.
            rem = fixed_total_mass_kg - float(masses.sum())
            if rem > 1e-9:
                headroom = np.clip(avail_vec - masses, 0.0, None)
                hs = float(headroom.sum())
                if hs > 0.0:
                    masses += rem * (headroom / hs)
            masses = np.minimum(masses, avail_vec)
        else:
            times = rng.uniform(0.0, ub_times)
            masses = np.clip(times * mdot_vec, 0.0, avail_vec)

        err, result = score(masses)
        if err < best_error:
            best_error = err
            best_masses = masses.copy()
            best_times = np.clip(best_masses / mdot_vec, 0.0, ub_times)
            best_result = result

    if best_result is None:
        raise RuntimeError("Monte Carlo search failed to produce any candidate")

    mode = "A_fixed_total_mass" if fixed_total_mass_kg is not None else "B_free_total_mass"
    return {
        "mode": mode,
        "success": True,
        "message": "Monte Carlo completed",
        "final_error": float(best_error),
        "best_times_s": dict(zip(silo_ids, best_times.tolist())),
        "best_masses_kg": dict(zip(silo_ids, best_masses.tolist())),
        "mdot_kg_s": dict(zip(silo_ids, mdot_vec.tolist())),
        "available_mass_kg": dict(zip(silo_ids, avail_vec.tolist())),
        "best_result": best_result,
        "optimizer_result": {"n_samples": n_samples, "seed": seed},
    }
