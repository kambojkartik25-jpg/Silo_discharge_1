from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize

from .models import BeverlooParams, Material, Silo
from .simulate import beverloo_mass_flow_rate_kg_s, run_three_silo_blend


def compute_mdot_per_silo(
    df_silos: pd.DataFrame, material: Material, bev: BeverlooParams
) -> dict[str, float]:
    """Compute Beverloo mass flow rate for each silo."""

    required = {"silo_id", "capacity_kg", "body_diameter_m", "outlet_diameter_m"}
    missing = required - set(df_silos.columns)
    if missing:
        raise ValueError(f"df_silos missing columns: {missing}")

    if "initial_mass_kg" not in df_silos.columns:
        df_silos = df_silos.copy()
        df_silos["initial_mass_kg"] = 0.0

    out: dict[str, float] = {}
    for _, row in df_silos.iterrows():
        silo = Silo(
            silo_id=str(row["silo_id"]),
            capacity_kg=float(row["capacity_kg"]),
            body_diameter_m=float(row["body_diameter_m"]),
            outlet_diameter_m=float(row["outlet_diameter_m"]),
            initial_mass_kg=float(row["initial_mass_kg"]),
        )
        out[silo.silo_id] = beverloo_mass_flow_rate_kg_s(silo, material, bev)
    return out


def available_mass_per_silo(df_layers: pd.DataFrame) -> dict[str, float]:
    """Total available segment mass by silo."""

    required = {"silo_id", "segment_mass_kg"}
    missing = required - set(df_layers.columns)
    if missing:
        raise ValueError(f"df_layers missing columns: {missing}")

    grouped = (
        df_layers.groupby("silo_id", as_index=False)["segment_mass_kg"].sum().assign(
            silo_id=lambda d: d["silo_id"].astype(str)
        )
    )
    return dict(zip(grouped["silo_id"], grouped["segment_mass_kg"].astype(float)))


def optimize_valve_times(
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
    initial_times_s: list[float] | tuple[float, ...] | np.ndarray | None = None,
    maxiter: int = 200,
    ftol: float = 1e-9,
    cache: bool = True,
) -> dict[str, Any]:
    """Optimize valve open times with SLSQP to match target blended parameters."""

    if not target_params:
        raise ValueError("target_params must not be empty")

    weights = weights or {}
    silo_ids = df_silos["silo_id"].astype(str).tolist()
    if len(set(silo_ids)) != len(silo_ids):
        raise ValueError("df_silos has duplicate silo_id values")

    mdot_map = compute_mdot_per_silo(df_silos, material, bev)
    avail_map = available_mass_per_silo(df_layers)

    mdot_vec = np.array([mdot_map[s] for s in silo_ids], dtype=float)
    avail_vec = np.array([float(avail_map.get(s, 0.0)) for s in silo_ids], dtype=float)

    if np.any(mdot_vec <= 0.0):
        bad = [silo_ids[i] for i, v in enumerate(mdot_vec) if v <= 0.0]
        raise ValueError(f"Non-positive m_dot for silos: {bad}")

    ub_times = avail_vec / mdot_vec
    bounds = Bounds(lb=np.zeros_like(ub_times), ub=ub_times)

    if fixed_total_mass_kg is not None:
        if fixed_total_mass_kg < 0.0:
            raise ValueError("fixed_total_mass_kg must be >= 0")
        if fixed_total_mass_kg > float(avail_vec.sum()) + 1e-9:
            raise ValueError("fixed_total_mass_kg exceeds total available mass")

    if initial_times_s is None:
        if fixed_total_mass_kg is None:
            x0 = np.minimum(0.30 * ub_times, ub_times)
        else:
            per = fixed_total_mass_kg / len(silo_ids)
            x0 = np.minimum(per / mdot_vec, ub_times)
    else:
        x0 = np.asarray(initial_times_s, dtype=float)
        if x0.shape != ub_times.shape:
            raise ValueError(f"initial_times_s must match shape {ub_times.shape}")
        x0 = np.clip(x0, 0.0, ub_times)

    cache_store: dict[tuple[float, ...], tuple[float, dict[str, Any]]] = {}
    heavy_penalty = 1e12

    def evaluate_masses(masses: np.ndarray) -> tuple[float, dict[str, Any]]:
        key = tuple(np.round(masses, 3).tolist())
        if cache and key in cache_store:
            return cache_store[key]

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
                err += heavy_penalty * w
            else:
                err += w * (float(v) - float(target)) ** 2

        out = (float(err), result)
        if cache:
            cache_store[key] = out
        return out

    def objective(times: np.ndarray) -> float:
        masses = np.clip(mdot_vec * np.asarray(times, dtype=float), 0.0, avail_vec)
        err, _ = evaluate_masses(masses)
        return err

    constraints: list[dict[str, Any]] = []
    mode = "A_fixed_total_mass" if fixed_total_mass_kg is not None else "B_free_total_mass"
    if fixed_total_mass_kg is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda t: float(np.dot(mdot_vec, np.asarray(t, dtype=float)) - fixed_total_mass_kg),
            }
        )

    opt = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol, "disp": False},
    )

    best_times = np.clip(np.asarray(opt.x, dtype=float), 0.0, ub_times)
    best_masses = np.clip(mdot_vec * best_times, 0.0, avail_vec)
    final_error, best_result = evaluate_masses(best_masses)

    return {
        "mode": mode,
        "success": bool(opt.success),
        "message": str(opt.message),
        "final_error": float(final_error),
        "best_times_s": dict(zip(silo_ids, best_times.tolist())),
        "best_masses_kg": dict(zip(silo_ids, best_masses.tolist())),
        "mdot_kg_s": dict(zip(silo_ids, mdot_vec.tolist())),
        "available_mass_kg": dict(zip(silo_ids, avail_vec.tolist())),
        "best_result": best_result,
        "optimizer_result": opt,
    }
