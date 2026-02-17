from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from .models import BeverlooParams, Material
from .optimize import available_mass_per_silo, compute_mdot_per_silo
from .simulate import run_three_silo_blend


def objective_evaluator(
    times_s: np.ndarray,
    *,
    silo_ids: list[str],
    mdot_vec: np.ndarray,
    available_vec: np.ndarray,
    df_silos: pd.DataFrame,
    df_layers: pd.DataFrame,
    df_suppliers: pd.DataFrame,
    material: Material,
    bev: BeverlooParams,
    sigma_m: float,
    target_params: dict[str, float],
    weights: dict[str, float],
    steps: int,
    auto_adjust: bool,
    penalty_scale: float,
    cache_store: dict[tuple[float, ...], tuple[float, dict[str, Any]]] | None = None,
) -> tuple[float, dict[str, Any], np.ndarray]:
    """Evaluate objective for a candidate time vector and return loss/result/masses."""

    times = np.asarray(times_s, dtype=float)
    masses_raw = mdot_vec * times
    below = np.clip(-masses_raw, 0.0, None)
    above = np.clip(masses_raw - available_vec, 0.0, None)
    violation = below + above
    penalty = penalty_scale * float(np.sum((violation / (available_vec + 1e-9)) ** 2))

    masses = np.clip(masses_raw, 0.0, available_vec)
    key = tuple(np.round(masses, 3).tolist())
    if cache_store is not None and key in cache_store:
        recipe_loss, result = cache_store[key]
        return recipe_loss + penalty, result, masses

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
    recipe_loss = 0.0
    heavy_penalty = 1e12
    for p, target in target_params.items():
        w = float(weights.get(p, 1.0))
        v = pred.get(p, np.nan)
        if v is None or not np.isfinite(v):
            recipe_loss += heavy_penalty * w
        else:
            recipe_loss += w * (float(v) - float(target)) ** 2

    if cache_store is not None:
        cache_store[key] = (float(recipe_loss), result)

    return float(recipe_loss + penalty), result, masses


def optimize_valve_times_bayes(
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
    mode: Literal["A", "B"] = "A",
    fixed_total_mass_kg: float | None = None,
    n_calls: int = 60,
    n_initial_points: int = 15,
    random_state: int = 42,
    penalty_scale: float = 1e3,
    cache: bool = True,
) -> dict[str, Any]:
    """Bayesian optimization of valve times with gp_minimize."""

    if not target_params:
        raise ValueError("target_params must not be empty")
    if n_calls <= 0:
        raise ValueError("n_calls must be > 0")

    weights = weights or {}
    silo_ids = df_silos["silo_id"].astype(str).tolist()
    if len(set(silo_ids)) != len(silo_ids):
        raise ValueError("df_silos has duplicate silo_id values")

    mdot_map = compute_mdot_per_silo(df_silos, material, bev)
    avail_map = available_mass_per_silo(df_layers)
    mdot_vec = np.array([mdot_map[s] for s in silo_ids], dtype=float)
    available_vec = np.array([float(avail_map.get(s, 0.0)) for s in silo_ids], dtype=float)

    if np.any(mdot_vec <= 0.0):
        bad = [silo_ids[i] for i, v in enumerate(mdot_vec) if v <= 0.0]
        raise ValueError(f"Non-positive m_dot for silos: {bad}")

    ub_times = available_vec / mdot_vec
    cache_store = {} if cache else None

    best_error = float("inf")
    best_times = np.zeros_like(mdot_vec)
    best_masses = np.zeros_like(mdot_vec)
    best_result: dict[str, Any] | None = None

    if mode == "A":
        if fixed_total_mass_kg is None:
            raise ValueError("fixed_total_mass_kg is required for mode='A'")
        if fixed_total_mass_kg < 0.0:
            raise ValueError("fixed_total_mass_kg must be >= 0")

        space = [Real(0.0, 1.0, name="u1"), Real(0.0, 1.0, name="u2")]

        def f(x: list[float]) -> float:
            nonlocal best_error, best_times, best_masses, best_result
            u1, u2 = float(x[0]), float(x[1])
            w1 = u1
            w2 = (1.0 - u1) * u2
            w3 = 1.0 - w1 - w2
            w = np.array([w1, w2, w3], dtype=float)
            masses = fixed_total_mass_kg * w
            times = masses / mdot_vec

            loss, result, masses_used = objective_evaluator(
                times,
                silo_ids=silo_ids,
                mdot_vec=mdot_vec,
                available_vec=available_vec,
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
                penalty_scale=penalty_scale,
                cache_store=cache_store,
            )
            if loss < best_error:
                best_error = loss
                best_times = np.clip(times, 0.0, ub_times)
                best_masses = masses_used
                best_result = result
            return float(loss)

        opt = gp_minimize(
            func=f,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=min(n_initial_points, n_calls),
            random_state=random_state,
            acq_func="EI",
        )
        mode_name = "A_fixed_total_mass"

    elif mode == "B":
        space = [Real(0.0, float(ub), name=f"t_{sid}") for sid, ub in zip(silo_ids, ub_times)]

        def f(x: list[float]) -> float:
            nonlocal best_error, best_times, best_masses, best_result
            times = np.array(x, dtype=float)
            loss, result, masses_used = objective_evaluator(
                times,
                silo_ids=silo_ids,
                mdot_vec=mdot_vec,
                available_vec=available_vec,
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
                penalty_scale=penalty_scale,
                cache_store=cache_store,
            )
            if loss < best_error:
                best_error = loss
                best_times = np.clip(times, 0.0, ub_times)
                best_masses = masses_used
                best_result = result
            return float(loss)

        opt = gp_minimize(
            func=f,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=min(n_initial_points, n_calls),
            random_state=random_state,
            acq_func="EI",
        )
        mode_name = "B_free_total_mass"
    else:
        raise ValueError("mode must be 'A' or 'B'")

    if best_result is None:
        masses = np.clip(mdot_vec * best_times, 0.0, available_vec)
        df_discharge = pd.DataFrame({"silo_id": silo_ids, "discharge_mass_kg": masses})
        best_result = run_three_silo_blend(
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

    return {
        "optimizer_name": "bayes_opt_gp_minimize",
        "mode": mode_name,
        "success": True,
        "message": "Bayesian optimization completed",
        "final_error": float(best_error),
        "best_times_s": dict(zip(silo_ids, best_times.tolist())),
        "best_masses_kg": dict(zip(silo_ids, best_masses.tolist())),
        "mdot_kg_s": dict(zip(silo_ids, mdot_vec.tolist())),
        "available_mass_kg": dict(zip(silo_ids, available_vec.tolist())),
        "best_result": best_result,
        "optimizer_result": opt,
    }
