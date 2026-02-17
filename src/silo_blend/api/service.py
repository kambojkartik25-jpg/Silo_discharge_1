from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .. import bayes_opt as bayes_opt_module
from .. import optimize as optimize_module
from ..models import BeverlooParams, Material
from ..simulate import run_three_silo_blend
from .schemas import OptimizeRequest, SimulationRequest

DEFAULT_PARAM_NAMES = [
    "moisture_pct",
    "fine_extract_db_pct",
    "wort_pH",
    "diastatic_power_WK",
    "total_protein_pct",
    "wort_colour_EBC",
]


def available_optimizers() -> list[str]:
    out = ["slsqp"]
    if hasattr(optimize_module, "optimize_valve_times_least_squares"):
        out.append("least_squares")
    if hasattr(optimize_module, "optimize_valve_times_trust_constr"):
        out.append("trust_constr")
    if hasattr(bayes_opt_module, "optimize_valve_times_bayes"):
        out.append("bayes")
    return out


def infer_param_names(suppliers: list[dict[str, Any]] | None = None) -> list[str]:
    if not suppliers:
        return DEFAULT_PARAM_NAMES
    cols = list(pd.DataFrame(suppliers).columns)
    return [c for c in cols if c != "supplier"] or DEFAULT_PARAM_NAMES


def _to_df_silos(silos: list[Any]) -> pd.DataFrame:
    rows = [s.model_dump() for s in silos]
    return pd.DataFrame(rows)


def _to_df_layers(layers: list[Any]) -> pd.DataFrame:
    rows = [l.model_dump() for l in layers]
    return pd.DataFrame(rows)


def _to_df_suppliers(suppliers: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(suppliers)
    return df


def _to_df_discharge(discharge: list[Any]) -> pd.DataFrame:
    rows = [d.model_dump() for d in discharge]
    return pd.DataFrame(rows)


def validate_inputs(
    *,
    df_silos: pd.DataFrame,
    df_layers: pd.DataFrame,
    df_suppliers: pd.DataFrame,
    sigma_m: float,
    steps: int,
    mode: str | None = None,
    fixed_total_mass_kg: float | None = None,
) -> list[str]:
    warnings: list[str] = []

    if sigma_m <= 0:
        raise ValueError("sigma_m must be > 0")
    if steps <= 0 or steps > 10000:
        raise ValueError("steps must be in range [1, 10000]")

    if "silo_id" not in df_silos.columns:
        raise ValueError("silos must include silo_id")
    if df_silos["silo_id"].astype(str).duplicated().any():
        raise ValueError("silo_id values must be unique")

    req_layer = {"silo_id", "layer_index", "lot_id", "supplier", "segment_mass_kg"}
    missing_layer = req_layer - set(df_layers.columns)
    if missing_layer:
        raise ValueError(f"layers missing columns: {sorted(missing_layer)}")

    if "supplier" not in df_suppliers.columns:
        raise ValueError("suppliers must include supplier column")

    if (df_layers["segment_mass_kg"].astype(float) < 0).any():
        raise ValueError("segment_mass_kg cannot be negative")

    supplier_set = set(df_suppliers["supplier"].astype(str).tolist())
    used_supplier_set = set(df_layers["supplier"].astype(str).tolist())
    unknown = sorted(used_supplier_set - supplier_set)
    if unknown:
        raise ValueError(f"layer suppliers not found in suppliers table: {unknown}")

    for silo_id, group in df_layers.groupby(df_layers["silo_id"].astype(str)):
        idx = sorted(group["layer_index"].astype(int).tolist())
        expected = list(range(1, len(idx) + 1))
        if idx != expected:
            raise ValueError(f"layer_index for silo {silo_id} must be contiguous 1..N; got {idx}")

    if "capacity_kg" in df_silos.columns:
        for _, row in df_silos.iterrows():
            sid = str(row["silo_id"])
            cap = float(row["capacity_kg"])
            total = float(df_layers[df_layers["silo_id"].astype(str) == sid]["segment_mass_kg"].sum())
            if total > cap + 1e-9:
                warnings.append(
                    f"silo {sid}: segment mass sum ({total:.3f}) exceeds capacity ({cap:.3f})"
                )

    if mode == "A":
        if fixed_total_mass_kg is None:
            raise ValueError("fixed_total_mass_kg is required when mode='A'")
        total_available = float(df_layers["segment_mass_kg"].sum())
        if fixed_total_mass_kg > total_available + 1e-9:
            raise ValueError(
                f"fixed_total_mass_kg ({fixed_total_mass_kg:.3f}) exceeds total available ({total_available:.3f})"
            )

    return warnings


def _build_material_bev(request_obj: SimulationRequest | OptimizeRequest) -> tuple[Material, BeverlooParams]:
    material = Material(
        rho_bulk_kg_m3=request_obj.material.rho_bulk_kg_m3,
        grain_diameter_m=request_obj.material.grain_diameter_m,
    )
    bev = BeverlooParams(
        C=request_obj.beverloo.C,
        k=request_obj.beverloo.k,
        g_m_s2=request_obj.beverloo.g_m_s2,
    )
    return material, bev


def _safe_optimizer_summary(opt_result: Any) -> dict[str, Any] | None:
    if opt_result is None:
        return None
    out: dict[str, Any] = {}
    for key in ["success", "status", "message", "fun", "nit", "nfev"]:
        if hasattr(opt_result, key):
            value = getattr(opt_result, key)
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            out[key] = value
        elif isinstance(opt_result, dict) and key in opt_result:
            out[key] = opt_result[key]
    if hasattr(opt_result, "x"):
        x = getattr(opt_result, "x")
        if x is not None:
            out["x"] = np.asarray(x, dtype=float).tolist()
    if isinstance(opt_result, dict):
        for key in ["n_samples", "seed", "random_state"]:
            if key in opt_result:
                out[key] = opt_result[key]
    return out


def run_simulation_service(request: SimulationRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    df_silos = _to_df_silos(request.silos)
    df_layers = _to_df_layers(request.layers)
    df_suppliers = _to_df_suppliers(request.suppliers)
    df_discharge = _to_df_discharge(request.discharge)

    warnings = validate_inputs(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        sigma_m=request.sigma_m,
        steps=request.steps,
    )

    if (df_discharge["discharge_mass_kg"].astype(float) < 0).any():
        raise ValueError("discharge_mass_kg cannot be negative")

    material, bev = _build_material_bev(request)

    result = run_three_silo_blend(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        df_discharge=df_discharge,
        material=material,
        bev=bev,
        sigma_m=request.sigma_m,
        steps=request.steps,
        auto_adjust=request.auto_adjust,
    )

    per_silo: dict[str, Any] = {}
    for sid, sres in result["per_silo"].items():
        lot_df = sres["df_lot_contrib"]
        lot_items = [
            {
                "lot_id": str(r["lot_id"]),
                "supplier": str(r["supplier"]),
                "discharged_mass_kg": float(r["discharged_mass_kg"]),
            }
            for _, r in lot_df.iterrows()
        ]
        per_silo[sid] = {
            "discharged_mass_kg": float(sres["discharged_mass_kg"]),
            "mass_flow_rate_kg_s": float(sres["mass_flow_rate_kg_s"]),
            "discharge_time_s": float(sres["discharge_time_s"]),
            "sigma_m": float(sres["sigma_m"]),
            "lot_contrib": lot_items,
            "blended_params_per_silo": {
                k: float(v) for k, v in sres["blended_params_per_silo"].items()
            },
        }

    t1 = time.perf_counter()
    return {
        "inputs_echo": {
            "discharge": [d.model_dump() for d in request.discharge],
            "sigma_m": request.sigma_m,
            "steps": request.steps,
        },
        "per_silo": per_silo,
        "total": {
            "total_discharged_mass_kg": float(result["total_discharged_mass_kg"]),
            "total_blended_params": {
                k: float(v) for k, v in result["total_blended_params"].items()
            },
        },
        "diagnostics": {
            "warnings": warnings,
            "timing_ms": (t1 - t0) * 1000.0,
            "n_function_evals": None,
            "constraints_satisfied": None,
        },
    }


def run_optimization_service(request: OptimizeRequest) -> dict[str, Any]:
    t0 = time.perf_counter()
    df_silos = _to_df_silos(request.silos)
    df_layers = _to_df_layers(request.layers)
    df_suppliers = _to_df_suppliers(request.suppliers)

    warnings = validate_inputs(
        df_silos=df_silos,
        df_layers=df_layers,
        df_suppliers=df_suppliers,
        sigma_m=request.sigma_m,
        steps=request.steps,
        mode=request.mode,
        fixed_total_mass_kg=request.fixed_total_mass_kg,
    )

    material, bev = _build_material_bev(request)
    settings = request.optimizer_settings or {}

    supported = available_optimizers()
    if request.optimizer not in supported:
        raise RuntimeError(
            f"optimizer '{request.optimizer}' not available. supported={supported}"
        )

    if request.optimizer == "slsqp":
        out = optimize_module.optimize_valve_times(
            df_silos=df_silos,
            df_layers=df_layers,
            df_suppliers=df_suppliers,
            material=material,
            bev=bev,
            sigma_m=request.sigma_m,
            target_params=request.target_params,
            weights=request.weights,
            steps=request.steps,
            auto_adjust=request.auto_adjust,
            fixed_total_mass_kg=request.fixed_total_mass_kg if request.mode == "A" else None,
            initial_times_s=settings.get("initial_times_s"),
            maxiter=int(settings.get("maxiter", 200)),
            ftol=float(settings.get("ftol", 1e-9)),
            cache=bool(settings.get("cache", True)),
        )
        optimizer_name = "slsqp"
    elif request.optimizer == "least_squares":
        func = getattr(optimize_module, "optimize_valve_times_least_squares")
        out = func(
            df_silos=df_silos,
            df_layers=df_layers,
            df_suppliers=df_suppliers,
            material=material,
            bev=bev,
            sigma_m=request.sigma_m,
            target_params=request.target_params,
            weights=request.weights,
            steps=request.steps,
            auto_adjust=request.auto_adjust,
            fixed_total_mass_kg=request.fixed_total_mass_kg if request.mode == "A" else None,
            **settings,
        )
        optimizer_name = "least_squares"
    elif request.optimizer == "trust_constr":
        func = getattr(optimize_module, "optimize_valve_times_trust_constr")
        out = func(
            df_silos=df_silos,
            df_layers=df_layers,
            df_suppliers=df_suppliers,
            material=material,
            bev=bev,
            sigma_m=request.sigma_m,
            target_params=request.target_params,
            weights=request.weights,
            steps=request.steps,
            auto_adjust=request.auto_adjust,
            fixed_total_mass_kg=request.fixed_total_mass_kg if request.mode == "A" else None,
            **settings,
        )
        optimizer_name = "trust_constr"
    elif request.optimizer == "bayes":
        func = getattr(bayes_opt_module, "optimize_valve_times_bayes")
        out = func(
            df_silos=df_silos,
            df_layers=df_layers,
            df_suppliers=df_suppliers,
            material=material,
            bev=bev,
            sigma_m=request.sigma_m,
            target_params=request.target_params,
            weights=request.weights,
            steps=request.steps,
            auto_adjust=request.auto_adjust,
            mode=request.mode,
            fixed_total_mass_kg=request.fixed_total_mass_kg,
            n_calls=int(settings.get("n_calls", 60)),
            n_initial_points=int(settings.get("n_initial_points", 15)),
            random_state=int(settings.get("random_state", 42)),
            penalty_scale=float(settings.get("penalty_scale", 1e3)),
            cache=bool(settings.get("cache", True)),
        )
        optimizer_name = str(out.get("optimizer_name", "bayes"))
    else:
        raise RuntimeError("Unsupported optimizer")

    pred = out["best_result"]["total_blended_params"]
    deltas = {
        k: float(pred.get(k, np.nan) - v) if k in pred else float("nan")
        for k, v in request.target_params.items()
    }

    per_silo_blended = {
        sid: {k: float(v) for k, v in sres["blended_params_per_silo"].items()}
        for sid, sres in out["best_result"]["per_silo"].items()
    }

    total_mass = float(sum(out["best_masses_kg"].values()))
    constraints_satisfied = True
    if request.mode == "A" and request.fixed_total_mass_kg is not None:
        constraints_satisfied = abs(total_mass - request.fixed_total_mass_kg) <= 1e-3

    n_function_evals = None
    opt_res = out.get("optimizer_result")
    if hasattr(opt_res, "nfev"):
        n_function_evals = int(getattr(opt_res, "nfev"))
    elif hasattr(opt_res, "func_vals"):
        n_function_evals = int(len(getattr(opt_res, "func_vals")))
    elif isinstance(opt_res, dict) and "n_samples" in opt_res:
        n_function_evals = int(opt_res["n_samples"])

    t1 = time.perf_counter()
    response: dict[str, Any] = {
        "optimizer_name": optimizer_name,
        "mode": out["mode"],
        "success": bool(out["success"]),
        "message": str(out["message"]),
        "final_error": float(out["final_error"]),
        "best_times_s": {k: float(v) for k, v in out["best_times_s"].items()},
        "best_masses_kg": {k: float(v) for k, v in out["best_masses_kg"].items()},
        "predicted_total_blended_params": {k: float(v) for k, v in pred.items()},
        "deltas_vs_target": deltas,
        "per_silo_blended_params": per_silo_blended,
        "diagnostics": {
            "warnings": warnings,
            "timing_ms": (t1 - t0) * 1000.0,
            "n_function_evals": n_function_evals,
            "constraints_satisfied": constraints_satisfied,
        },
    }

    if request.return_debug:
        response["debug"] = {
            "optimizer_result": _safe_optimizer_summary(out.get("optimizer_result")),
            "available_mass_kg": {k: float(v) for k, v in out.get("available_mass_kg", {}).items()},
            "mdot_kg_s": {k: float(v) for k, v in out.get("mdot_kg_s", {}).items()},
        }

    return response
