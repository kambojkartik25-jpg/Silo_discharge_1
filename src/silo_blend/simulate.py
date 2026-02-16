from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .models import BeverlooParams, Material, Silo, normal_cdf


def beverloo_mass_flow_rate_kg_s(silo: Silo, material: Material, bev: BeverlooParams) -> float:
    """Compute mass flow rate using Beverloo equation."""

    deff = silo.outlet_diameter_m - bev.k * material.grain_diameter_m
    if deff <= 0.0:
        raise ValueError(
            f"Invalid Beverloo geometry for {silo.silo_id}: D-k*d={deff:.6f} <= 0"
        )
    return bev.C * material.rho_bulk_kg_m3 * sqrt(bev.g_m_s2) * (deff ** 2.5)


def build_intervals_from_layers(
    df_layers: pd.DataFrame, silo: Silo, material: Material
) -> tuple[pd.DataFrame, float]:
    """Build bottom-to-top segment intervals [z0, z1] in meters for one silo."""

    required = {"silo_id", "layer_index", "lot_id", "supplier", "segment_mass_kg"}
    missing = required - set(df_layers.columns)
    if missing:
        raise ValueError(f"df_layers missing columns: {missing}")

    d = df_layers[df_layers["silo_id"].astype(str) == silo.silo_id].copy()
    if d.empty:
        raise ValueError(f"No layers found for silo_id={silo.silo_id}")

    d["layer_index"] = d["layer_index"].astype(int)
    d = d.sort_values("layer_index", kind="mergesort").reset_index(drop=True)
    expected = list(range(1, len(d) + 1))
    if d["layer_index"].tolist() != expected:
        raise ValueError(
            f"layer_index for {silo.silo_id} must be contiguous 1..N, got {d['layer_index'].tolist()}"
        )

    d["segment_mass_kg"] = d["segment_mass_kg"].astype(float)
    if (d["segment_mass_kg"] < 0.0).any():
        raise ValueError(f"Negative segment_mass_kg found in silo {silo.silo_id}")

    total_mass = float(d["segment_mass_kg"].sum())
    if total_mass > silo.capacity_kg + 1e-9:
        warnings.warn(
            f"Silo {silo.silo_id}: total segment mass ({total_mass:.2f}) exceeds capacity ({silo.capacity_kg:.2f})",
            stacklevel=2,
        )

    heights = d["segment_mass_kg"].to_numpy() / (material.rho_bulk_kg_m3 * silo.cross_section_area_m2)
    z1 = np.cumsum(heights)
    z0 = np.concatenate(([0.0], z1[:-1]))

    d["z0_m"] = z0
    d["z1_m"] = z1
    return d, float(z1[-1]) if len(z1) else 0.0


def layer_probabilities(
    z_front: float, sigma_m: float, intervals_df: pd.DataFrame, total_height_m: float
) -> pd.Series:
    """Probability of sampling each layer at a given front height using truncated Gaussian."""

    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be > 0")

    z0 = intervals_df["z0_m"].to_numpy(dtype=float)
    z1 = intervals_df["z1_m"].to_numpy(dtype=float)

    denom = normal_cdf((total_height_m - z_front) / sigma_m) - normal_cdf((0.0 - z_front) / sigma_m)
    if denom <= 1e-15:
        return pd.Series(np.zeros_like(z0), index=intervals_df.index)

    cdf_hi = np.vectorize(normal_cdf)((z1 - z_front) / sigma_m)
    cdf_lo = np.vectorize(normal_cdf)((z0 - z_front) / sigma_m)
    probs = np.clip((cdf_hi - cdf_lo) / denom, 0.0, None)
    s = probs.sum()
    if s > 0.0:
        probs = probs / s
    return pd.Series(probs, index=intervals_df.index)


def _resolve_discharge_mass(df_discharge_row: pd.Series, total_mass_kg: float) -> float:
    has_mass = "discharge_mass_kg" in df_discharge_row.index and pd.notna(df_discharge_row["discharge_mass_kg"])
    has_frac = "discharge_fraction" in df_discharge_row.index and pd.notna(df_discharge_row["discharge_fraction"])

    if has_mass:
        m = float(df_discharge_row["discharge_mass_kg"])
    elif has_frac:
        m = float(df_discharge_row["discharge_fraction"]) * total_mass_kg
    else:
        raise ValueError("df_discharge row must contain discharge_mass_kg or discharge_fraction")

    if m < 0.0:
        raise ValueError("discharge mass cannot be negative")
    if m > total_mass_kg + 1e-9:
        raise ValueError(
            f"Requested discharge ({m:.3f}) exceeds total mass in silo ({total_mass_kg:.3f})"
        )
    return m


def estimate_discharge_contrib_for_silo(
    silo: Silo,
    df_layers: pd.DataFrame,
    df_discharge_row: pd.Series,
    material: Material,
    bev: BeverlooParams,
    sigma_m: float,
    steps: int,
) -> pd.DataFrame:
    """Integrate discharge over time and return per-segment discharged masses."""

    if steps <= 0:
        raise ValueError("steps must be > 0")

    intervals_df, total_height_m = build_intervals_from_layers(df_layers, silo, material)
    total_mass_kg = float(intervals_df["segment_mass_kg"].sum())
    discharge_mass_kg = _resolve_discharge_mass(df_discharge_row, total_mass_kg)

    m_dot = beverloo_mass_flow_rate_kg_s(silo, material, bev)
    t_end = discharge_mass_kg / m_dot if m_dot > 0 else 0.0
    dt = t_end / steps if steps > 0 else 0.0
    dm = m_dot * dt

    seg = intervals_df.copy()
    seg["discharged_mass_kg"] = 0.0

    if discharge_mass_kg == 0.0:
        return seg

    area = silo.cross_section_area_m2
    for i in range(steps):
        t_mid = (i + 0.5) * dt
        m_removed = min(discharge_mass_kg, m_dot * t_mid)
        z_front = m_removed / (material.rho_bulk_kg_m3 * area)
        p = layer_probabilities(z_front, sigma_m, seg, total_height_m).to_numpy()
        seg["discharged_mass_kg"] += dm * p

    s = float(seg["discharged_mass_kg"].sum())
    if s > 0.0:
        seg["discharged_mass_kg"] *= discharge_mass_kg / s

    return seg


def _blend_params_from_contrib(df_contrib: pd.DataFrame, df_suppliers: pd.DataFrame) -> dict[str, float]:
    required = {"supplier", "discharged_mass_kg"}
    missing = required - set(df_contrib.columns)
    if missing:
        raise ValueError(f"contrib missing columns: {missing}")

    if "supplier" not in df_suppliers.columns:
        raise ValueError("df_suppliers must include supplier column")

    param_cols = [c for c in df_suppliers.columns if c != "supplier"]
    if not param_cols:
        raise ValueError("df_suppliers must contain parameter columns")

    merged = df_contrib.merge(df_suppliers, on="supplier", how="left")
    if merged[param_cols].isna().any().any():
        raise ValueError("Missing supplier specs for one or more contributions")

    total = float(merged["discharged_mass_kg"].sum())
    if total <= 0.0:
        return {p: float("nan") for p in param_cols}

    w = merged["discharged_mass_kg"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    for p in param_cols:
        out[p] = float(np.dot(w, merged[p].to_numpy(dtype=float)) / total)
    return out


def _build_silo_map(df_silos: pd.DataFrame) -> dict[str, Silo]:
    required = {"silo_id", "capacity_kg", "body_diameter_m", "outlet_diameter_m"}
    missing = required - set(df_silos.columns)
    if missing:
        raise ValueError(f"df_silos missing columns: {missing}")

    if "initial_mass_kg" not in df_silos.columns:
        df_silos = df_silos.copy()
        df_silos["initial_mass_kg"] = 0.0

    silos: dict[str, Silo] = {}
    for _, row in df_silos.iterrows():
        sid = str(row["silo_id"])
        silos[sid] = Silo(
            silo_id=sid,
            capacity_kg=float(row["capacity_kg"]),
            body_diameter_m=float(row["body_diameter_m"]),
            outlet_diameter_m=float(row["outlet_diameter_m"]),
            initial_mass_kg=float(row["initial_mass_kg"]),
        )
    return silos


def _simulate_one_silo(
    silo: Silo,
    df_layers: pd.DataFrame,
    df_discharge_row: pd.Series,
    df_suppliers: pd.DataFrame,
    material: Material,
    bev: BeverlooParams,
    sigma_m: float,
    steps: int,
    auto_adjust: bool,
) -> dict[str, Any]:
    sigma = sigma_m
    seg = estimate_discharge_contrib_for_silo(silo, df_layers, df_discharge_row, material, bev, sigma, steps)

    if auto_adjust:
        for _ in range(8):
            lot_mass = seg.groupby("lot_id", as_index=False)["discharged_mass_kg"].sum()
            if (lot_mass["discharged_mass_kg"] > 1e-3).sum() >= 2:
                break
            sigma *= 1.35
            seg = estimate_discharge_contrib_for_silo(
                silo, df_layers, df_discharge_row, material, bev, sigma, steps
            )

    lot = (
        seg.groupby(["silo_id", "lot_id", "supplier"], as_index=False)["discharged_mass_kg"]
        .sum()
        .sort_values(["silo_id", "lot_id"])
        .reset_index(drop=True)
    )

    blended = _blend_params_from_contrib(lot, df_suppliers)
    discharged_mass_kg = float(seg["discharged_mass_kg"].sum())
    m_dot = beverloo_mass_flow_rate_kg_s(silo, material, bev)

    return {
        "silo_id": silo.silo_id,
        "discharged_mass_kg": discharged_mass_kg,
        "mass_flow_rate_kg_s": m_dot,
        "discharge_time_s": discharged_mass_kg / m_dot if m_dot > 0 else 0.0,
        "sigma_m": sigma,
        "df_segment_contrib": seg[
            ["silo_id", "layer_index", "lot_id", "supplier", "segment_mass_kg", "discharged_mass_kg"]
        ].copy(),
        "df_lot_contrib": lot,
        "blended_params_per_silo": blended,
    }


def run_three_silo_blend(
    df_silos: pd.DataFrame,
    df_layers: pd.DataFrame,
    df_suppliers: pd.DataFrame,
    df_discharge: pd.DataFrame,
    material: Material,
    bev: BeverlooParams,
    sigma_m: float,
    steps: int = 2000,
    auto_adjust: bool = False,
) -> dict[str, Any]:
    """Run discharge simulation for all silos and compute total blended parameters."""

    if "silo_id" not in df_discharge.columns:
        raise ValueError("df_discharge must include silo_id")

    silos = _build_silo_map(df_silos)
    if len(silos) != 3:
        warnings.warn(f"Expected 3 silos; got {len(silos)}", stacklevel=2)

    per_silo: dict[str, Any] = {}
    segment_parts: list[pd.DataFrame] = []
    lot_parts: list[pd.DataFrame] = []

    for sid, silo in silos.items():
        row = df_discharge[df_discharge["silo_id"].astype(str) == sid]
        if row.empty:
            raise ValueError(f"No discharge row for silo_id={sid}")
        sim = _simulate_one_silo(
            silo=silo,
            df_layers=df_layers,
            df_discharge_row=row.iloc[0],
            df_suppliers=df_suppliers,
            material=material,
            bev=bev,
            sigma_m=sigma_m,
            steps=steps,
            auto_adjust=auto_adjust,
        )
        per_silo[sid] = sim
        segment_parts.append(sim["df_segment_contrib"])
        lot_parts.append(sim["df_lot_contrib"])

    df_segment_all = pd.concat(segment_parts, ignore_index=True)
    df_lot_all = pd.concat(lot_parts, ignore_index=True)
    total_blended = _blend_params_from_contrib(df_lot_all, df_suppliers)

    return {
        "per_silo": per_silo,
        "df_segment_contrib_all": df_segment_all,
        "df_lot_contrib_all": df_lot_all,
        "total_discharged_mass_kg": float(df_segment_all["discharged_mass_kg"].sum()),
        "total_blended_params": total_blended,
        "material": asdict(material),
        "beverloo": asdict(bev),
    }
