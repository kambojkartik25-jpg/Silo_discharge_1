from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SupplierSpec(BaseModel):
    supplier: str = Field(..., examples=["BBM"])
    params: dict[str, float] = Field(default_factory=dict, examples=[{"moisture_pct": 4.2}])


class SiloSpec(BaseModel):
    silo_id: str = Field(..., examples=["S1"])
    capacity_kg: float = Field(..., ge=0)
    body_diameter_m: float = Field(..., gt=0)
    outlet_diameter_m: float = Field(..., gt=0)
    initial_mass_kg: float = Field(0.0, ge=0)


class LayerSegment(BaseModel):
    silo_id: str
    layer_index: int = Field(..., ge=1)
    lot_id: str
    supplier: str
    segment_mass_kg: float = Field(..., ge=0)


class MaterialSpec(BaseModel):
    rho_bulk_kg_m3: float = Field(..., gt=0)
    grain_diameter_m: float = Field(..., gt=0)


class BeverlooSpec(BaseModel):
    C: float = 0.58
    k: float = 1.4
    g_m_s2: float = 9.81


class DischargeByMass(BaseModel):
    silo_id: str
    discharge_mass_kg: float = Field(..., ge=0)


class DischargeByTime(BaseModel):
    silo_id: str
    valve_time_s: float = Field(..., ge=0)


class SimulationRequest(BaseModel):
    silos: list[SiloSpec]
    layers: list[LayerSegment]
    suppliers: list[dict[str, Any]]
    discharge: list[DischargeByMass]
    material: MaterialSpec
    beverloo: BeverlooSpec
    sigma_m: float = Field(..., gt=0)
    steps: int = Field(1200, ge=1, le=10000)
    auto_adjust: bool = False


class OptimizeRequest(BaseModel):
    silos: list[SiloSpec]
    layers: list[LayerSegment]
    suppliers: list[dict[str, Any]]
    material: MaterialSpec
    beverloo: BeverlooSpec
    sigma_m: float = Field(..., gt=0)
    steps: int = Field(1200, ge=1, le=10000)
    auto_adjust: bool = False
    target_params: dict[str, float]
    weights: dict[str, float] | None = None
    mode: Literal["A", "B"] = "A"
    fixed_total_mass_kg: float | None = Field(default=None, ge=0)
    optimizer: Literal["auto", "slsqp", "montecarlo"] = "auto"
    mc_samples: int = Field(500, ge=1)
    mc_seed: int = 123
    optimizer_settings: dict[str, Any] | None = None
    return_debug: bool = False


class LotContribItem(BaseModel):
    lot_id: str
    supplier: str
    discharged_mass_kg: float


class PerSiloSimulationResult(BaseModel):
    discharged_mass_kg: float
    mass_flow_rate_kg_s: float
    discharge_time_s: float
    sigma_m: float
    lot_contrib: list[LotContribItem]
    blended_params_per_silo: dict[str, float]


class SimulationInputsEcho(BaseModel):
    discharge: list[DischargeByMass]
    sigma_m: float
    steps: int


class Diagnostics(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    timing_ms: float
    n_function_evals: int | None = None
    constraints_satisfied: bool | None = None


class SimulationResponse(BaseModel):
    inputs_echo: SimulationInputsEcho
    per_silo: dict[str, PerSiloSimulationResult]
    total: dict[str, Any]
    diagnostics: Diagnostics


class OptimizeResponse(BaseModel):
    optimizer_name: str
    mode: str
    success: bool
    message: str
    final_error: float
    best_optimizer: str | None = None
    compared: dict[str, float | None] | None = None
    runner_up_error: float | None = None
    best_times_s: dict[str, float]
    best_masses_kg: dict[str, float]
    best_result: dict[str, Any] | None = None
    predicted_total_blended_params: dict[str, float]
    deltas_vs_target: dict[str, float]
    per_silo_blended_params: dict[str, dict[str, float]] | None = None
    diagnostics: Diagnostics
    debug: dict[str, Any] | None = None


class OptimizerSchemaResponse(BaseModel):
    supported_optimizers: list[str]


class ParamsSchemaResponse(BaseModel):
    params: list[str]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    detail: str
