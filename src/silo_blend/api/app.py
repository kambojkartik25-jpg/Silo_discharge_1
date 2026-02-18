from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    OptimizeRequest,
    OptimizeResponse,
    OptimizerSchemaResponse,
    ParamsSchemaResponse,
    SimulationRequest,
    SimulationResponse,
)
from .service import available_optimizers, infer_param_names, run_optimization_service, run_simulation_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("silo_blend.api")

app = FastAPI(
    title="silo_blend API",
    version="0.1.0",
    description="HTTP API for 3-silo simulation and optimization",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest) -> dict:
    try:
        return run_simulation_service(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest) -> dict:
    logger.info(
        "POST /optimize optimizer=%s mode=%s silos=%d steps=%d",
        payload.optimizer,
        payload.mode,
        len(payload.silos),
        payload.steps,
    )
    try:
        response = run_optimization_service(payload)
        logger.info(
            "POST /optimize completed success=%s final_error=%.6f",
            response.get("success"),
            float(response.get("final_error", float("nan"))),
        )
        return response
    except ValueError as exc:
        logger.warning("POST /optimize validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("POST /optimize runtime error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/schema/optimizers", response_model=OptimizerSchemaResponse)
def schema_optimizers() -> dict[str, list[str]]:
    return {"supported_optimizers": available_optimizers()}


@app.get("/schema/params", response_model=ParamsSchemaResponse)
def schema_params() -> dict[str, list[str]]:
    return {"params": infer_param_names(None)}
