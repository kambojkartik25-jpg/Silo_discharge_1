from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from silo_blend.api.app import app


def _load_payload_parts() -> tuple[list[dict], list[dict], list[dict]]:
    root = Path(__file__).resolve().parents[1]
    silos = pd.read_csv(root / "data" / "sample_silos.csv").to_dict(orient="records")
    layers = pd.read_csv(root / "data" / "sample_layers.csv").to_dict(orient="records")
    suppliers = pd.read_csv(root / "data" / "sample_suppliers.csv").to_dict(orient="records")
    return silos, layers, suppliers


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_simulate_endpoint() -> None:
    silos, layers, suppliers = _load_payload_parts()
    client = TestClient(app)

    payload = {
        "silos": silos,
        "layers": layers,
        "suppliers": suppliers,
        "discharge": [
            {"silo_id": "S1", "discharge_mass_kg": 1000.0},
            {"silo_id": "S2", "discharge_mass_kg": 900.0},
            {"silo_id": "S3", "discharge_mass_kg": 600.0},
        ],
        "material": {"rho_bulk_kg_m3": 610.0, "grain_diameter_m": 0.004},
        "beverloo": {"C": 0.58, "k": 1.4, "g_m_s2": 9.81},
        "sigma_m": 0.12,
        "steps": 120,
        "auto_adjust": False,
    }

    resp = client.post("/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "total_blended_params" in data["total"]
    assert abs(data["total"]["total_discharged_mass_kg"] - 2500.0) < 1e-6


def test_optimize_endpoint_mode_a() -> None:
    silos, layers, suppliers = _load_payload_parts()
    client = TestClient(app)

    payload = {
        "silos": silos,
        "layers": layers,
        "suppliers": suppliers,
        "material": {"rho_bulk_kg_m3": 610.0, "grain_diameter_m": 0.004},
        "beverloo": {"C": 0.58, "k": 1.4, "g_m_s2": 9.81},
        "sigma_m": 0.12,
        "steps": 120,
        "auto_adjust": False,
        "target_params": {"moisture_pct": 4.28, "wort_pH": 5.96},
        "weights": {"moisture_pct": 1.0, "wort_pH": 5.0},
        "mode": "A",
        "fixed_total_mass_kg": 1200.0,
        "optimizer": "slsqp",
        "optimizer_settings": {"maxiter": 80, "ftol": 1e-8},
        "return_debug": False,
    }

    resp = client.post("/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"]
    assert abs(sum(data["best_masses_kg"].values()) - 1200.0) <= 1e-3
