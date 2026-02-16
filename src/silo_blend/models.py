from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt


@dataclass(frozen=True)
class Material:
    """Bulk material properties used in flow and layer geometry."""

    rho_bulk_kg_m3: float
    grain_diameter_m: float


@dataclass(frozen=True)
class BeverlooParams:
    """Beverloo discharge parameters."""

    C: float = 0.58
    k: float = 1.4
    g_m_s2: float = 9.81


@dataclass(frozen=True)
class Silo:
    """Cylindrical silo geometry and metadata."""

    silo_id: str
    capacity_kg: float
    body_diameter_m: float
    outlet_diameter_m: float
    initial_mass_kg: float = 0.0

    @property
    def cross_section_area_m2(self) -> float:
        return pi * (self.body_diameter_m / 2.0) ** 2


@dataclass(frozen=True)
class Lot:
    """Optional lot descriptor."""

    lot_id: str
    supplier: str
    mass_kg: float


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
