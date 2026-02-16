from silo_blend.models import BeverlooParams, Material, Silo, normal_cdf
from silo_blend.simulate import beverloo_mass_flow_rate_kg_s


def test_normal_cdf_midpoint() -> None:
    assert abs(normal_cdf(0.0) - 0.5) < 1e-12


def test_beverloo_positive_flow() -> None:
    material = Material(rho_bulk_kg_m3=610.0, grain_diameter_m=0.004)
    bev = BeverlooParams(C=0.58, k=1.4)
    silo = Silo(silo_id="S1", capacity_kg=4000.0, body_diameter_m=3.0, outlet_diameter_m=0.2)
    assert beverloo_mass_flow_rate_kg_s(silo, material, bev) > 0.0
