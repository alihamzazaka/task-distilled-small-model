"""Honest cost model (SPEC §7): teacher API $/1k requests vs the student's
*amortized* local cost (GPU capex + electricity — never "it's free locally"),
plus the break-even request volume. Pure Python, unit-tested locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

HOURS_PER_DAY = 24.0
DAYS_PER_YEAR = 365.0


# ---------------------------------------------------------------------------
# Teacher (API) side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeacherPricing:
    input_per_mtok: float   # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens


def teacher_cost_per_request(
    pricing: TeacherPricing,
    avg_input_tokens: float,
    avg_output_tokens: float,
) -> float:
    """USD per single API request given an average token profile."""
    if avg_input_tokens < 0 or avg_output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return (
        avg_input_tokens / 1e6 * pricing.input_per_mtok
        + avg_output_tokens / 1e6 * pricing.output_per_mtok
    )


def teacher_cost_per_1k(
    pricing: TeacherPricing,
    avg_input_tokens: float,
    avg_output_tokens: float,
) -> float:
    return 1000.0 * teacher_cost_per_request(pricing, avg_input_tokens, avg_output_tokens)


# ---------------------------------------------------------------------------
# Student (local GPU) side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudentCostParams:
    gpu_price_usd: float = 1100.0        # RTX 5080 capex
    gpu_lifetime_years: float = 3.0      # straight-line amortization horizon
    gpu_utilization: float = 1.0         # fraction of amortization billed to this workload
    gpu_power_watts: float = 360.0       # draw under inference load
    electricity_usd_per_kwh: float = 0.15
    throughput_rps: float = 4.0          # sustained requests/second

    def validate(self) -> None:
        if self.gpu_lifetime_years <= 0:
            raise ValueError("gpu_lifetime_years must be > 0")
        if not (0 < self.gpu_utilization <= 1):
            raise ValueError("gpu_utilization must be in (0, 1]")
        if self.throughput_rps <= 0:
            raise ValueError("throughput_rps must be > 0")
        if self.gpu_price_usd < 0 or self.gpu_power_watts < 0 or self.electricity_usd_per_kwh < 0:
            raise ValueError("prices/power must be non-negative")


def gpu_amortized_usd_per_hour(p: StudentCostParams) -> float:
    """Straight-line capex per hour, scaled by the utilization share."""
    p.validate()
    total_hours = p.gpu_lifetime_years * DAYS_PER_YEAR * HOURS_PER_DAY
    return (p.gpu_price_usd / total_hours) * p.gpu_utilization


def electricity_usd_per_hour(p: StudentCostParams) -> float:
    p.validate()
    return (p.gpu_power_watts / 1000.0) * p.electricity_usd_per_kwh


def student_cost_per_request(p: StudentCostParams) -> float:
    """USD per request assuming the GPU is busy while serving:
    (amortization + electricity) per hour / requests per hour."""
    p.validate()
    per_hour = gpu_amortized_usd_per_hour(p) + electricity_usd_per_hour(p)
    requests_per_hour = p.throughput_rps * 3600.0
    return per_hour / requests_per_hour


def student_cost_per_1k(p: StudentCostParams) -> float:
    return 1000.0 * student_cost_per_request(p)


# ---------------------------------------------------------------------------
# Break-even
# ---------------------------------------------------------------------------

def break_even_requests_per_day(
    pricing: TeacherPricing,
    avg_input_tokens: float,
    avg_output_tokens: float,
    p: StudentCostParams,
) -> Optional[float]:
    """Requests/day above which the local student is cheaper than the API.

    Model: the GPU's amortization is a FIXED daily cost (you own it whether
    or not it serves), electricity is variable with load. The teacher is
    purely variable. Solve for volume V:

        teacher_per_req * V  >=  gpu_amortized_per_day + energy_per_req * V
        V  >=  gpu_daily / (teacher_per_req - energy_per_req)

    Returns None when there is no break-even (teacher's marginal cost is
    already <= the student's marginal electricity cost).
    """
    p.validate()
    t_req = teacher_cost_per_request(pricing, avg_input_tokens, avg_output_tokens)
    gpu_daily = gpu_amortized_usd_per_hour(p) * HOURS_PER_DAY
    # energy consumed per request = busy-time per request * kW * rate
    energy_per_req = electricity_usd_per_hour(p) / (p.throughput_rps * 3600.0)
    marginal_gap = t_req - energy_per_req
    if marginal_gap <= 0:
        return None
    return gpu_daily / marginal_gap


def cost_multiple(
    pricing: TeacherPricing,
    avg_input_tokens: float,
    avg_output_tokens: float,
    p: StudentCostParams,
) -> float:
    """The headline 'Nx cheaper' number: teacher $/1k over student $/1k."""
    s = student_cost_per_1k(p)
    if s <= 0:
        raise ValueError("student cost must be positive")
    return teacher_cost_per_1k(pricing, avg_input_tokens, avg_output_tokens) / s


def daily_cost_curves(
    pricing: TeacherPricing,
    avg_input_tokens: float,
    avg_output_tokens: float,
    p: StudentCostParams,
    volumes: list[int],
) -> dict[str, list[float]]:
    """Teacher vs student daily USD across request volumes (for the dashboard)."""
    t_req = teacher_cost_per_request(pricing, avg_input_tokens, avg_output_tokens)
    gpu_daily = gpu_amortized_usd_per_hour(p) * HOURS_PER_DAY
    energy_per_req = electricity_usd_per_hour(p) / (p.throughput_rps * 3600.0)
    return {
        "volume": [float(v) for v in volumes],
        "teacher_usd_per_day": [t_req * v for v in volumes],
        "student_usd_per_day": [gpu_daily + energy_per_req * v for v in volumes],
    }


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def params_from_config(cfg: dict[str, Any]) -> tuple[TeacherPricing, StudentCostParams, float, float]:
    """Build (TeacherPricing, StudentCostParams, avg_in, avg_out) from the
    YAML config, using the price-table row of the configured teacher model."""
    ccfg = cfg["cost"]
    tcfg = cfg["teacher"]
    table = tcfg.get("price_table", {})
    model = tcfg["model"]
    row = table.get(model)
    if row is None:
        for key, r in table.items():
            if model.startswith(key):
                row = r
                break
    if row is None:
        raise KeyError(f"no price-table entry for teacher model {model!r}")
    pricing = TeacherPricing(
        input_per_mtok=float(row["input_per_mtok"]),
        output_per_mtok=float(row["output_per_mtok"]),
    )
    params = StudentCostParams(
        gpu_price_usd=float(ccfg["gpu_price_usd"]),
        gpu_lifetime_years=float(ccfg["gpu_lifetime_years"]),
        gpu_utilization=float(ccfg["gpu_utilization"]),
        gpu_power_watts=float(ccfg["gpu_power_watts"]),
        electricity_usd_per_kwh=float(ccfg["electricity_usd_per_kwh"]),
        throughput_rps=float(ccfg["student_throughput_rps"]),
    )
    return (
        pricing,
        params,
        float(ccfg["teacher_avg_input_tokens"]),
        float(ccfg["teacher_avg_output_tokens"]),
    )