"""Cost-model tests: teacher vs student $/1k, cost multiple, break-even volume,
daily curves, and config plumbing."""
from __future__ import annotations

import pytest

from distil_task.cost_model import (
    StudentCostParams,
    TeacherPricing,
    break_even_requests_per_day,
    cost_multiple,
    daily_cost_curves,
    electricity_usd_per_hour,
    gpu_amortized_usd_per_hour,
    params_from_config,
    student_cost_per_1k,
    student_cost_per_request,
    teacher_cost_per_1k,
    teacher_cost_per_request,
)

PRICING = TeacherPricing(input_per_mtok=3.0, output_per_mtok=15.0)
AVG_IN, AVG_OUT = 900.0, 350.0
PARAMS = StudentCostParams()   # defaults: $1100 GPU, 3yr, 360W, $0.15/kWh, 4 rps


# ---------------------------------------------------------------------------
# Teacher side
# ---------------------------------------------------------------------------

def test_teacher_cost_per_request():
    # 900/1e6*3 + 350/1e6*15 = 0.0027 + 0.00525 = 0.00795
    assert teacher_cost_per_request(PRICING, AVG_IN, AVG_OUT) == pytest.approx(0.00795)


def test_teacher_cost_per_1k():
    assert teacher_cost_per_1k(PRICING, AVG_IN, AVG_OUT) == pytest.approx(7.95)


def test_teacher_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        teacher_cost_per_request(PRICING, -1, 10)


# ---------------------------------------------------------------------------
# Student side
# ---------------------------------------------------------------------------

def test_gpu_amortized_per_hour():
    # 1100 / (3 * 365 * 24) * 1.0
    assert gpu_amortized_usd_per_hour(PARAMS) == pytest.approx(1100.0 / (3 * 365 * 24))


def test_electricity_per_hour():
    # 360W = 0.36 kW * $0.15 = 0.054
    assert electricity_usd_per_hour(PARAMS) == pytest.approx(0.054)


def test_student_cost_per_request_and_1k():
    per_hour = gpu_amortized_usd_per_hour(PARAMS) + electricity_usd_per_hour(PARAMS)
    expected_req = per_hour / (4.0 * 3600.0)
    assert student_cost_per_request(PARAMS) == pytest.approx(expected_req)
    assert student_cost_per_1k(PARAMS) == pytest.approx(expected_req * 1000)


def test_student_much_cheaper_than_teacher():
    assert student_cost_per_1k(PARAMS) < teacher_cost_per_1k(PRICING, AVG_IN, AVG_OUT)


# ---------------------------------------------------------------------------
# Cost multiple + break-even
# ---------------------------------------------------------------------------

def test_cost_multiple_is_large():
    mult = cost_multiple(PRICING, AVG_IN, AVG_OUT, PARAMS)
    assert mult > 1000   # teacher $7.95/1k vs student <$0.01/1k


def test_break_even_matches_closed_form():
    be = break_even_requests_per_day(PRICING, AVG_IN, AVG_OUT, PARAMS)
    t_req = teacher_cost_per_request(PRICING, AVG_IN, AVG_OUT)
    gpu_daily = gpu_amortized_usd_per_hour(PARAMS) * 24.0
    energy_per_req = electricity_usd_per_hour(PARAMS) / (PARAMS.throughput_rps * 3600.0)
    expected = gpu_daily / (t_req - energy_per_req)
    assert be == pytest.approx(expected)
    assert be == pytest.approx(126.4, abs=1.0)   # sanity: ~126 req/day


def test_break_even_none_when_teacher_marginal_below_student_energy():
    cheap_teacher = TeacherPricing(input_per_mtok=0.0, output_per_mtok=0.0)
    assert break_even_requests_per_day(cheap_teacher, AVG_IN, AVG_OUT, PARAMS) is None


def test_above_break_even_local_is_cheaper():
    be = break_even_requests_per_day(PRICING, AVG_IN, AVG_OUT, PARAMS)
    curves = daily_cost_curves(PRICING, AVG_IN, AVG_OUT, PARAMS, [int(be * 2)])
    assert curves["student_usd_per_day"][0] < curves["teacher_usd_per_day"][0]


def test_below_break_even_teacher_is_cheaper():
    be = break_even_requests_per_day(PRICING, AVG_IN, AVG_OUT, PARAMS)
    low = max(int(be / 2), 1)
    curves = daily_cost_curves(PRICING, AVG_IN, AVG_OUT, PARAMS, [low])
    assert curves["teacher_usd_per_day"][0] < curves["student_usd_per_day"][0]


# ---------------------------------------------------------------------------
# Daily curves
# ---------------------------------------------------------------------------

def test_daily_curves_structure_and_monotonicity():
    volumes = [0, 1000, 5000, 20000]
    curves = daily_cost_curves(PRICING, AVG_IN, AVG_OUT, PARAMS, volumes)
    assert curves["volume"] == [0.0, 1000.0, 5000.0, 20000.0]
    # teacher is purely variable -> 0 at volume 0
    assert curves["teacher_usd_per_day"][0] == pytest.approx(0.0)
    # student carries fixed GPU amortization even at volume 0
    assert curves["student_usd_per_day"][0] == pytest.approx(gpu_amortized_usd_per_hour(PARAMS) * 24.0)
    # both non-decreasing
    for series in ("teacher_usd_per_day", "student_usd_per_day"):
        vals = curves[series]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"gpu_lifetime_years": 0.0},
    {"gpu_utilization": 0.0},
    {"gpu_utilization": 1.5},
    {"throughput_rps": 0.0},
    {"gpu_price_usd": -1.0},
])
def test_student_params_validation(kwargs):
    with pytest.raises(ValueError):
        StudentCostParams(**kwargs).validate()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def _sample_cfg() -> dict:
    return {
        "teacher": {
            "model": "claude-sonnet-4-5",
            "price_table": {
                "claude-sonnet-4-5": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
            },
        },
        "cost": {
            "gpu_price_usd": 1100.0,
            "gpu_lifetime_years": 3.0,
            "gpu_utilization": 1.0,
            "gpu_power_watts": 360.0,
            "electricity_usd_per_kwh": 0.15,
            "student_throughput_rps": 4.0,
            "teacher_avg_input_tokens": 900,
            "teacher_avg_output_tokens": 350,
        },
    }


def test_params_from_config_exact_and_prefix_match():
    pricing, params, avg_in, avg_out = params_from_config(_sample_cfg())
    assert pricing.input_per_mtok == 3.0 and pricing.output_per_mtok == 15.0
    assert params.gpu_price_usd == 1100.0 and params.throughput_rps == 4.0
    assert (avg_in, avg_out) == (900.0, 350.0)

    # prefix match: a dated model id resolves to its family row
    cfg = _sample_cfg()
    cfg["teacher"]["model"] = "claude-sonnet-4-5-20990101"
    pricing2, _, _, _ = params_from_config(cfg)
    assert pricing2.input_per_mtok == 3.0


def test_params_from_config_unknown_model_raises():
    cfg = _sample_cfg()
    cfg["teacher"]["model"] = "gpt-nonexistent"
    with pytest.raises(KeyError):
        params_from_config(cfg)
