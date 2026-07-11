"""Cost dashboard (SPEC §8 deliverable): teacher-API vs local-student economics.

An interactive Streamlit page that reads the measured eval JSON in `reports/`
and lets you drag the assumptions — **requests/day, teacher token prices, GPU
price / power / electricity** — to see, live:

- a **break-even chart**: teacher vs student *daily* USD across request volume,
  with the break-even crossover and your selected volume marked;
- a **quality-vs-cost table** (the "money table"): metric M student-vs-teacher,
  teacher-agreement, schema-valid rate, latency p50/p95, and $/1k both sides.

Run it:  `streamlit run dashboard/app.py`

The `import streamlit` is guarded so this file still imports/compiles in a
plain-Python environment (e.g. the CI/laptop test box) that has no Streamlit;
the pure helpers below are importable and unit-testable either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from distil_task.config import load_config, resolve_path
from distil_task.cost_model import (
    StudentCostParams,
    TeacherPricing,
    break_even_requests_per_day,
    cost_multiple,
    daily_cost_curves,
    student_cost_per_1k,
    teacher_cost_per_1k,
)

try:  # guarded: keep this module importable where Streamlit is absent
    import streamlit as st

    _HAVE_STREAMLIT = True
except ImportError:  # pragma: no cover - exercised only on the GPU/dashboard box
    st = None  # type: ignore[assignment]
    _HAVE_STREAMLIT = False


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit dependency)
# ---------------------------------------------------------------------------

def load_eval_report(cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Load reports/eval_report.json if it exists (produced by evaluate.py)."""
    path = resolve_path(cfg, "reports_dir") / "eval_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def volume_axis(max_volume: float, n: int = 60) -> list[int]:
    """A clean ascending list of request volumes for the x-axis."""
    top = max(int(max_volume), 10)
    step = max(top // n, 1)
    return list(range(0, top + step, step))


def summary_rows(
    pricing: TeacherPricing,
    params: StudentCostParams,
    avg_in: float,
    avg_out: float,
    report: Optional[dict[str, Any]],
    metric_name: str,
    bar_ratio: float,
) -> list[dict[str, str]]:
    """Build the quality-vs-cost ("money") table as a list of row dicts."""
    t_1k = teacher_cost_per_1k(pricing, avg_in, avg_out)
    s_1k = student_cost_per_1k(params)
    mult = cost_multiple(pricing, avg_in, avg_out, params)
    breakeven = break_even_requests_per_day(pricing, avg_in, avg_out, params)

    q = (report or {}).get("quality", {}) or {}
    tq = (report or {}).get("teacher_quality") or {}
    student_m = q.get(metric_name)
    teacher_m = tq.get(metric_name, 1.0)
    ratio = (report or {}).get("student_over_teacher_ratio")
    if ratio is None and student_m is not None and teacher_m:
        ratio = student_m / teacher_m
    agreement = (report or {}).get("teacher_agreement")
    latency = (report or {}).get("latency_s", {}) or {}

    def pct(x: Optional[float]) -> str:
        return f"{x:.1%}" if isinstance(x, (int, float)) else "n/a"

    def num(x: Optional[float], fmt: str) -> str:
        return format(x, fmt) if isinstance(x, (int, float)) else "n/a"

    rows = [
        {"Metric": "$/1k requests", "Teacher (API)": f"${t_1k:,.2f}",
         "Student (local)": f"${s_1k:,.4f}", "Win": f"{mult:,.0f}x cheaper"},
        {"Metric": f"Quality ({metric_name})", "Teacher (API)": num(teacher_m, ".4f") + " (ref)",
         "Student (local)": num(student_m, ".4f"),
         "Win": (f"{pct(ratio)} of teacher — "
                 + ("meets" if isinstance(ratio, (int, float)) and ratio >= bar_ratio else "below")
                 + f" {bar_ratio:.0%} bar")},
        {"Metric": "Teacher-agreement (fresh pool)", "Teacher (API)": "—",
         "Student (local)": pct(agreement), "Win": "generalization"},
        {"Metric": "Schema-valid rate", "Teacher (API)": "—",
         "Student (local)": pct(q.get("schema_valid_rate")), "Win": "robustness"},
        {"Metric": "Exact match", "Teacher (API)": "—",
         "Student (local)": pct(q.get("exact_match")), "Win": "strictest view"},
        {"Metric": "p95 latency", "Teacher (API)": "~2.1 s (typ. API)",
         "Student (local)": (num(latency.get("p95", None) and latency["p95"] * 1000, ",.0f") + " ms")
         if latency.get("p95") is not None else "n/a", "Win": "lower"},
        {"Metric": "Data egress", "Teacher (API)": "leaves org",
         "Student (local)": "stays local", "Win": "privacy"},
        {"Metric": "Break-even volume", "Teacher (API)": "—",
         "Student (local)": (f"~{breakeven:,.0f} req/day" if breakeven else "none"),
         "Win": "above this, local wins"},
    ]
    return rows


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render() -> None:  # pragma: no cover - requires Streamlit runtime
    import pandas as pd  # Streamlit always ships pandas

    st.set_page_config(page_title="Distilled invoice model — cost dashboard",
                       page_icon="💸", layout="wide")
    st.title("Distilled invoice extractor — cost & quality dashboard")
    st.caption("Teacher API vs local RTX 5080 student. Drag the assumptions; "
               "the break-even and money table update live.")

    cfg = load_config()
    report = load_eval_report(cfg)
    ccfg = cfg["cost"]
    tcfg = cfg["teacher"]
    metric_name = str(cfg["task"]["metric"])
    bar_ratio = float(cfg["task"]["quality_bar_ratio"])

    if report is None:
        st.warning("No `reports/eval_report.json` found — run `python scripts/evaluate.py` "
                   "on the GPU box first. Showing config-default assumptions; quality cells "
                   "will read 'n/a'.")

    # ---- defaults (measured report overrides config) ----------------------
    price_row = (tcfg.get("price_table", {}) or {}).get(tcfg["model"], {})
    default_in_price = float(price_row.get("input_per_mtok", 3.0))
    default_out_price = float(price_row.get("output_per_mtok", 15.0))
    assumptions = ((report or {}).get("cost", {}) or {}).get("assumptions", {}) or {}
    default_avg_in = float(assumptions.get("teacher_avg_input_tokens", ccfg["teacher_avg_input_tokens"]))
    default_avg_out = float(assumptions.get("teacher_avg_output_tokens", ccfg["teacher_avg_output_tokens"]))
    measured_rps = float(assumptions.get("measured_throughput_rps", ccfg["student_throughput_rps"]))

    # ---- sidebar sliders --------------------------------------------------
    st.sidebar.header("Volume")
    requests_per_day = st.sidebar.slider("Requests / day", 100, 200_000, 20_000, step=100)

    st.sidebar.header("Teacher (API) pricing")
    in_price = st.sidebar.slider("Input $ / 1M tokens", 0.0, 30.0, default_in_price, step=0.25)
    out_price = st.sidebar.slider("Output $ / 1M tokens", 0.0, 120.0, default_out_price, step=0.5)
    avg_in = st.sidebar.number_input("Avg input tokens / request", 1.0, 100_000.0, default_avg_in, step=50.0)
    avg_out = st.sidebar.number_input("Avg output tokens / request", 1.0, 100_000.0, default_avg_out, step=25.0)

    st.sidebar.header("Student (local GPU)")
    gpu_price = st.sidebar.slider("GPU price $", 200.0, 5_000.0, float(ccfg["gpu_price_usd"]), step=50.0)
    gpu_watts = st.sidebar.slider("GPU power (W)", 50.0, 700.0, float(ccfg["gpu_power_watts"]), step=10.0)
    kwh_price = st.sidebar.slider("Electricity $ / kWh", 0.0, 1.0, float(ccfg["electricity_usd_per_kwh"]), step=0.01)
    throughput = st.sidebar.slider("Throughput (req/s)", 0.1, 50.0, measured_rps, step=0.1)
    lifetime = st.sidebar.slider("GPU amortization (years)", 1.0, 6.0, float(ccfg["gpu_lifetime_years"]), step=0.5)

    pricing = TeacherPricing(input_per_mtok=in_price, output_per_mtok=out_price)
    params = StudentCostParams(
        gpu_price_usd=gpu_price,
        gpu_lifetime_years=lifetime,
        gpu_utilization=float(ccfg["gpu_utilization"]),
        gpu_power_watts=gpu_watts,
        electricity_usd_per_kwh=kwh_price,
        throughput_rps=throughput,
    )

    # ---- headline numbers -------------------------------------------------
    t_1k = teacher_cost_per_1k(pricing, avg_in, avg_out)
    s_1k = student_cost_per_1k(params)
    mult = cost_multiple(pricing, avg_in, avg_out, params)
    breakeven = break_even_requests_per_day(pricing, avg_in, avg_out, params)

    # Exact daily cost at the selected volume (fixed GPU amortization + variable parts).
    curves_here = daily_cost_curves(pricing, avg_in, avg_out, params, [requests_per_day])
    t_day = curves_here["teacher_usd_per_day"][0]
    s_day = curves_here["student_usd_per_day"][0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teacher $/1k", f"${t_1k:,.2f}")
    c2.metric("Student $/1k", f"${s_1k:,.4f}")
    c3.metric("Cost multiple", f"{mult:,.0f}x cheaper")
    c4.metric("Break-even", f"{breakeven:,.0f}/day" if breakeven else "none")

    d1, d2 = st.columns(2)
    d1.metric(f"Teacher cost @ {requests_per_day:,}/day", f"${t_day:,.2f}")
    d2.metric(f"Student cost @ {requests_per_day:,}/day", f"${s_day:,.2f}",
              delta=f"${t_day - s_day:,.2f} saved/day", delta_color="normal")

    # ---- break-even chart -------------------------------------------------
    st.subheader("Break-even: daily cost vs request volume")
    top = max(requests_per_day * 2, (breakeven or 0) * 2, 2000)
    volumes = volume_axis(top)
    curves = daily_cost_curves(pricing, avg_in, avg_out, params, volumes)
    chart_df = pd.DataFrame(
        {"Teacher (API)": curves["teacher_usd_per_day"],
         "Student (local)": curves["student_usd_per_day"]},
        index=pd.Index(curves["volume"], name="requests/day"),
    )
    st.line_chart(chart_df, y=["Teacher (API)", "Student (local)"])
    if breakeven:
        st.caption(f"Lines cross at ~**{breakeven:,.0f} requests/day** — above it, the local "
                   f"student is cheaper. Your selected volume: {requests_per_day:,}/day "
                   f"({'local wins' if requests_per_day >= breakeven else 'teacher still cheaper'}).")
    else:
        st.caption("No break-even: the teacher's marginal per-request cost is already below the "
                   "student's marginal electricity cost (raise teacher price or token counts).")

    # ---- quality-vs-cost table -------------------------------------------
    st.subheader("Quality-vs-cost (money table)")
    rows = summary_rows(pricing, params, avg_in, avg_out, report, metric_name, bar_ratio)
    st.table(pd.DataFrame(rows).set_index("Metric"))

    with st.expander("Assumptions & provenance"):
        st.json({
            "teacher_model": tcfg["model"],
            "metric": metric_name,
            "quality_bar_ratio": bar_ratio,
            "teacher_price_input_per_mtok": in_price,
            "teacher_price_output_per_mtok": out_price,
            "teacher_avg_input_tokens": avg_in,
            "teacher_avg_output_tokens": avg_out,
            "gpu_price_usd": gpu_price,
            "gpu_power_watts": gpu_watts,
            "electricity_usd_per_kwh": kwh_price,
            "throughput_rps": throughput,
            "gpu_lifetime_years": lifetime,
            "eval_report_loaded": report is not None,
        })


def main() -> int:
    if not _HAVE_STREAMLIT:
        print(
            "Streamlit is not installed in this environment.\n"
            "Install it on the dashboard box and launch the app:\n"
            "    pip install streamlit\n"
            "    streamlit run dashboard/app.py"
        )
        return 1
    render()
    return 0


if __name__ == "__main__":
    # `streamlit run` and `python dashboard/app.py` both enter here.
    raise SystemExit(main())
