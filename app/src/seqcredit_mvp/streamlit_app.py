"""SeqCredit Institutional Risk Intelligence Platform.

Engineered for risk committees, banking institutions, and credit bureaus to evaluate
temporal sequence modeling versus static aggregate credit-risk discrimination.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from seqcredit_mvp.dashboard import build_dashboard_payload
from seqcredit_mvp.presets import ARCHETYPE_DEFINITIONS, detect_archetypes, filter_cohort
from seqcredit_mvp.reports import (
    generate_borrower_memo_html,
    generate_borrower_memo_pdf,
    generate_portfolio_summary_html,
    generate_portfolio_summary_pdf,
)
from seqcredit_mvp.scenarios import (
    DEFAULT_TENURES,
    TENURE_CONFIGS,
    apply_cashflow_stress,
    compute_borrower_liquidity,
    evaluate_borrower_stress,
    evaluate_window_sensitivity,
    scenario_index,
    simulate_multi_term_curves,
    simulate_policy_sandbox,
    simulate_term_sizing,
)
from seqcredit_mvp.scoring import DEFAULT_ARTIFACT_DIR, DemoScorer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_CSV_PATH = DATA_DIR / "sample_transactions.csv"


@st.cache_resource
def get_scorer(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> DemoScorer:
    """Load and cache calibrated scoring models and governance manifest."""
    return DemoScorer(artifact_dir)


@st.cache_data
def process_data(csv_bytes: bytes, _scorer: DemoScorer) -> dict:
    """Validate schema, evaluate model contracts, and build portfolio analytics payload."""
    transactions = pd.read_csv(io.BytesIO(csv_bytes))
    return build_dashboard_payload(transactions, _scorer)


def trend_label(value: float) -> str:
    if value > 0.05:
        return "Increasing Trajectory"
    if value < -0.05:
        return "Contracting Trajectory"
    return "Stable Trajectory"


def comparison_phrase(ratio: float) -> str:
    if ratio < 0.95:
        return "below baseline historical averages"
    if ratio > 1.05:
        return "above baseline historical averages"
    return "broadly consistent with historical baseline"


def format_money(val: float | int) -> str:
    return f"GHS {val:,.2f}"


# --- INSTITUTIONAL ECHARTS BUILDERS ---


def build_timeline_echarts_options(
    dates: list[str], amounts: list[float], balances: list[float]
) -> dict:
    return {
        "tooltip": {
            "trigger": "axis",
            "valueFormatter": JsCode("function(v){return 'GHS ' + Number(v).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});}"),
        },
        "legend": {
            "data": ["Transaction Amount", "Opening Balance"],
            "top": "0%",
            "right": "2%",
            "textStyle": {"color": "#475569", "fontSize": 10},
        },
        "toolbox": {
            "show": False,
        },
        "grid": {"left": "2%", "right": "3%", "top": "12%", "bottom": "8%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": dates,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {
                "color": "#64748b",
                "fontSize": 10,
                "formatter": "GHS {value}",
            },
        },
        "dataZoom": [
            {"type": "inside", "start": 0, "end": 100},
        ],
        "series": [
            {
                "name": "Transaction Amount",
                "type": "line",
                "smooth": True,
                "data": amounts,
                "itemStyle": {"color": "#2563eb"},
                "areaStyle": {"opacity": 0.08, "color": "#2563eb"},
            },
            {
                "name": "Opening Balance",
                "type": "line",
                "smooth": True,
                "data": balances,
                "itemStyle": {"color": "#64748b"},
                "lineStyle": {"width": 2, "type": "dashed"},
            },
        ],
    }


def build_loan_curve_echarts_options(
    curve_amounts: list[int],
    curve_risks: list[float],
    selected_amount: int,
    selected_risk: float,
    multi_curves: dict[int, list[float]] | None = None,
    selected_tenure: int = 30,
) -> dict:
    if multi_curves:
        tenure_colors = {
            7: "#dc2626",   # Red
            14: "#d97706",  # Amber
            30: "#2563eb",  # Enterprise blue
        }
        series = []
        for tenure in [7, 14, 30]:
            if tenure in multi_curves:
                is_selected = (tenure == selected_tenure)
                label = f"{tenure}-Day Facility"
                s = {
                    "name": label,
                    "type": "line",
                    "smooth": True,
                    "data": [round(r, 4) for r in multi_curves[tenure]],
                    "itemStyle": {"color": tenure_colors.get(tenure, "#2563eb")},
                    "lineStyle": {
                        "width": 2.5 if is_selected else 1.5,
                        "type": "solid" if is_selected else "dashed",
                    },
                    "areaStyle": {
                        "opacity": 0.08 if is_selected else 0.0,
                        "color": tenure_colors.get(tenure, "#2563eb"),
                    },
                }
                if is_selected:
                    s["markPoint"] = {
                        "data": [
                            {
                                "name": "Active Scenario",
                                "coord": [str(selected_amount), round(selected_risk, 4)],
                                "value": f"{round(selected_risk * 100)} / 100 ({tenure}D)",
                                "itemStyle": {"color": tenure_colors.get(tenure, "#2563eb")},
                            }
                        ]
                    }
                series.append(s)
        legend = {
            "data": [f"{t}-Day Facility" for t in [7, 14, 30] if t in multi_curves],
            "top": "0%",
            "right": "2%",
            "textStyle": {"fontSize": 10, "color": "#475569"},
        }
        tooltip = {
            "trigger": "axis",
            "formatter": JsCode(
                "function(params){"
                "  var out = 'Facility Principal: GHS ' + Number(params[0].name).toLocaleString() + '<br/>';"
                "  params.forEach(function(p){"
                "    out += p.marker + ' ' + p.seriesName + ': <b>' + Math.round(p.value * 100) + ' / 100</b><br/>';"
                "  });"
                "  return out;"
                "}"
            ),
        }
    else:
        series = [
            {
                "name": "Stress Curve",
                "type": "line",
                "smooth": True,
                "data": [round(r, 4) for r in curve_risks],
                "itemStyle": {"color": "#2563eb"},
                "areaStyle": {"opacity": 0.08, "color": "#2563eb"},
                "markPoint": {
                    "data": [
                        {
                            "name": "Active Scenario",
                            "coord": [str(selected_amount), round(selected_risk, 4)],
                            "value": f"{round(selected_risk * 100)} / 100",
                            "itemStyle": {"color": "#2563eb"},
                        }
                    ]
                },
            }
        ]
        legend = None
        tooltip = {
            "trigger": "axis",
            "formatter": JsCode(
                "function(params){"
                "  var p = params[0];"
                "  return 'Simulated Facility: GHS ' + Number(p.name).toLocaleString() + '<br/>Stress Index: ' + Math.round(p.value * 100) + ' / 100';"
                "}"
            ),
        }

    opts = {
        "tooltip": tooltip,
        "toolbox": {
            "show": False,
        },
        "grid": {"left": "2%", "right": "3%", "top": "8%", "bottom": "12%" if legend else "6%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": [str(a) for a in curve_amounts],
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "min": 0.0,
            "max": 1.0,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "series": series,
    }
    if legend:
        opts["legend"] = legend
    return opts


def build_stress_comparison_echarts_options(
    baseline_score: float,
    stressed_score: float,
    baseline_band: str,
    stressed_band: str,
) -> dict:
    categories = ["Baseline", "Stressed Shock"]
    values = [round(baseline_score * 100, 1), round(stressed_score * 100, 1)]
    colors = ["#2563eb", "#dc2626" if stressed_score >= baseline_score else "#059669"]

    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "16%", "bottom": "8%", "top": "8%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": 100,
            "axisLabel": {"fontSize": 10, "color": "#64748b"},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
        },
        "yAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"fontSize": 11, "fontWeight": 600, "color": "#1e293b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        },
        "series": [
            {
                "name": "Risk Index",
                "type": "bar",
                "barWidth": 24,
                "data": [
                    {"value": values[0], "itemStyle": {"color": colors[0]}},
                    {"value": values[1], "itemStyle": {"color": colors[1]}},
                ],
                "label": {
                    "show": True,
                    "position": "right",
                    "formatter": "{c} / 100",
                    "fontSize": 11,
                    "fontWeight": 600,
                    "color": "#0f172a",
                },
            }
        ],
    }


def build_gauge_echarts_options(score_val: float) -> dict:
    return {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "min": 0,
                "max": 100,
                "splitNumber": 5,
                "itemStyle": {"color": "#2563eb"},
                "progress": {"show": True, "roundCap": True, "width": 12},
                "pointer": {"show": False},
                "axisLine": {"roundCap": True, "lineStyle": {"width": 12, "color": [[1, "#e2e8f0"]]}},
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": True, "distance": 20, "fontSize": 9, "color": "#64748b"},
                "title": {"show": True, "offsetCenter": [0, "22%"], "fontSize": 11, "fontWeight": 600, "color": "#64748b"},
                "detail": {
                    "valueAnimation": True,
                    "offsetCenter": [0, "-15%"],
                    "fontSize": 24,
                    "fontWeight": "bold",
                    "formatter": "{value}",
                    "color": "#0f172a",
                },
                "data": [{"value": round(score_val), "name": "Stress Index"}],
            }
        ]
    }


def build_event_mix_echarts_options(mix_data: list[dict[str, str | float]]) -> dict:
    return {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: <b>{d}%</b> ({c})",
        },
        "legend": {
            "bottom": "0%",
            "left": "center",
            "textStyle": {"color": "#475569", "fontSize": 10},
        },
        "color": ["#2563eb", "#3b82f6", "#60a5fa", "#94a3b8", "#cbd5e1"],
        "series": [
            {
                "name": "Transaction Type",
                "type": "pie",
                "radius": ["45%", "72%"],
                "center": ["50%", "45%"],
                "avoidLabelOverlap": False,
                "itemStyle": {"borderRadius": 4, "borderColor": "#ffffff", "borderWidth": 2},
                "label": {"show": False, "position": "center"},
                "emphasis": {
                    "label": {"show": True, "fontSize": 13, "fontWeight": "bold", "color": "#0f172a"}
                },
                "data": mix_data,
            }
        ],
    }


def build_batch_distribution_echarts_options(labels: list[str], counts: list[int]) -> dict:
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "2%", "right": "3%", "bottom": "8%", "top": "12%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisTick": {"alignWithLabel": True},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "series": [
            {
                "name": "Borrowers",
                "type": "bar",
                "barWidth": "42%",
                "data": counts,
                "itemStyle": {"color": "#2563eb", "borderRadius": [2, 2, 0, 0]},
                "label": {"show": True, "position": "top", "fontSize": 10, "color": "#475569"},
            }
        ],
    }


def build_auc_comparison_echarts_options(
    real_static: float, real_seq: float, demo_static: float, demo_seq: float
) -> dict:
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {
            "data": ["Static Baseline", "Order-Aware Sequence"],
            "top": "0%",
            "right": "2%",
            "textStyle": {"color": "#475569", "fontSize": 10},
        },
        "grid": {"left": "3%", "right": "6%", "bottom": "6%", "top": "14%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "min": 0.68,
            "max": 0.78,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10, "formatter": "{value}"},
        },
        "yAxis": {
            "type": "category",
            "data": ["Archived Study", "Cohort Demo"],
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#334155", "fontSize": 11, "fontWeight": 500},
        },
        "series": [
            {
                "name": "Static Baseline",
                "type": "bar",
                "data": [round(real_static, 4), round(demo_static, 4)],
                "itemStyle": {"color": "#94a3b8", "borderRadius": [0, 2, 2, 0]},
                "label": {"show": True, "position": "insideRight", "color": "#ffffff", "fontSize": 10},
            },
            {
                "name": "Order-Aware Sequence",
                "type": "bar",
                "data": [round(real_seq, 4), round(demo_seq, 4)],
                "itemStyle": {"color": "#2563eb", "borderRadius": [0, 2, 2, 0]},
                "label": {"show": True, "position": "insideRight", "color": "#ffffff", "fontSize": 10},
            },
        ],
    }


def build_calibration_echarts_options(calibration_data: dict) -> dict:
    curve = calibration_data.get("calibration_curve", {})
    stat = curve.get("static", {})
    seq = curve.get("sequence", {})

    stat_pts = [
        [round(x, 4), round(y, 4)]
        for x, y in zip(stat.get("mean_predicted_value", []), stat.get("fraction_of_positives", []))
    ]
    seq_pts = [
        [round(x, 4), round(y, 4)]
        for x, y in zip(seq.get("mean_predicted_value", []), seq.get("fraction_of_positives", []))
    ]

    return {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
        },
        "legend": {
            "data": ["Perfect Calibration", "Static Baseline", "Order-Aware Sequence"],
            "right": "2%",
            "top": "0%",
            "textStyle": {"color": "#475569", "fontSize": 10},
        },
        "grid": {"left": "3%", "right": "4%", "bottom": "12%", "top": "14%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "name": "Predicted Risk",
            "nameLocation": "middle",
            "nameGap": 20,
            "min": 0,
            "max": 1,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "name": "Default Rate",
            "nameLocation": "middle",
            "nameGap": 24,
            "min": 0,
            "max": 1,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "series": [
            {
                "name": "Perfect Calibration",
                "type": "line",
                "data": [[0, 0], [1, 1]],
                "lineStyle": {"color": "#94a3b8", "type": "dashed", "width": 1.5},
                "symbol": "none",
                "silent": True,
            },
            {
                "name": "Static Baseline",
                "type": "line",
                "data": stat_pts,
                "lineStyle": {"color": "#94a3b8", "width": 2},
                "itemStyle": {"color": "#94a3b8"},
                "symbolSize": 6,
            },
            {
                "name": "Order-Aware Sequence",
                "type": "line",
                "data": seq_pts,
                "lineStyle": {"color": "#2563eb", "width": 2.5},
                "itemStyle": {"color": "#2563eb"},
                "symbolSize": 7,
            },
        ],
    }


def build_sensitivity_echarts_options(window_points: list[dict]) -> dict:
    windows = [f"{p['window_size']} Ops" for p in window_points]
    seq_scores = [round(p["sequence_score"] * 100, 2) for p in window_points]
    stat_scores = [round(p["static_score"] * 100, 2) for p in window_points]
    deltas = [round(p["score_delta"] * 100, 2) for p in window_points]

    return {
        "tooltip": {"trigger": "axis"},
        "legend": {
            "data": ["Order-Aware Index", "Static Score", "Delta (pts)"],
            "right": "2%",
            "top": "0%",
            "textStyle": {"color": "#475569", "fontSize": 10},
        },
        "grid": {"left": "3%", "right": "4%", "bottom": "8%", "top": "12%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": windows,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#475569", "fontSize": 10, "fontWeight": 500},
        },
        "yAxis": [
            {
                "type": "value",
                "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
                "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
                "axisLabel": {"color": "#64748b", "fontSize": 10},
            },
            {
                "type": "value",
                "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
                "splitLine": {"show": False},
                "axisLabel": {"color": "#64748b", "fontSize": 10},
            },
        ],
        "series": [
            {
                "name": "Order-Aware Index",
                "type": "line",
                "data": seq_scores,
                "lineStyle": {"color": "#2563eb", "width": 2.5},
                "itemStyle": {"color": "#2563eb"},
                "symbolSize": 8,
            },
            {
                "name": "Static Score",
                "type": "line",
                "data": stat_scores,
                "lineStyle": {"color": "#94a3b8", "width": 2, "type": "dotted"},
                "itemStyle": {"color": "#94a3b8"},
                "symbolSize": 7,
            },
            {
                "name": "Delta (pts)",
                "type": "bar",
                "yAxisIndex": 1,
                "data": deltas,
                "itemStyle": {
                    "color": "#64748b",
                    "borderRadius": [2, 2, 0, 0],
                    "opacity": 0.65,
                },
                "barWidth": "30%",
            },
        ],
    }


def build_waterfall_echarts_options(waterfall_data: dict) -> dict:
    steps = waterfall_data.get("steps", [])
    base_idx = float(waterfall_data.get("baseline_index", 0.0))
    final_idx = float(waterfall_data.get("final_index", 0.0))

    label_map = {
        "Cohort Baseline": "Baseline",
        "Average Account Liquidity": "Avg Balance",
        "Minimum Reserve Liquidity": "Min Balance",
        "Liquidity Dispersion": "Volatility",
        "Disbursement Sizing Trajectory": "Disburse Trend",
        "Inter-Event Velocity": "Tx Velocity",
        "Liquidity Depletion Trajectory": "Deplete Trend",
        "Recent Cash-Out Concentration": "Recent Cashout",
        "Recent Liquidity Strain Frequency": "Strain Freq",
        "Recent Velocity Multiplier": "Velocity Multi",
        "Recent Liquidity Coverage": "Coverage Ratio",
        "Off-Hours Velocity Shift": "Night Shift",
        "Cash-Out Acceleration": "Cashout Trend",
        "Other Combined Factors": "Other Factors",
        "Account Index": "Account Score",
    }

    raw_categories = ["Cohort Baseline"] + [s["name"] for s in steps] + ["Account Index"]
    display_categories = [label_map.get(c, c if len(c) <= 13 else c[:11] + "..") for c in raw_categories]

    placeholders = [0.0]
    bar_values = [base_idx]
    colors = ["#2563eb"]
    labels_text = [f"{base_idx:.1f}"]

    current = base_idx
    for s in steps:
        impact = float(s["impact"])
        if impact >= 0:
            placeholders.append(round(current, 2))
            bar_values.append(round(impact, 2))
            colors.append("#dc2626")  # Red for risk escalation
            labels_text.append(f"+{impact:.1f}" if impact != 0 else "")
            current += impact
        else:
            current += impact
            placeholders.append(round(current, 2))
            bar_values.append(round(abs(impact), 2))
            colors.append("#16a34a")  # Green for risk mitigation
            labels_text.append(f"{impact:.1f}" if impact != 0 else "")

    placeholders.append(0.0)
    bar_values.append(round(final_idx, 2))
    colors.append("#1d4ed8")  # Dark blue for account score
    labels_text.append(f"{final_idx:.1f}")

    series_data = []
    for val, c, lbl in zip(bar_values, colors, labels_text):
        series_data.append({
            "value": val,
            "itemStyle": {"color": c, "borderRadius": 3},
            "label": {
                "show": True,
                "position": "top",
                "formatter": lbl,
                "fontSize": 10,
                "fontWeight": 600,
                "color": "#334155",
            },
        })

    return {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "formatter": JsCode(
                "function(params){"
                "  var p = params[1] || params[0];"
                "  var val = p.value;"
                "  return '<b>' + p.name + '</b><br/>Impact: ' + (val >= 0 ? '+' : '') + val + ' pts';"
                "}"
            ),
        },
        "grid": {"left": "2%", "right": "2%", "bottom": "10%", "top": "12%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": display_categories,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisTick": {"show": False},
            "axisLabel": {
                "color": "#475569",
                "fontSize": 10,
                "fontWeight": 500,
                "interval": 0,
                "rotate": 0,
            },
        },
        "yAxis": {
            "type": "value",
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "series": [
            {
                "name": "Base",
                "type": "bar",
                "stack": "Total",
                "silent": True,
                "itemStyle": {"borderColor": "transparent", "color": "transparent"},
                "data": placeholders,
            },
            {
                "name": "Attribution Impact",
                "type": "bar",
                "stack": "Total",
                "data": series_data,
            },
        ],
    }


def build_drivers_divergence_echarts_options(drivers: list[dict]) -> dict:
    sorted_drivers = sorted(drivers, key=lambda d: float(d["impact_points"]))
    names = [d["display_name"] for d in sorted_drivers]
    values = []
    for d in sorted_drivers:
        pts = float(d["impact_points"])
        color = "#dc2626" if pts >= 0 else "#16a34a"
        values.append({"value": pts, "itemStyle": {"color": color, "borderRadius": 2}})

    return {
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "formatter": JsCode(
                "function(params){"
                "  var p = params[0];"
                "  var sign = p.value >= 0 ? '+' : '';"
                "  return '<b>' + p.name + '</b><br/>Impact: ' + sign + p.value + ' pts';"
                "}"
            ),
        },
        "grid": {"left": "3%", "right": "6%", "top": "4%", "bottom": "6%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "name": "Index Points",
            "nameTextStyle": {"color": "#64748b", "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#f1f5f9"}},
            "axisLabel": {"color": "#64748b", "fontSize": 10},
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "axisLabel": {"color": "#334155", "fontSize": 11},
        },
        "series": [
            {
                "name": "Impact",
                "type": "bar",
                "data": values,
            }
        ],
    }


# --- MAIN APPLICATION ENTRYPOINT ---



def main() -> None:
    st.set_page_config(
        page_title="SeqCredit — Transaction Sequence Risk Monitor",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimalist, zero-scroll enterprise styling
    st.markdown(
        """
        <style>
        html, body, [class*="css"], .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        header[data-testid="stHeader"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.75rem !important;
            padding-bottom: 0.75rem !important;
            padding-left: 1.75rem !important;
            padding-right: 1.75rem !important;
            max-width: 100% !important;
        }
        h3 {
            margin-top: 0.2rem !important;
            padding-top: 0 !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
            line-height: 1.3 !important;
        }
        code, pre, .font-mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        }
        [data-testid="stMetric"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            box-shadow: none !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        }
        .compliance-pill {
            display: inline-block;
            background-color: #f1f5f9;
            color: #475569;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.70rem;
            font-weight: 600;
            border: 1px solid #e2e8f0;
        }
        .band-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.76rem;
        }
        .band-lower { background-color: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
        .band-watch { background-color: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .band-elevated { background-color: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .driver-badge-escalator { background-color: #fef2f2; color: #dc2626; border: 1px solid #fecaca; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; }
        .driver-badge-mitigator { background-color: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; }
        .summary-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #2563eb;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            line-height: 1.45;
            margin-bottom: 8px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 4px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            background-color: transparent;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 500;
            transition: all 0.15s ease-in-out;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background-color: #f1f5f9;
        }
        hr {
            margin: 0.65rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        scorer = get_scorer()
    except Exception as exc:
        st.error(f"Failed to load risk model contracts: {exc}")
        return

    # Auto-initialize session state with sample dataset
    if "csv_bytes" not in st.session_state or st.session_state.csv_bytes is None:
        if SAMPLE_CSV_PATH.exists():
            st.session_state.csv_bytes = SAMPLE_CSV_PATH.read_bytes()
            st.session_state.source_name = "sample_transactions.csv"

    if not st.session_state.csv_bytes:
        st.info("Awaiting batch ingestion. Upload a banking CSV file or load the sample cohort.")
        return

    try:
        payload = process_data(st.session_state.csv_bytes, scorer)
    except Exception as exc:
        st.error(f"Batch Processing Error: {exc}")
        return

    scores_df = pd.DataFrame(payload["scores"])
    borrowers_list = sorted(scores_df["borrower_id"].astype(str).unique().tolist())
    archetypes = payload.get("archetypes") or detect_archetypes(scores_df, payload["details"])

    # ---------------------------------------------------------
    # SIDEBAR: SIDE TABS & AUDIT CONTROLS
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 10px;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">SeqCredit</span>
                <span style="display: block; font-size: 0.72rem; color: #64748b; font-weight: 500;">Sequence Risk Intelligence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        NAV_OPTIONS = [
            "Portfolio Overview",
            "Account Profile",
            "Risk Attribution",
            "Stress & Scenarios",
            "Activity Ledger",
            "Model Benchmark",
        ]

        active_nav = st.radio(
            "Navigation",
            NAV_OPTIONS,
            index=0,
            key="nav_side_tab",
            label_visibility="collapsed",
        )

        is_account_tool = active_nav in [
            "Account Profile",
            "Risk Attribution",
            "Stress & Scenarios",
            "Activity Ledger",
        ]

        if is_account_tool:
            st.divider()
            st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#475569; margin-bottom:4px;'>ACCOUNT FOCUS</div>", unsafe_allow_html=True)

            if "selected_borrower" not in st.session_state or str(st.session_state.selected_borrower) not in borrowers_list:
                st.session_state.selected_borrower = borrowers_list[0]

            arch_map = {f"{v['title']} ({v['borrower_id']})": str(v["borrower_id"]) for v in archetypes.values()}
            arch_choice = st.selectbox(
                "Representative Archetypes:",
                options=["— All Accounts —"] + list(arch_map.keys()),
                index=0,
                key="sidebar_archetype_select",
            )
            if arch_choice != "— All Accounts —":
                st.session_state.selected_borrower = arch_map[arch_choice]

            selected_borrower = st.selectbox(
                "Select Counterparty:",
                options=borrowers_list,
                index=borrowers_list.index(str(st.session_state.selected_borrower)),
                key="sidebar_borrower_select",
            )
            st.session_state.selected_borrower = selected_borrower
        else:
            selected_borrower = str(st.session_state.get("selected_borrower", borrowers_list[0]))

        st.divider()
        with st.expander("Data & Runtime Controls", expanded=False):
            uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])
            if uploaded_file is not None:
                st.session_state.csv_bytes = uploaded_file.getvalue()
                st.session_state.source_name = uploaded_file.name
                st.rerun()

            if st.button("Reload Reference Cohort", width="stretch"):
                if SAMPLE_CSV_PATH.exists():
                    st.session_state.csv_bytes = SAMPLE_CSV_PATH.read_bytes()
                    st.session_state.source_name = "sample_transactions.csv"
                    st.rerun()

            st.caption(f"Artifact: `{scorer.manifest['model_version']}`")
            st.caption("Air-Gapped Runtime · Synthetic Reference Dataset")

    # Resolve active borrower records
    borrower_score = scores_df[scores_df["borrower_id"].astype(str) == str(selected_borrower)].iloc[0]
    borrower_detail = payload["details"].get(str(selected_borrower), {})
    indicators = borrower_detail.get("indicators", {})
    attribution = borrower_detail.get("attribution", {})
    recent_txs = pd.DataFrame(borrower_detail.get("recent_transactions", []))

    # ---------------------------------------------------------
    # TOOL 1: PORTFOLIO OVERVIEW
    # ---------------------------------------------------------
    if active_nav == "Portfolio Overview":
        portfolio = payload["portfolio"]
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown("### Portfolio Cohort Overview")
            st.caption("Macro risk distribution, cross-sectional analytics, and batch account matrix.")
        with h_col2:
            st.markdown(f"<div style='text-align:right;'><span class='compliance-pill'>Cohort: {portfolio['borrowers']:,} Accounts</span></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Counterparties", f"{portfolio['borrowers']:,}", help="Evaluated accounts in current batch.")
        with m2:
            st.metric("Audited Transactions", f"{portfolio['transactions']:,}", help="Total transaction records ingested.")
        with m3:
            st.metric("Cohort Mean Risk", f"{round(portfolio['average_sequence_score'] * 100, 1)} / 100", help="Batch-wide average order-aware risk index.")
        with m4:
            elev_count = portfolio["bands"].get("Elevated demo risk", 0)
            st.metric("Elevated Risk Accounts", f"{elev_count:,}", delta=f"{elev_count/portfolio['borrowers']*100:.1f}% of cohort", delta_color="inverse")

        st.markdown("<hr>", unsafe_allow_html=True)

        c_left, c_right = st.columns([1, 2], gap="medium")
        with c_left:
            st.markdown("##### Score Distribution")
            bins = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]
            bin_labels = ["0–4", "5–9", "10–19", "20–39", "40+"]
            counts = pd.cut(scores_df["sequence_demo_score"], bins=bins, labels=bin_labels, right=False).value_counts().sort_index()
            dist_opts = build_batch_distribution_echarts_options(bin_labels, counts.values.tolist())
            st_echarts(options=dist_opts, height="240px", key="echarts_batch_dist")

            exp_c1, exp_c2 = st.columns(2)
            with exp_c1:
                st.download_button(
                    "Export CSV",
                    icon=":material/download:",
                    data=scores_df.to_csv(index=False).encode("utf-8"),
                    file_name="seqcredit_portfolio_evaluation.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with exp_c2:
                st.download_button(
                    "Portfolio PDF",
                    icon=":material/picture_as_pdf:",
                    data=generate_portfolio_summary_pdf(payload, scorer.manifest),
                    file_name="seqcredit_portfolio_summary.pdf",
                    mime="application/pdf",
                    width="stretch",
                )

        with c_right:
            st.markdown("##### Evaluated Accounts Matrix")
            f1, f2, f3 = st.columns([1, 1, 2])
            with f1:
                sel_band = st.selectbox("Risk Band:", ["All", "Lower demo risk", "Watch demo risk", "Elevated demo risk"], index=0, key="pv_band_filter")
            with f2:
                sel_traj = st.selectbox("Trajectory:", ["All", "Deteriorating", "Stable", "Improving"], index=0, key="pv_traj_filter")
            with f3:
                search_query = st.text_input("Search Account ID:", placeholder="e.g. DEMO_001", key="pv_search_id")

            filtered_scores = filter_cohort(
                scores_df=scores_df,
                details_dict=payload["details"],
                risk_bands=[sel_band] if sel_band != "All" else None,
                trajectory_types=[sel_traj] if sel_traj != "All" else None,
            )
            if search_query:
                filtered_scores = filtered_scores[filtered_scores["borrower_id"].astype(str).str.contains(search_query.strip(), case=False)]

            display_cols = ["borrower_id", "sequence_demo_score", "static_demo_score", "score_delta", "demo_risk_band", "trajectory", "balance_tier"]
            clean_tbl = filtered_scores[[c for c in display_cols if c in filtered_scores.columns]].copy()
            clean_tbl["Order-Aware Index"] = (clean_tbl["sequence_demo_score"] * 100).round(1)
            clean_tbl["Static Baseline"] = (clean_tbl["static_demo_score"] * 100).round(1)
            clean_tbl["Delta (pts)"] = (clean_tbl["score_delta"] * 100).round(1)
            clean_tbl = clean_tbl.rename(columns={"borrower_id": "Account ID", "demo_risk_band": "Risk Band", "trajectory": "Trajectory", "balance_tier": "Liquidity"})
            st.dataframe(
                clean_tbl[["Account ID", "Order-Aware Index", "Static Baseline", "Delta (pts)", "Risk Band", "Trajectory", "Liquidity"]],
                width="stretch",
                hide_index=True,
                height=260,
            )

    # ---------------------------------------------------------
    # TOOL 2: ACCOUNT PROFILE
    # ---------------------------------------------------------
    elif active_nav == "Account Profile":
        band_name = borrower_score["demo_risk_band"].replace(" demo risk", "")
        band_css = "band-lower" if "lower" in band_name.lower() else ("band-watch" if "watch" in band_name.lower() else "band-elevated")
        seq_val = round(borrower_score["sequence_demo_score"] * 100)
        stat_val = round(borrower_score["static_demo_score"] * 100)
        diff_val = seq_val - stat_val
        balances_sparkline = recent_txs["balance_before"].iloc[::-1].tolist() if not recent_txs.empty else None

        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown(f"### Account Profile: `{selected_borrower}`")
            st.caption("Discrete sequence risk evaluation against cumulative static baseline.")
        with h_col2:
            st.markdown(f"<div style='text-align:right;'><span class='band-tag {band_css}'>{band_name.upper()} RISK</span></div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Sequence Risk Index", f"{seq_val} / 100", chart_data=balances_sparkline, chart_type="line", help="Order-aware risk index derived from temporal sequence dynamics.")
        with k2:
            st.metric("Static Baseline Score", f"{stat_val} / 100", delta=f"{diff_val:+d} Sequence Delta", delta_color="off", help="Aggregate static baseline excluding temporal sequence order.")
        with k3:
            st.metric("Disbursement Velocity", trend_label(indicators.get("amount_slope", 0.0)), help="Temporal trend in transaction sizes.")
        with k4:
            st.metric("Liquidity Stability", trend_label(indicators.get("balance_slope", 0.0)), help="Opening balance trajectory over recent history.")

        st.markdown("<hr>", unsafe_allow_html=True)

        c_left, c_right = st.columns([1, 1], gap="medium")
        with c_left:
            st.markdown("##### Risk Gauge & Narrative")
            gauge_opts = build_gauge_echarts_options(float(seq_val))
            st_echarts(options=gauge_opts, height="230px", key="profile_gauge")

            amt_comp = comparison_phrase(indicators.get("amount_recent_ratio", 1.0))
            bal_comp = comparison_phrase(indicators.get("balance_recent_ratio", 1.0))
            cashout_pct = round(indicators.get("pct_cashout", 0.0) * 100)
            st.markdown(
                f"""
                <div class="summary-card">
                    <strong>Executive Assessment:</strong><br>
                    Recent liquidity balances are <em>{bal_comp}</em>. Transaction velocity is <em>{amt_comp}</em>,
                    and cash-out operations account for <strong>{cashout_pct}%</strong> of evaluated events.
                    <br><small style="color:#64748b;">Synthetic prototype demonstration · Non-causal evaluation.</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_right:
            st.markdown("##### Quantitative Behavioral Indicators")
            ind_data = [
                {"Signal": "Cash-Out Concentration", "Value": f"{indicators.get('pct_cashout', 0)*100:.1f}%", "Context": "Share of outgoing transfers/cashouts"},
                {"Signal": "Night-Time Activity", "Value": f"{indicators.get('pct_night', 0)*100:.1f}%", "Context": "Operations between 22:00 and 05:00"},
                {"Signal": "Average Balance", "Value": f"GHS {indicators.get('avg_balance', 0):,.2f}", "Context": "Historical mean opening balance"},
                {"Signal": "Minimum Liquidity Floor", "Value": f"GHS {indicators.get('min_balance', 0):,.2f}", "Context": "Lowest observed balance point"},
                {"Signal": "Total Fees Incurred", "Value": f"GHS {indicators.get('total_fees', 0):,.2f}", "Context": "Cumulative fee burden"},
                {"Signal": "Audited Operations", "Value": f"{indicators.get('transaction_count', 0)} events", "Context": "Observation window size"},
            ]
            st.dataframe(pd.DataFrame(ind_data), hide_index=True, width="stretch", height=220)

            st.download_button(
                "Download Account Credit Memo (PDF)",
                icon=":material/picture_as_pdf:",
                data=generate_borrower_memo_pdf(str(selected_borrower), borrower_score.to_dict(), borrower_detail, scorer.manifest),
                file_name=f"seqcredit_memo_{selected_borrower}.pdf",
                mime="application/pdf",
                width="stretch",
            )

    # ---------------------------------------------------------
    # TOOL 3: RISK ATTRIBUTION
    # ---------------------------------------------------------
    elif active_nav == "Risk Attribution":
        base_idx = float(attribution.get("cohort_baseline_index", 0.0))
        final_idx = float(attribution.get("borrower_sequence_index", 0.0))
        traj_lift = float(attribution.get("trajectory_lift_pts", 0.0))
        total_delta = float(attribution.get("total_delta_pts", 0.0))

        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown(f"### Factor Attribution: `{selected_borrower}`")
            st.caption("Counterfactual marginal departure isolating temporal sequence dynamics from cohort median.")
        with h_col2:
            st.markdown(f"<div style='text-align:right;'><span class='compliance-pill'>Account: {selected_borrower}</span></div>", unsafe_allow_html=True)

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("Cohort Baseline", f"{base_idx:.1f} / 100", help="Evaluated risk index on cohort median feature baseline.")
        with a2:
            st.metric("Account Index", f"{final_idx:.1f} / 100", delta=f"{final_idx - base_idx:+.1f} pts from baseline")
        with a3:
            st.metric("Sequence Trajectory Lift", f"{traj_lift:+.1f} pts", help="Impact attributable strictly to order-aware features.")
        with a4:
            st.metric("Sequence vs Static Delta", f"{total_delta:+.1f} pts", help="Delta between order-aware score and static random forest score.")

        st.markdown("<hr>", unsafe_allow_html=True)

        c_left, c_right = st.columns([3, 2], gap="medium")
        with c_left:
            st.markdown("##### Factor Decomposition Waterfall")
            waterfall_data = attribution.get("waterfall", [])
            if waterfall_data:
                waterfall_opts = build_waterfall_echarts_options(waterfall_data)
                st_echarts(options=waterfall_opts, height="320px", key="attribution_waterfall")
            else:
                st.info("Waterfall decomposition is not available for this account.")
        with c_right:
            st.markdown("##### Dominant Feature Drivers")
            escalators = attribution.get("top_escalators", [])
            mitigators = attribution.get("top_mitigators", [])

            st.markdown("<small style='font-weight:600; color:#dc2626;'>TOP RISK ESCALATORS</small>", unsafe_allow_html=True)
            if escalators:
                for esc in escalators[:3]:
                    st.markdown(f"• **{esc['display_name']}**: `<span class='driver-badge-escalator'>+{esc['impact_points']:.2f} pts</span>` ({esc['direction']})", unsafe_allow_html=True)
            else:
                st.caption("No significant risk-increasing departures.")

            st.markdown("<div style='margin-top:8px;'><small style='font-weight:600; color:#16a34a;'>TOP RISK MITIGATORS</small></div>", unsafe_allow_html=True)
            if mitigators:
                for mit in mitigators[:3]:
                    st.markdown(f"• **{mit['display_name']}**: `<span class='driver-badge-mitigator'>{mit['impact_points']:.2f} pts</span>` ({mit['direction']})", unsafe_allow_html=True)
            else:
                st.caption("No significant risk-mitigating departures.")

            st.markdown(
                """
                <div class="summary-card" style="margin-top:12px;">
                    <strong>Order-Aware Velocity Note:</strong><br>
                    Sequence dynamics detect decaying balances and recent cash-out bursts even when aggregate mean balances appear stable.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # TOOL 4: STRESS & SCENARIOS
    # ---------------------------------------------------------
    elif active_nav == "Stress & Scenarios":
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown(f"### Stress Testing & What-If Sandbox: `{selected_borrower}`")
            st.caption("Simulate facility exposures, adverse cashflow shocks, and underwriting policy cutoff shifts.")
        with h_col2:
            st.markdown(f"<div style='text-align:right;'><span class='compliance-pill'>Account: {selected_borrower}</span></div>", unsafe_allow_html=True)

        sc_left, sc_right = st.columns([1, 2], gap="medium")
        with sc_left:
            st.markdown("##### Simulation Controls")
            tenure_choice = st.radio(
                "Facility Term:",
                options=[7, 14, 30],
                index=2,
                format_func=lambda t: f"{TENURE_CONFIGS[t]['name']} ({TENURE_CONFIGS[t]['flat_fee_rate']*100:.0f}% fee)",
                horizontal=True,
                key="sc_tenure_radio",
            )
            loan_slider = st.slider("Requested Principal (GHS):", min_value=25, max_value=1000, value=250, step=25, key="sc_loan_slider")

            with st.expander("Adverse Cashflow Shocks", expanded=False):
                shock_inc = st.slider("Inbound Transfer Drop (%):", 0, 50, 20, 5, key="sc_shock_inc")
                shock_fee = st.slider("Fee Burden Surge (%):", 0, 50, 30, 5, key="sc_shock_fee")
                shock_inact = st.slider("Inactivity Gap (Days):", 0, 30, 7, 1, key="sc_shock_inact")

        term_result = simulate_term_sizing(
            base_score=borrower_score["sequence_demo_score"],
            principal=loan_slider,
            tenure_days=tenure_choice,
            indicators=indicators,
        )
        amounts_range = list(range(25, 1025, 25))
        multi_curves = simulate_multi_term_curves(
            base_score=borrower_score["sequence_demo_score"],
            capacity=term_result.absorption_capacity,
            amounts_range=amounts_range,
        )
        seq_dict = borrower_detail.get("sequence_features", {})
        sta_dict = borrower_detail.get("static_features", {})
        stress_res = evaluate_borrower_stress(
            scorer=scorer,
            sequence_row=pd.Series(seq_dict),
            static_row=pd.Series(sta_dict),
            inbound_drop_pct=float(shock_inc) / 100.0 if "shock_inc" in locals() else 0.20,
            fee_surge_pct=float(shock_fee) / 100.0 if "shock_fee" in locals() else 0.30,
            inactivity_days=float(shock_inact) if "shock_inact" in locals() else 7.0,
        )

        with sc_right:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Repayment", format_money(term_result.total_repayment), delta=f"{format_money(term_result.daily_debt_service)} / day", delta_color="off")
            with m2:
                dstc_pct = term_result.dstc_ratio * 100
                st.metric("DSTC Ratio", f"{dstc_pct:.1f}%", delta="High Burden" if dstc_pct > 35 else "Manageable", delta_color="inverse" if dstc_pct > 35 else "normal")
            with m3:
                stressed_pts = round(stress_res["stressed"]["sequence_score"] * 100, 1)
                st.metric("Stressed Index", f"{stressed_pts} / 100", delta=f"{stress_res['delta_pts']:+.1f} pts shock", delta_color="inverse")

            tab_curve, tab_shock = st.tabs(["Facility Sizing Curves", "Adverse Shock Impact"])
            with tab_curve:
                loan_opts = build_loan_curve_echarts_options(
                    amounts_range,
                    multi_curves[tenure_choice],
                    loan_slider,
                    term_result.scenario_risk_index,
                    multi_curves=multi_curves,
                    selected_tenure=tenure_choice,
                )
                st_echarts(options=loan_opts, height="240px", key="sc_loan_curve")
            with tab_shock:
                stress_opts = build_stress_comparison_echarts_options(
                    baseline_score=stress_res["baseline"]["sequence_score"],
                    stressed_score=stress_res["stressed"]["sequence_score"],
                    baseline_band=stress_res["baseline"]["risk_band"],
                    stressed_band=stress_res["stressed"]["risk_band"],
                )
                st_echarts(options=stress_opts, height="240px", key="sc_stress_bar")

    # ---------------------------------------------------------
    # TOOL 5: ACTIVITY LEDGER
    # ---------------------------------------------------------
    elif active_nav == "Activity Ledger":
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown(f"### Activity & Liquidity Ledger: `{selected_borrower}`")
            st.caption("Chronological transaction audit, liquidity velocity, and event type composition.")
        with h_col2:
            st.markdown(f"<div style='text-align:right;'><span class='compliance-pill'>Audited Operations: {len(recent_txs)}</span></div>", unsafe_allow_html=True)

        l_left, l_right = st.columns([1, 1], gap="medium")
        with l_left:
            st.markdown("##### Event Type Breakdown")
            mix_data = [
                {"name": "Cash Out", "value": round(indicators.get("pct_cashout", 0) * 100, 1)},
                {"name": "Transfer", "value": round(indicators.get("pct_transfer", 0) * 100, 1)},
                {"name": "Debit", "value": round(indicators.get("pct_debit", 0) * 100, 1)},
                {"name": "Payment", "value": round(indicators.get("pct_payment", 0) * 100, 1)},
            ]
            mix_opts = build_event_mix_echarts_options(mix_data)
            st_echarts(options=mix_opts, height="200px", key="activity_mix_pie")

            st.markdown("##### Liquidity Trend")
            if not recent_txs.empty:
                chrono_txs = recent_txs.sort_values("timestamp")
                dates = [str(ts)[:16].replace("T", " ") for ts in chrono_txs["timestamp"]]
                amounts = [float(a) for a in chrono_txs["amount"]]
                balances = [float(b) for b in chrono_txs["balance_before"]]
                timeline_opts = build_timeline_echarts_options(dates, amounts, balances)
                st_echarts(options=timeline_opts, height="200px", key="activity_timeline")

        with l_right:
            st.markdown("##### Recent Audited Records")
            if not recent_txs.empty:
                tx_display = recent_txs.copy()
                if "timestamp" in tx_display.columns:
                    tx_display["timestamp"] = tx_display["timestamp"].astype(str).str[:16].str.replace("T", " ")
                tx_display = tx_display.rename(columns={
                    "timestamp": "Timestamp",
                    "transaction_type": "Type",
                    "amount": "Amount",
                    "balance_before": "Opening Balance",
                    "balance_after": "Closing Balance",
                    "recipient_id": "Counterparty",
                })
                show_cols = [c for c in ["Timestamp", "Type", "Amount", "Opening Balance", "Closing Balance", "Counterparty"] if c in tx_display.columns]
                st.dataframe(tx_display[show_cols], hide_index=True, width="stretch", height=430)
            else:
                st.info("No transaction events available for this account.")

    # ---------------------------------------------------------
    # TOOL 6: MODEL BENCHMARK
    # ---------------------------------------------------------
    elif active_nav == "Model Benchmark":
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown("### Model Governance & Empirical Validation")
            st.caption("Validation against preserved empirical research benchmarks and reliability curves.")
        with h_col2:
            st.markdown(f"<div style='text-align:right;'><span class='compliance-pill'>Version: {scorer.manifest['model_version']}</span></div>", unsafe_allow_html=True)

        bm1, bm2, bm3 = st.columns(3)
        with bm1:
            st.metric("Static Baseline AUC", "0.7286", delta="Ref: 0.721", delta_color="off", help="RandomForest evaluated on cumulative aggregate features.")
        with bm2:
            st.metric("Order-Aware Surrogate AUC", "0.7551", delta="Ref: 0.752", delta_color="off", help="HistGradientBoosting evaluated on order-aware temporal features.")
        with bm3:
            st.metric("Sequential Lift", "+0.0265 AUC", delta="Ref: +0.030", help="Discrimination advantage unlocked by temporal sequencing.")

        st.markdown("<hr>", unsafe_allow_html=True)

        b_left, b_right = st.columns([1, 1], gap="medium")
        with b_left:
            st.markdown("##### Discrimination Benchmark (AUC-ROC)")
            real_ref = scorer.manifest.get("real_reference", {})
            auc_opts = build_auc_comparison_echarts_options(
                float(real_ref.get("static_auc", 0.721)),
                float(real_ref.get("gru_auc", 0.752)),
                float(scorer.manifest["models"]["static"]["metrics"]["auc_roc"]),
                float(scorer.manifest["models"]["sequence"]["metrics"]["auc_roc"]),
            )
            st_echarts(options=auc_opts, height="260px", key="model_auc_bar")

            st.markdown(
                """
                <div class="summary-card">
                    <strong>Scope & Governance Boundaries:</strong><br>
                    • Sequential models consistently improve strict default discrimination over aggregate baselines.<br>
                    • Scores are educational 0–100 index points; they are not autonomous lending recommendations.<br>
                    • Zero data egress: all inference and calibrations execute entirely offline.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b_right:
            st.markdown("##### Reliability Calibration Diagram")
            calib_data = payload.get("calibration", {})
            if calib_data:
                calib_opts = build_calibration_echarts_options(calib_data)
                st_echarts(options=calib_opts, height="260px", key="model_calib_curve")

            brier_data = payload.get("calibration", {}).get("brier_decomposition", {})
            if brier_data:
                brier_rows = [
                    {"Model": "Static Baseline", "Total Brier": brier_data["static"]["total_brier"], "Reliability": brier_data["static"]["reliability"], "Resolution": brier_data["static"]["resolution"]},
                    {"Model": "Order-Aware Surrogate", "Total Brier": brier_data["sequence"]["total_brier"], "Reliability": brier_data["sequence"]["reliability"], "Resolution": brier_data["sequence"]["resolution"]},
                ]
                st.dataframe(pd.DataFrame(brier_rows), hide_index=True, width="stretch", height=100)


if __name__ == "__main__":
    main()
