"""Upload-first single-borrower SeqCredit report."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from seqcredit_mvp.dashboard import build_dashboard_payload
from seqcredit_mvp.reports import generate_borrower_memo_pdf
from seqcredit_mvp.scenarios import (
    DEFAULT_TENURES,
    evaluate_borrower_stress,
    simulate_multi_term_curves,
    simulate_term_sizing,
)
from seqcredit_mvp.scoring import DEFAULT_ARTIFACT_DIR, DemoScorer


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV_PATH = ROOT / "data" / "sample_transactions.csv"


def _render_echarts(options: dict, height: str, key: str) -> None:
    """Load ECharts only when a real Streamlit render needs it."""
    from streamlit_echarts import st_echarts

    st_echarts(options=options, height=height, key=key)


@st.cache_resource
def get_scorer() -> DemoScorer:
    return DemoScorer(DEFAULT_ARTIFACT_DIR)


@st.cache_data(show_spinner=False)
def analyse_csv(csv_bytes: bytes) -> dict:
    transactions = pd.read_csv(io.BytesIO(csv_bytes))
    if "borrower_id" not in transactions.columns:
        transactions["borrower_id"] = "PERSON_001"
    if "transaction_id" not in transactions.columns:
        transactions["transaction_id"] = [f"TXN_{i:08d}" for i in range(len(transactions))]
    return build_dashboard_payload(transactions, get_scorer())


def _score_pct(value: float) -> float:
    return round(float(value) * 100, 1)


def _money(value: float) -> str:
    return f"GHS {float(value):,.2f}"


def _band_style(band: str) -> tuple[str, str]:
    band = band.lower()
    if "elevated" in band:
        return "#fff1f2", "#be123c"
    if "watch" in band:
        return "#fffbeb", "#b45309"
    return "#ecfdf5", "#047857"


def _recommendation(band: str) -> tuple[str, str]:
    band = band.lower()
    if "elevated" in band:
        return "Pause and review", "Do not approve from this history alone. Verify income, existing debt, identity, and repayment capacity."
    if "watch" in band:
        return "Manual review recommended", "The history is mixed. Review affordability and repayment capacity before making a lending decision."
    return "Continue to normal underwriting", "The observed pattern is lower risk in this demo. Complete normal affordability and policy checks before approval."


def _story_step(number: str, title: str, text: str, active: bool = False) -> str:
    border = "#5267d9" if active else "#e5e7ef"
    title_color = "#334155" if active else "#64748b"
    return (
        f"<div style='border-top:3px solid {border};padding:.65rem .75rem .4rem;min-height:88px'>"
        f"<div style='font-size:.72rem;color:#64748b;font-weight:700'>STEP {number}</div>"
        f"<div style='font-weight:700;color:{title_color};margin:.2rem 0'>{title}</div>"
        f"<div style='font-size:.82rem;color:#64748b;line-height:1.35'>{text}</div></div>"
    )


def _metric(detail: dict, static_features: dict, key: str, default: float = 0.0) -> float:
    return float(detail.get("indicators", {}).get(key, static_features.get(key, default)))


def _bar_options(labels: list[str], values: list[float], colors: list[str] | None = None, suffix: str = "") -> dict:
    colors = colors or ["#5267d9"] * len(labels)
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "4%", "right": "5%", "top": "8%", "bottom": "12%", "containLabel": True},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": "#64748b", "fontSize": 10, "interval": 0}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#64748b", "fontSize": 10, "formatter": f"{{value}}{suffix}"}, "splitLine": {"lineStyle": {"color": "#eef0f6"}}},
        "series": [{"type": "bar", "barWidth": "48%", "data": [{"value": round(v, 2), "itemStyle": {"color": c, "borderRadius": [4, 4, 0, 0]}} for v, c in zip(values, colors)], "label": {"show": True, "position": "top", "fontSize": 10, "color": "#334155", "formatter": f"{{c}}{suffix}"}}],
    }


def _timeline_options(transactions: pd.DataFrame) -> dict:
    tx = transactions.sort_values("timestamp")
    dates = [str(v)[:16].replace("T", " ") for v in tx["timestamp"]]
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Amount", "Balance before"], "top": 0, "textStyle": {"color": "#64748b", "fontSize": 10}},
        "grid": {"left": "3%", "right": "4%", "top": "16%", "bottom": "14%", "containLabel": True},
        "xAxis": {"type": "category", "data": dates, "axisLabel": {"color": "#64748b", "fontSize": 9}},
        "yAxis": {"type": "value", "axisLabel": {"color": "#64748b", "fontSize": 9, "formatter": "GHS {value}"}, "splitLine": {"lineStyle": {"color": "#eef0f6"}}},
        "dataZoom": [{"type": "inside", "start": 0, "end": 100}],
        "series": [
            {"name": "Amount", "type": "line", "smooth": True, "data": tx["amount"].round(2).tolist(), "itemStyle": {"color": "#5267d9"}, "areaStyle": {"opacity": 0.08, "color": "#5267d9"}},
            {"name": "Balance before", "type": "line", "smooth": True, "data": tx["balance_before"].round(2).tolist(), "itemStyle": {"color": "#94a3b8"}, "lineStyle": {"type": "dashed"}},
        ],
    }


def _waterfall_options(waterfall: dict) -> dict:
    steps = waterfall.get("steps", [])
    labels = ["Cohort baseline"] + [str(s["name"]) for s in steps] + ["Final index"]
    values = [float(waterfall.get("baseline_index", 0))] + [float(s.get("impact", 0)) for s in steps] + [float(waterfall.get("final_index", 0))]
    colors = ["#94a3b8"] + ["#dc6470" if v >= 0 else "#25a58a" for v in values[1:-1]] + ["#5267d9"]
    return _bar_options(labels, values, colors, " pts")


def _loan_curve_options(amounts: list[int], curves: dict[int, list[float]], selected_amount: int, selected_tenure: int) -> dict:
    colors = {7: "#dc6470", 14: "#d97706", 30: "#5267d9"}
    series = []
    for tenure in DEFAULT_TENURES:
        if tenure not in curves:
            continue
        values = [_score_pct(v) for v in curves[tenure]]
        item = {"name": f"{tenure}-day", "type": "line", "smooth": True, "data": values, "itemStyle": {"color": colors[tenure]}, "lineStyle": {"width": 3 if tenure == selected_tenure else 1.5, "type": "solid" if tenure == selected_tenure else "dashed"}}
        if tenure == selected_tenure:
            idx = amounts.index(selected_amount)
            item["markPoint"] = {"data": [{"coord": [str(selected_amount), values[idx]], "value": f"{values[idx]} / 100", "itemStyle": {"color": colors[tenure]}}]}
        series.append(item)
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": [f"{t}-day" for t in DEFAULT_TENURES], "top": 0, "textStyle": {"color": "#64748b", "fontSize": 10}},
        "grid": {"left": "4%", "right": "5%", "top": "15%", "bottom": "14%", "containLabel": True},
        "xAxis": {"type": "category", "data": [str(v) for v in amounts], "name": "Requested amount (GHS)", "nameLocation": "middle", "nameGap": 28, "axisLabel": {"color": "#64748b", "fontSize": 9}},
        "yAxis": {"type": "value", "min": 0, "max": 100, "name": "Scenario risk index", "axisLabel": {"color": "#64748b", "fontSize": 9}, "splitLine": {"lineStyle": {"color": "#eef0f6"}}},
        "series": series,
    }


def _stress_options(base: float, stressed: float) -> dict:
    return _bar_options(["Baseline", "Adverse shock"], [_score_pct(base), _score_pct(stressed)], ["#5267d9", "#dc6470"])


def _show_summary(payload: dict, borrower_id: str, score: pd.Series, detail: dict, scorer: DemoScorer) -> None:
    st.markdown("#### Start with the answer")
    st.caption("This first view gives the screening outcome. The next tabs show the evidence behind it and test whether a particular facility fits the borrower’s observed cashflow.")
    band = str(score["demo_risk_band"])
    title, explanation = _recommendation(band)
    bg, fg = _band_style(band)
    static_features = detail.get("static_features", {})
    st.markdown(f"<div style='background:{bg};border-left:5px solid {fg};border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem'><div style='color:{fg};font-size:1.25rem;font-weight:750'>{title}</div><div style='margin-top:.35rem'>{explanation}</div></div>", unsafe_allow_html=True)
    st.caption("This is a screening recommendation from a synthetic demonstration model, not an automatic approval or decline.")
    cols = st.columns(4)
    cols[0].metric("Pattern risk index", f"{_score_pct(score['sequence_demo_score']):.1f} / 100")
    cols[1].metric("Static baseline", f"{_score_pct(score['static_demo_score']):.1f} / 100", delta=f"{_score_pct(score['sequence_demo_score'] - score['static_demo_score']):+.1f} pts order effect", delta_color="off")
    cols[2].metric("Transactions", f"{int(_metric(detail, static_features, 'transaction_count')):,}")
    cols[3].metric("Average balance", _money(_metric(detail, static_features, "avg_balance")))
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### What the history says")
        st.write(f"The file contains **{int(_metric(detail, static_features, 'transaction_count')):,}** transactions across **{int(_metric(detail, static_features, 'unique_recipients')):,}** counterparties.")
        st.write(f"Average transaction size is **{_money(_metric(detail, static_features, 'avg_amount'))}**; the lowest observed balance is **{_money(_metric(detail, static_features, 'min_balance'))}**.")
        st.write(f"Cash-outs represent **{_metric(detail, static_features, 'pct_cashout') * 100:.1f}%** of activity and low-balance events represent **{_metric(detail, static_features, 'pct_low_balance') * 100:.1f}%**.")
    with right:
        st.markdown("#### Core signals")
        rows = [
            {"Signal": "Cash-out frequency", "Value": f"{_metric(detail, static_features, 'pct_cashout') * 100:.1f}%"},
            {"Signal": "Off-hours activity", "Value": f"{_metric(detail, static_features, 'pct_night') * 100:.1f}%"},
            {"Signal": "Low-balance frequency", "Value": f"{_metric(detail, static_features, 'pct_low_balance') * 100:.1f}%"},
            {"Signal": "Total turnover", "Value": _money(_metric(detail, static_features, 'total_volume'))},
            {"Signal": "Fees paid", "Value": _money(_metric(detail, static_features, 'total_fees'))},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button("Download borrower credit memo (PDF)", data=generate_borrower_memo_pdf(borrower_id, score.to_dict(), detail, scorer.manifest), file_name=f"seqcredit_memo_{borrower_id}.pdf", mime="application/pdf")


def _show_behaviour(detail: dict, tx: pd.DataFrame) -> None:
    st.markdown("#### Step 2 · Follow the money")
    st.caption("Before interpreting the score, look at how money enters, leaves, and accumulates. Withdrawal concentration, low-balance events, timing, and balance movement provide the behavioral context.")
    static_features = detail.get("static_features", {})
    mix = [("Transfer", _metric(detail, static_features, "pct_transfer") * 100), ("Debit", _metric(detail, static_features, "pct_debit") * 100), ("Payment", _metric(detail, static_features, "pct_payment") * 100), ("Cash-out", _metric(detail, static_features, "pct_cashout") * 100)]
    activity = [("Cash-out", _metric(detail, static_features, "pct_cashout") * 100), ("Off-hours", _metric(detail, static_features, "pct_night") * 100), ("Low balance", _metric(detail, static_features, "pct_low_balance") * 100), ("Recent cash-out", _metric(detail, detail.get("sequence_features", {}), "recent_cashout_rate") * 100)]
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Transaction type mix")
        _render_echarts(_bar_options([x[0] for x in mix], [x[1] for x in mix], ["#5267d9", "#7c8be4", "#9da9ec", "#dc6470"], "%"), "270px", "behaviour_mix")
    with right:
        st.markdown("#### Behavioural frequencies")
        _render_echarts(_bar_options([x[0] for x in activity], [x[1] for x in activity], ["#dc6470", "#d97706", "#b45309", "#5267d9"], "%"), "270px", "behaviour_frequency")
    st.markdown("#### Activity and liquidity over time")
    if not tx.empty:
        _render_echarts(_timeline_options(tx), "320px", "behaviour_timeline")
    st.markdown("#### Recent transaction ledger")
    recent = tx.sort_values("timestamp", ascending=False).head(20).copy()
    if not recent.empty:
        recent["timestamp"] = recent["timestamp"].astype(str).str[:16].str.replace("T", " ")
        recent = recent.rename(columns={"timestamp": "Timestamp", "transaction_type": "Type", "amount": "Amount", "balance_before": "Opening balance", "balance_after": "Closing balance", "recipient_id": "Counterparty"})
        columns = [c for c in ["Timestamp", "Type", "Amount", "Opening balance", "Closing balance", "Counterparty"] if c in recent.columns]
        st.dataframe(recent[columns], hide_index=True, width="stretch", height=300)


def _show_why(detail: dict) -> None:
    st.markdown("#### Step 3 · Connect behaviour to the result")
    st.caption("The waterfall compares this borrower with the cohort reference. Red contributions push the index higher; green contributions pull it lower.")
    attribution = detail.get("attribution", {})
    cols = st.columns(4)
    cols[0].metric("Cohort baseline", f"{attribution.get('cohort_baseline_index', 0):.1f} / 100")
    cols[1].metric("Borrower index", f"{attribution.get('borrower_sequence_index', 0):.1f} / 100")
    cols[2].metric("Order-aware lift", f"{attribution.get('trajectory_lift_pts', 0):+.1f} pts")
    cols[3].metric("Sequence vs static", f"{attribution.get('total_delta_pts', 0):+.1f} pts")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("#### What moved the score")
        waterfall = attribution.get("waterfall", {})
        if waterfall.get("steps"):
            _render_echarts(_waterfall_options(waterfall), "360px", "why_waterfall")
    with right:
        st.markdown("#### Main risk escalators")
        for item in attribution.get("top_escalators", []):
            st.markdown(f"**{item['display_name']}**  `+{item['impact_points']:.2f} pts`")
            st.caption(f"Observed {item['actual_value']} vs cohort reference {item['baseline_value']}. {item['description']}.")
        if not attribution.get("top_escalators"):
            st.caption("No material risk escalators detected.")
        st.markdown("#### Main mitigators")
        for item in attribution.get("top_mitigators", []):
            st.markdown(f"**{item['display_name']}**  `{item['impact_points']:.2f} pts`")
            st.caption(f"Observed {item['actual_value']} vs cohort reference {item['baseline_value']}. {item['description']}.")
        if not attribution.get("top_mitigators"):
            st.caption("No material risk mitigators detected.")
    st.info("Attribution shows which model inputs moved this score relative to the cohort baseline. It does not prove that any single behavior causes repayment problems.")


def _show_scenarios(detail: dict, score: pd.Series, scorer: DemoScorer) -> None:
    st.markdown("#### Step 4 · Test the facility")
    st.caption("A lower-risk history may still struggle with a loan that is too large or too short. Move the amount and term to see how repayment pressure changes.")
    indicators = detail.get("indicators", {})
    static_features = detail.get("static_features", {})
    merged = {**static_features, **indicators}
    tenure = st.radio("Repayment term", options=[7, 14, 30], format_func=lambda v: f"{v} days", horizontal=True, key="scenario_tenure")
    amount = st.slider("Requested loan amount (GHS)", min_value=25, max_value=2000, value=250, step=25, key="scenario_amount")
    result = simulate_term_sizing(float(score["sequence_demo_score"]), amount, tenure, merged)
    amounts = list(range(25, 2025, 25))
    curves = simulate_multi_term_curves(float(score["sequence_demo_score"]), result.absorption_capacity, amounts)
    _render_echarts(_loan_curve_options(amounts, curves, amount, tenure), "360px", "scenario_curves")
    cards = st.columns(4)
    cards[0].metric("Total repayment", _money(result.total_repayment))
    cards[1].metric("Daily debt service", _money(result.daily_debt_service))
    cards[2].metric("Debt service / cashflow", f"{result.dstc_ratio * 100:.1f}%")
    cards[3].metric("Scenario risk", f"{_score_pct(result.scenario_risk_index):.1f} / 100")
    st.info(result.risk_assessment)
    st.caption(f"Estimated daily cashflow: {_money(result.estimated_daily_cashflow)} · Estimated absorption capacity: {_money(result.absorption_capacity)}")
    with st.expander("Test adverse cashflow conditions"):
        c1, c2, c3 = st.columns(3)
        inbound_drop = c1.slider("Inbound cashflow drop", 0, 50, 20, 5, format="%d%%", key="stress_inbound")
        fee_surge = c2.slider("Fee burden increase", 0, 50, 30, 5, format="%d%%", key="stress_fee")
        inactivity = c3.slider("Added inactivity gap", 0, 30, 7, 1, format="%d days", key="stress_inactivity")
        stress = evaluate_borrower_stress(scorer, pd.Series(detail["sequence_features"]), pd.Series(detail["static_features"]), inbound_drop / 100, fee_surge / 100, inactivity)
        _render_echarts(_stress_options(stress["baseline"]["sequence_score"], stress["stressed"]["sequence_score"]), "250px", "scenario_stress")
        st.caption(f"Stress change: {stress['delta_pts']:+.1f} points · {stress['baseline']['risk_band']} → {stress['stressed']['risk_band']}")


def _show_report(payload: dict, source_name: str, scorer: DemoScorer) -> None:
    scores = pd.DataFrame(payload["scores"])
    borrower_ids = scores["borrower_id"].astype(str).tolist()
    if len(borrower_ids) > 1:
        default_id = str(scores.sort_values("sequence_demo_score").iloc[0]["borrower_id"])
        default_index = borrower_ids.index(default_id)
        borrower_id = st.selectbox("Choose borrower to review", borrower_ids, index=default_index, key="report_borrower_v2")
        if source_name == "sample_transactions.csv":
            st.caption("The sample opens on a lower-risk illustrative borrower. Use the selector to explore the other profiles.")
    else:
        borrower_id = borrower_ids[0]
    score = scores[scores["borrower_id"].astype(str) == borrower_id].iloc[0]
    detail = payload["details"][borrower_id]
    tx = st.session_state.get("uploaded_transactions", pd.DataFrame()).copy()
    if not tx.empty:
        if "borrower_id" not in tx.columns:
            tx["borrower_id"] = borrower_id
        tx["borrower_id"] = tx["borrower_id"].astype(str)
        tx = tx[tx["borrower_id"] == borrower_id].copy()
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    else:
        tx = pd.DataFrame(detail.get("recent_transactions", []))
    st.success(f"Report ready · {source_name}")
    st.subheader(f"Borrower report: {borrower_id}")
    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin:0 0 1.25rem'>"
        + _story_step("1", "Read the history", "What does money movement look like?", True)
        + _story_step("2", "Check stability", "Is liquidity holding up over time?")
        + _story_step("3", "Explain the score", "Which behaviors moved the result?")
        + _story_step("4", "Test the loan", "What happens at different amounts?")
        + "</div>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["1 · Decision summary", "2 · Behaviour", "3 · Why this result", "4 · Loan scenarios"])
    with tabs[0]:
        _show_summary(payload, borrower_id, score, detail, scorer)
    with tabs[1]:
        _show_behaviour(detail, tx)
    with tabs[2]:
        _show_why(detail)
    with tabs[3]:
        _show_scenarios(detail, score, scorer)


def main() -> None:
    st.set_page_config(page_title="SeqCredit · Borrower report", page_icon="📊", layout="wide")
    st.markdown("""<style>
    .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { background:#fff; border:1px solid #e5e7ef; border-radius:12px; padding:.8rem 1rem; }
    [data-testid="stFileUploader"] { border:1px dashed #8790c9; border-radius:14px; padding:.5rem; background:#fafaff; }
    </style>""", unsafe_allow_html=True)
    st.title("SeqCredit borrower report")
    st.write("Upload one person’s transaction history and understand the behaviour, score drivers, and loan-size risk behind the screening result.")
    st.info("This is a synthetic demonstration model. It provides screening support, not an automatic lending decision.", icon="ℹ️")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"], help="One row per transaction. Required fields: timestamp, transaction type, amount, balances, fee, and recipient ID.")
    use_sample = st.button("Try the sample file", type="secondary")
    csv_bytes: bytes | None = None
    source_name = ""
    if uploaded is not None:
        csv_bytes = uploaded.getvalue()
        source_name = uploaded.name
        st.session_state.uploaded_transactions = pd.read_csv(io.BytesIO(csv_bytes))
    elif use_sample:
        csv_bytes = SAMPLE_CSV_PATH.read_bytes()
        source_name = "sample_transactions.csv"
        st.session_state.uploaded_transactions = pd.read_csv(io.BytesIO(csv_bytes))
    if csv_bytes is None:
        st.markdown("#### Upload one person’s history to begin")
        st.caption("The report will examine transaction mix, withdrawal behaviour, liquidity, sequence trends, risk drivers, and loan-size scenarios.")
        return
    with st.spinner("Building borrower report…"):
        try:
            payload = analyse_csv(csv_bytes)
            scorer = get_scorer()
        except Exception as exc:
            st.error(f"We could not analyse that file: {exc}")
            st.caption("Check the CSV columns and values, then try again.")
            return
    _show_report(payload, source_name, scorer)


if __name__ == "__main__":
    main()
