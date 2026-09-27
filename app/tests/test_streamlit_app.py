from pathlib import Path
import pandas as pd
import pytest

from seqcredit_mvp.scoring import DemoScorer
from seqcredit_mvp.streamlit_app import (
    build_auc_comparison_echarts_options,
    build_batch_distribution_echarts_options,
    build_event_mix_echarts_options,
    build_gauge_echarts_options,
    build_loan_curve_echarts_options,
    build_stress_comparison_echarts_options,
    build_timeline_echarts_options,
    comparison_phrase,
    format_money,
    process_data,
    scenario_index,
    trend_label,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV_PATH = ROOT / "data" / "sample_transactions.csv"


def test_streamlit_helpers():
    assert scenario_index(0.5, 250, 1000) > 0
    assert scenario_index(0.01, 10, 1000) < scenario_index(0.99, 1000, 100)
    assert trend_label(0.1) == "Increasing Trajectory"
    assert trend_label(-0.1) == "Contracting Trajectory"
    assert trend_label(0.01) == "Stable Trajectory"
    assert comparison_phrase(0.8) == "below baseline historical averages"
    assert comparison_phrase(1.2) == "above baseline historical averages"
    assert comparison_phrase(1.0) == "broadly consistent with historical baseline"
    assert format_money(1234.5) == "GHS 1,234.50"


def test_streamlit_process_data():
    if not SAMPLE_CSV_PATH.exists():
        pytest.skip("sample_transactions.csv does not exist")
    scorer = DemoScorer()
    csv_bytes = SAMPLE_CSV_PATH.read_bytes()
    payload = process_data(csv_bytes, scorer)
    assert "scores" in payload
    assert "portfolio" in payload
    assert len(payload["scores"]) > 0


def test_timeline_echarts_options():
    dates = ["2026-01-01 10:00", "2026-01-02 12:00"]
    amounts = [100.0, 150.0]
    balances = [500.0, 400.0]
    opts = build_timeline_echarts_options(dates, amounts, balances)
    assert "series" in opts
    assert len(opts["series"]) == 2
    assert opts["xAxis"]["data"] == dates
    assert "dataZoom" in opts
    assert "toolbox" in opts


def test_loan_curve_echarts_options():
    amounts = [25, 50, 75]
    risks = [0.1, 0.2, 0.3]
    opts = build_loan_curve_echarts_options(amounts, risks, 50, 0.2)
    assert "series" in opts
    assert len(opts["series"]) == 1
    assert "markPoint" in opts["series"][0]
    assert opts["series"][0]["markPoint"]["data"][0]["coord"] == ["50", 0.2]


def test_gauge_and_event_mix_options():
    gauge_opts = build_gauge_echarts_options(45.2)
    assert gauge_opts["series"][0]["type"] == "gauge"
    assert gauge_opts["series"][0]["data"][0]["value"] == 45

    mix_data = [{"name": "Transfer", "value": 50.0}, {"name": "Debit", "value": 50.0}]
    mix_opts = build_event_mix_echarts_options(mix_data)
    assert mix_opts["series"][0]["type"] == "pie"
    assert mix_opts["series"][0]["data"] == mix_data


def test_distribution_and_auc_options():
    dist_opts = build_batch_distribution_echarts_options(["0-4", "5-9"], [10, 20])
    assert dist_opts["series"][0]["type"] == "bar"

    auc_opts = build_auc_comparison_echarts_options(0.72, 0.75, 0.71, 0.74)
    assert len(auc_opts["series"]) == 2
    assert auc_opts["xAxis"]["min"] == 0.68
    assert auc_opts["xAxis"]["max"] == 0.78


def test_multi_term_loan_curve_echarts_options():
    amounts = [25, 50, 75]
    risks = [0.1, 0.2, 0.3]
    multi = {
        7: [0.15, 0.25, 0.35],
        14: [0.12, 0.22, 0.32],
        30: [0.10, 0.20, 0.30],
    }
    opts = build_loan_curve_echarts_options(amounts, risks, 50, 0.22, multi_curves=multi, selected_tenure=14)
    assert "series" in opts
    assert len(opts["series"]) == 3
    assert "legend" in opts
    assert len(opts["legend"]["data"]) == 3
    # 14D series should have the markPoint
    assert "markPoint" in opts["series"][1]
    assert opts["series"][1]["markPoint"]["data"][0]["coord"] == ["50", 0.22]


def test_stress_comparison_echarts_options():
    opts = build_stress_comparison_echarts_options(
        baseline_score=0.03,
        stressed_score=0.08,
        baseline_band="Lower demo risk",
        stressed_band="Watch demo risk",
    )
    assert "series" in opts
    assert opts["series"][0]["type"] == "bar"
    assert len(opts["series"][0]["data"]) == 2
    assert opts["series"][0]["data"][0]["value"] == 3.0
    assert opts["series"][0]["data"][1]["value"] == 8.0


def test_streamlit_reports_export():
    if not SAMPLE_CSV_PATH.exists():
        pytest.skip("sample_transactions.csv does not exist")
    scorer = DemoScorer()
    csv_bytes = SAMPLE_CSV_PATH.read_bytes()
    payload = process_data(csv_bytes, scorer)
    from seqcredit_mvp.reports import (
        generate_borrower_memo_pdf,
        generate_borrower_memo_html,
        generate_portfolio_summary_pdf,
        generate_portfolio_summary_html,
    )
    b_id = str(payload["scores"][0]["borrower_id"])
    b_score = payload["scores"][0]
    b_detail = payload["details"][b_id]

    pdf_memo = generate_borrower_memo_pdf(b_id, b_score, b_detail, scorer.manifest)
    assert pdf_memo.startswith(b"%PDF-")

    html_memo = generate_borrower_memo_html(b_id, b_score, b_detail, scorer.manifest)
    assert "@media print" in html_memo

    pdf_port = generate_portfolio_summary_pdf(payload, scorer.manifest)
    assert pdf_port.startswith(b"%PDF-")

    html_port = generate_portfolio_summary_html(payload, scorer.manifest)
    assert "Portfolio Batch Risk" in html_port


