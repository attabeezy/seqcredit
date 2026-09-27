from pathlib import Path
import pandas as pd
import pytest

from seqcredit_mvp.dashboard import build_dashboard_payload
from seqcredit_mvp.reports import (
    build_governance_addendum,
    generate_borrower_memo_html,
    generate_borrower_memo_pdf,
    generate_portfolio_summary_html,
    generate_portfolio_summary_pdf,
)
from seqcredit_mvp.scoring import DemoScorer

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV_PATH = ROOT / "data" / "sample_transactions.csv"


@pytest.fixture(scope="module")
def scorer():
    return DemoScorer()


@pytest.fixture(scope="module")
def payload(scorer):
    txs = pd.read_csv(SAMPLE_CSV_PATH)
    return build_dashboard_payload(txs, scorer)


def test_build_governance_addendum(scorer):
    gov = build_governance_addendum(scorer.manifest)
    assert "model_version" in gov
    assert "status" in gov
    assert gov["status"] == "PASSED"
    assert "static_sha256" in gov
    assert "sequence_sha256" in gov
    assert len(gov["static_sha256"]) == 64
    assert len(gov["sequence_sha256"]) == 64
    assert "checks" in gov
    assert "default_rate" in gov["checks"]
    assert "static_auc" in gov["checks"]
    assert "sequence_auc" in gov["checks"]
    assert "sequence_gain" in gov["checks"]
    assert gov["checks"]["default_rate"]["status"] == "PASSED"
    assert "regulatory_disclaimer" in gov
    assert "STATUTORY" in gov["regulatory_disclaimer"]


def test_generate_borrower_memo_html(scorer, payload):
    b_id = str(payload["scores"][0]["borrower_id"])
    b_score = payload["scores"][0]
    b_detail = payload["details"][b_id]

    html_memo = generate_borrower_memo_html(
        borrower_id=b_id,
        borrower_score=b_score,
        borrower_detail=b_detail,
        manifest=scorer.manifest,
        scenario_result={
            "principal": 250,
            "tenure_days": 30,
            "total_repayment": 285.0,
            "total_fee": 35.0,
            "daily_debt_service": 9.5,
            "estimated_daily_cashflow": 25.0,
            "dstc_ratio": 0.38,
            "risk_assessment": "Well-Supported Liquidity",
        },
        stress_result={
            "baseline": {"sequence_score": 0.05, "risk_band": "Lower demo risk"},
            "stressed": {"sequence_score": 0.09, "risk_band": "Watch demo risk"},
            "delta_pts": 4.0,
            "band_shifted": True,
        },
    )

    assert "<!doctype html>" in html_memo
    assert "@media print" in html_memo
    assert b_id in html_memo
    assert "Institutional Credit Risk Memorandum" in html_memo
    assert "Facility Sizing &amp; Liquidity Absorption" in html_memo or "Facility Sizing & Liquidity Absorption" in html_memo
    assert "Counterfactual Cashflow Shock Resilience" in html_memo
    assert "Model Lineage &amp; Cryptographic Governance Addendum" in html_memo or "Model Lineage & Cryptographic Governance Addendum" in html_memo
    assert scorer.manifest["models"]["static"]["sha256"][:16] in html_memo


def test_generate_borrower_memo_pdf(scorer, payload):
    b_id = str(payload["scores"][0]["borrower_id"])
    b_score = payload["scores"][0]
    b_detail = payload["details"][b_id]

    pdf_bytes = generate_borrower_memo_pdf(
        borrower_id=b_id,
        borrower_score=b_score,
        borrower_detail=b_detail,
        manifest=scorer.manifest,
        scenario_result={
            "principal": 300,
            "tenure_days": 14,
            "total_repayment": 324.0,
            "total_fee": 24.0,
            "daily_debt_service": 23.14,
            "estimated_daily_cashflow": 30.0,
            "dstc_ratio": 0.77,
            "risk_assessment": "Elevated Debt Service Pressure",
        },
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes[-1024:]


def test_generate_portfolio_summary_html(scorer, payload):
    html_summary = generate_portfolio_summary_html(payload, scorer.manifest)

    assert "<!doctype html>" in html_summary
    assert "@media print" in html_summary
    assert "Portfolio Batch Risk &amp; Sequence Migration Report" in html_summary or "Portfolio Batch Risk & Sequence Migration Report" in html_summary
    assert "3×3 Risk Band Migration Matrix" in html_summary or "3x3 Risk Band Migration Matrix" in html_summary
    assert "Score Range" in html_summary
    assert scorer.manifest["models"]["sequence"]["sha256"][:16] in html_summary


def test_generate_portfolio_summary_pdf(scorer, payload):
    pdf_bytes = generate_portfolio_summary_pdf(payload, scorer.manifest)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes[-1024:]
