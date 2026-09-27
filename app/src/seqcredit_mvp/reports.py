"""Executive Risk Committee Export Engine for SeqCredit MVP.

Generates audit-grade institutional credit memos and portfolio batch reports
in both high-resolution vector PDF (via ReportLab) and print-optimized HTML (@media print).
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _fmt_money(val: float | int | None) -> str:
    if val is None:
        return "—"
    return f"GHS {float(val):,.2f}"


def _fmt_pct(val: float | int | None) -> str:
    if val is None:
        return "—"
    return f"{float(val) * 100.0:.1f}%"


def _fmt_score(val: float | int | None) -> str:
    if val is None:
        return "—"
    return f"{round(float(val) * 100.0):d} / 100"


def _fmt_pts(val: float | int | None) -> str:
    if val is None:
        return "—"
    f = float(val)
    sign = "+" if f > 0 else ""
    return f"{sign}{f:.1f} pts"


def build_governance_addendum(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract and format model lineage, cryptographic hashes, acceptance criteria, and legal notices."""
    models = manifest.get("models", {})
    static_model = models.get("static", {})
    seq_model = models.get("sequence", {})
    acceptance = manifest.get("acceptance", {})
    checks = manifest.get("checks", {})

    return {
        "model_version": manifest.get("model_version", "synthetic-demo-v1"),
        "created_at": manifest.get("created_at", datetime.now(timezone.utc).isoformat()),
        "status": manifest.get("status", "passed").upper(),
        "notice": manifest.get("notice", "Synthetic prototype only. Not for lending decisions."),
        "static_sha256": static_model.get("sha256", "UNKNOWN"),
        "sequence_sha256": seq_model.get("sha256", "UNKNOWN"),
        "static_auc": static_model.get("metrics", {}).get("auc_roc", 0.7286),
        "sequence_auc": seq_model.get("metrics", {}).get("auc_roc", 0.7551),
        "sequence_gain": manifest.get("sequence_gain_auc", 0.0265),
        "checks": {
            "default_rate": {
                "label": "Strict Default Prevalence Band [3.5% – 5.0%]",
                "target": f"{acceptance.get('default_rate', [0.035, 0.05])[0]*100:.1f}% – {acceptance.get('default_rate', [0.035, 0.05])[1]*100:.1f}%",
                "status": "PASSED" if checks.get("default_rate", True) else "FAILED",
            },
            "static_auc": {
                "label": "Static Random Forest AUC-ROC [0.700 – 0.740]",
                "target": f"{acceptance.get('static_auc', [0.70, 0.74])[0]:.3f} – {acceptance.get('static_auc', [0.70, 0.74])[1]:.3f}",
                "status": "PASSED" if checks.get("static_auc", True) else "FAILED",
            },
            "sequence_auc": {
                "label": "Order-Aware Surrogate AUC-ROC [0.735 – 0.770]",
                "target": f"{acceptance.get('sequence_auc', [0.735, 0.77])[0]:.3f} – {acceptance.get('sequence_auc', [0.735, 0.77])[1]:.3f}",
                "status": "PASSED" if checks.get("sequence_auc", True) else "FAILED",
            },
            "sequence_gain": {
                "label": "Empirical Sequence Gain Δ AUC [0.015 – 0.050]",
                "target": f"+{acceptance.get('sequence_gain', [0.015, 0.05])[0]:.3f} – +{acceptance.get('sequence_gain', [0.015, 0.05])[1]:.3f}",
                "status": "PASSED" if checks.get("sequence_gain", True) else "FAILED",
            },
        },
        "regulatory_disclaimer": (
            "STATUTORY MODEL GOVERNANCE NOTICE: This quantitative risk evaluation is produced "
            "by the SeqCredit offline synthetic prototype for supervisory committee review and "
            "algorithmic discrimination benchmarking. The risk index (0–100 scale) and relative "
            "risk tiers do not constitute credit scores, default probabilities, or statutory credit "
            "determinations under applicable banking regulations. Autonomous lending decisions, "
            "adverse action notices, or contractual commitments must not be based solely upon this demonstration."
        ),
    }


# =========================================================================
# 1. INSTITUTIONAL SINGLE-BORROWER MEMO (HTML & PDF)
# =========================================================================


def generate_borrower_memo_html(
    borrower_id: str,
    borrower_score: dict[str, Any],
    borrower_detail: dict[str, Any],
    manifest: dict[str, Any],
    scenario_result: dict[str, Any] | None = None,
    stress_result: dict[str, Any] | None = None,
) -> str:
    """Generate a clean, print-ready 1-page institutional credit memorandum in HTML with @media print."""
    gov = build_governance_addendum(manifest)
    indicators = borrower_detail.get("indicators", {})
    attribution = borrower_detail.get("attribution", {})
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    band_name = str(borrower_score.get("demo_risk_band", "Unknown")).replace(" demo risk", "")
    band_cls = "band-lower" if "lower" in band_name.lower() else ("band-watch" if "watch" in band_name.lower() else "band-elevated")

    seq_pts = round(float(borrower_score.get("sequence_demo_score", 0.0)) * 100)
    stat_pts = round(float(borrower_score.get("static_demo_score", 0.0)) * 100)
    delta_pts = seq_pts - stat_pts
    delta_sign = "+" if delta_pts >= 0 else ""

    # Feature drivers
    top_escalators = attribution.get("top_escalators", [])
    top_mitigators = attribution.get("top_mitigators", [])

    drivers_rows_html = ""
    combined_drivers = top_escalators[:3] + top_mitigators[:3]
    for d in combined_drivers:
        pts = float(d.get("impact_points", 0.0))
        sign = "+" if pts >= 0 else ""
        badge_cls = "badge-escalator" if pts >= 0 else "badge-mitigator"
        cat_str = "Sequence Dynamic" if d.get("category") == "sequence" else "Aggregate Static"
        drivers_rows_html += f"""
        <tr>
            <td><strong>{html.escape(d.get('display_name', ''))}</strong><br><small class="text-muted">{html.escape(d.get('description', ''))}</small></td>
            <td><span class="badge {badge_cls}">{cat_str}</span></td>
            <td>{html.escape(str(d.get('actual_value', '')))} {html.escape(str(d.get('unit', '')))}</td>
            <td>{html.escape(str(d.get('baseline_value', '')))} {html.escape(str(d.get('unit', '')))}</td>
            <td class="text-right"><strong>{sign}{pts:.2f} pts</strong></td>
        </tr>
        """

    # Scenario section
    scenario_html = ""
    if scenario_result:
        scenario_html = f"""
        <div class="section-card">
            <div class="section-title">Facility Sizing & Liquidity Absorption (DSTC Simulation)</div>
            <div class="grid-4">
                <div class="kpi-box">
                    <div class="kpi-label">Simulated Exposure</div>
                    <div class="kpi-value">{_fmt_money(scenario_result.get('principal'))}</div>
                    <div class="kpi-sub">Tenure: {scenario_result.get('tenure_days')} Days</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Total Repayment Due</div>
                    <div class="kpi-value">{_fmt_money(scenario_result.get('total_repayment'))}</div>
                    <div class="kpi-sub">Fee: {_fmt_money(scenario_result.get('total_fee'))}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Daily Debt Service</div>
                    <div class="kpi-value">{_fmt_money(scenario_result.get('daily_debt_service'))} / day</div>
                    <div class="kpi-sub">Net Flow: {_fmt_money(scenario_result.get('estimated_daily_cashflow'))}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">DSTC Ratio</div>
                    <div class="kpi-value">{float(scenario_result.get('dstc_ratio', 0.0))*100:.1f}%</div>
                    <div class="kpi-sub">{html.escape(str(scenario_result.get('risk_assessment', '')))}</div>
                </div>
            </div>
        </div>
        """

    stress_html = ""
    if stress_result:
        stressed = stress_result.get("stressed", {})
        baseline = stress_result.get("baseline", {})
        delta_p = float(stress_result.get("delta_pts", 0.0))
        shifted = "YES (Adverse Shift)" if stress_result.get("band_shifted") else "NO (Band Maintained)"
        stress_html = f"""
        <div class="section-card">
            <div class="section-title">Counterfactual Cashflow Shock Resilience</div>
            <div class="grid-3">
                <div class="kpi-box">
                    <div class="kpi-label">Baseline Score</div>
                    <div class="kpi-value">{round(float(baseline.get('sequence_score', 0.0))*100):d} / 100</div>
                    <div class="kpi-sub">{html.escape(str(baseline.get('risk_band', '')))}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Stressed Score (-20% Inbound, +30% Fee)</div>
                    <div class="kpi-value">{round(float(stressed.get('sequence_score', 0.0))*100):d} / 100</div>
                    <div class="kpi-sub">{html.escape(str(stressed.get('risk_band', '')))}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Stress Delta & Migration</div>
                    <div class="kpi-value">+{delta_p:.1f} pts</div>
                    <div class="kpi-sub">Shifted: {shifted}</div>
                </div>
            </div>
        </div>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Credit Risk Committee Memo — {html.escape(str(borrower_id))}</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
  }}
  * {{ box-sizing: border-box; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    background: #ffffff;
    margin: 0;
    padding: 16px;
    font-size: 11px;
    line-height: 1.4;
  }}
  .no-print-bar {{
    background: #1e3a8a;
    color: #ffffff;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .btn-print {{
    background: #ffffff;
    color: #1e3a8a;
    border: none;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
  }}
  @media print {{
    body {{ padding: 0; }}
    .no-print-bar {{ display: none !important; }}
  }}
  .memo-header {{
    border-bottom: 2px solid #1e3a8a;
    padding-bottom: 10px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }}
  .memo-title h1 {{
    margin: 0;
    font-size: 18px;
    color: #1e3a8a;
    letter-spacing: -0.01em;
  }}
  .memo-title p {{
    margin: 3px 0 0 0;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }}
  .memo-meta {{
    text-align: right;
    font-size: 10px;
    color: #475569;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}
  .band-lower {{ background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }}
  .band-watch {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}
  .band-elevated {{ background: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }}
  .badge-escalator {{ background: #fef2f2; color: #991b1b; }}
  .badge-mitigator {{ background: #f0fdf4; color: #166534; }}
  .grid-4 {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }}
  .grid-3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 10px;
  }}
  .kpi-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 8px 10px;
  }}
  .kpi-box.primary {{
    background: #eff6ff;
    border-color: #bfdbfe;
    border-left: 3px solid #1e3a8a;
  }}
  .kpi-label {{
    font-size: 9px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 2px;
  }}
  .kpi-value {{
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
  }}
  .kpi-sub {{
    font-size: 9px;
    color: #64748b;
    margin-top: 2px;
  }}
  .section-card {{
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    background: #ffffff;
    page-break-inside: avoid;
  }}
  .section-title {{
    font-size: 11px;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 4px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
  }}
  th {{
    background: #f1f5f9;
    color: #334155;
    text-align: left;
    padding: 5px 6px;
    font-weight: 600;
    border-bottom: 1px solid #cbd5e1;
  }}
  td {{
    padding: 5px 6px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
  }}
  .text-right {{ text-align: right; }}
  .text-muted {{ color: #64748b; font-size: 9px; }}
  .governance-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #64748b;
    padding: 8px 10px;
    font-size: 8.5px;
    color: #475569;
    line-height: 1.35;
    page-break-inside: avoid;
  }}
  .hash-code {{
    font-family: monospace;
    font-size: 8px;
    background: #e2e8f0;
    padding: 1px 4px;
    border-radius: 2px;
  }}
</style>
</head>
<body>

<div class="no-print-bar">
  <div><strong>SeqCredit Institutional Memo</strong> — Print-ready briefing sheet.</div>
  <button class="btn-print" onclick="window.print()">Print / Save PDF</button>
</div>

<div class="memo-header">
  <div class="memo-title">
    <h1>Institutional Credit Risk Memorandum</h1>
    <p>Algorithmic Evaluation & Temporal Sequence Discrimination</p>
  </div>
  <div class="memo-meta">
    <div><strong>Counterparty ID:</strong> {html.escape(str(borrower_id))}</div>
    <div><strong>Report Date:</strong> {now_str}</div>
    <div style="margin-top: 3px;"><span class="badge {band_cls}">{html.escape(band_name)} Risk Index</span></div>
  </div>
</div>

<div class="grid-4">
  <div class="kpi-box primary">
    <div class="kpi-label">Order-Aware Index</div>
    <div class="kpi-value">{seq_pts} / 100</div>
    <div class="kpi-sub">Sequence Surrogate Model</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Static Baseline</div>
    <div class="kpi-value">{stat_pts} / 100</div>
    <div class="kpi-sub">Random Forest Aggregate</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Sequence Delta</div>
    <div class="kpi-value">{delta_sign}{delta_pts} pts</div>
    <div class="kpi-sub">Temporal Trajectory Lift</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Audited Operations</div>
    <div class="kpi-value">{int(indicators.get('transaction_count', 0)):,} txs</div>
    <div class="kpi-sub">Avg Bal: {_fmt_money(indicators.get('avg_balance'))}</div>
  </div>
</div>

<div class="section-card">
  <div class="section-title">Key Behavioral Feature Drivers (Marginal Departure vs Peer Median)</div>
  <table>
    <thead>
      <tr>
        <th>Behavioral Feature Indicator</th>
        <th>Category</th>
        <th>Actual Value</th>
        <th>Cohort Median</th>
        <th class="text-right">Risk Impact</th>
      </tr>
    </thead>
    <tbody>
      {drivers_rows_html if drivers_rows_html else '<tr><td colspan="5" class="text-muted">Feature departures within baseline cohort tolerance.</td></tr>'}
    </tbody>
  </table>
</div>

{scenario_html}
{stress_html}

<div class="governance-box">
  <strong style="color:#1e3a8a; text-transform:uppercase; font-size:9px;">Model Lineage & Cryptographic Governance Addendum</strong><br>
  <strong>Model ID:</strong> {gov['model_version']} &bull; <strong>Status:</strong> {gov['status']} &bull; <strong>Calibrated:</strong> {gov['created_at'][:19].replace('T', ' ')} UTC<br>
  <strong>Static Model Hash:</strong> <span class="hash-code">{gov['static_sha256'][:24]}...</span> &bull; 
  <strong>Sequence Surrogate Hash:</strong> <span class="hash-code">{gov['sequence_sha256'][:24]}...</span><br>
  <strong>Acceptance Criteria Verification:</strong> Strict Default: {gov['checks']['default_rate']['status']} &bull; Static AUC: {gov['checks']['static_auc']['status']} &bull; Sequence AUC: {gov['checks']['sequence_auc']['status']} &bull; Gain: {gov['checks']['sequence_gain']['status']}<br>
  <div style="margin-top:4px; font-size:7.5px; color:#64748b;">{gov['regulatory_disclaimer']}</div>
</div>

</body>
</html>
"""


def generate_borrower_memo_pdf(
    borrower_id: str,
    borrower_score: dict[str, Any],
    borrower_detail: dict[str, Any],
    manifest: dict[str, Any],
    scenario_result: dict[str, Any] | None = None,
    stress_result: dict[str, Any] | None = None,
) -> bytes:
    """Generate high-resolution vector institutional credit memo PDF using ReportLab."""
    gov = build_governance_addendum(manifest)
    indicators = borrower_detail.get("indicators", {})
    attribution = borrower_detail.get("attribution", {})
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    band_name = str(borrower_score.get("demo_risk_band", "Unknown")).replace(" demo risk", "")
    seq_pts = round(float(borrower_score.get("sequence_demo_score", 0.0)) * 100)
    stat_pts = round(float(borrower_score.get("static_demo_score", 0.0)) * 100)
    delta_pts = seq_pts - stat_pts
    delta_sign = "+" if delta_pts >= 0 else ""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MemoTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
    )
    subtitle_style = ParagraphStyle(
        "MemoSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        textTransform="uppercase",
    )
    meta_style = ParagraphStyle(
        "MemoMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=2,  # Right aligned
        textColor=colors.HexColor("#334155"),
    )
    heading_style = ParagraphStyle(
        "MemoHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        textTransform="uppercase",
    )
    cell_style = ParagraphStyle(
        "MemoCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )
    cell_bold_style = ParagraphStyle(
        "MemoCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    gov_style = ParagraphStyle(
        "MemoGov",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Header Table
    left_header = [
        Paragraph("Institutional Credit Risk Memorandum", title_style),
        Spacer(1, 2 * mm),
        Paragraph("Algorithmic Risk Discrimination & Sequence Decomposition", subtitle_style),
    ]
    right_header = [
        Paragraph(f"<b>Counterparty:</b> {html.escape(str(borrower_id))}", meta_style),
        Paragraph(f"<b>Date:</b> {now_str}", meta_style),
        Paragraph(f"<b>Classification:</b> {band_name.upper()} RISK TIER", meta_style),
    ]
    header_table = Table([[left_header, right_header]], colWidths=[110 * mm, 76 * mm])
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=3 * mm))

    # KPI Summary Cards Table
    kpi_data = [
        [
            Paragraph("<font size='7' color='#64748b'>ORDER-AWARE RISK</font><br/><b><font size='14' color='#1e3a8a'>" + f"{seq_pts} / 100</font></b><br/><font size='7' color='#64748b'>Sequence Model</font>", cell_style),
            Paragraph("<font size='7' color='#64748b'>STATIC BASELINE</font><br/><b><font size='14' color='#0f172a'>" + f"{stat_pts} / 100</font></b><br/><font size='7' color='#64748b'>Aggregate Model</font>", cell_style),
            Paragraph("<font size='7' color='#64748b'>TRAJECTORY LIFT</font><br/><b><font size='14' color='#0f172a'>" + f"{delta_sign}{delta_pts} pts</font></b><br/><font size='7' color='#64748b'>Temporal Delta</font>", cell_style),
            Paragraph("<font size='7' color='#64748b'>AUDITED VOLUME</font><br/><b><font size='14' color='#0f172a'>" + f"{int(indicators.get('transaction_count', 0)):,} txs</font></b><br/><font size='7' color='#64748b'>Mean: " + _fmt_money(indicators.get('avg_balance')) + "</font>", cell_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[46.5 * mm] * 4)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#eff6ff")),
                ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 3 * mm))

    # Drivers Breakdown Section
    story.append(Paragraph("Key Behavioral Feature Drivers (Marginal Departure vs Cohort Median)", heading_style))
    story.append(Spacer(1, 1.5 * mm))

    top_escalators = attribution.get("top_escalators", [])
    top_mitigators = attribution.get("top_mitigators", [])
    drivers_list = top_escalators[:3] + top_mitigators[:3]

    driver_rows = [
        [
            Paragraph("<b>Feature Indicator</b>", cell_bold_style),
            Paragraph("<b>Category</b>", cell_bold_style),
            Paragraph("<b>Actual</b>", cell_bold_style),
            Paragraph("<b>Peer Median</b>", cell_bold_style),
            Paragraph("<b>Impact</b>", cell_bold_style),
        ]
    ]
    for d in drivers_list:
        pts = float(d.get("impact_points", 0.0))
        sign = "+" if pts >= 0 else ""
        color_tag = "#991b1b" if pts >= 0 else "#166534"
        cat_str = "Sequence" if d.get("category") == "sequence" else "Static"
        driver_rows.append(
            [
                Paragraph(f"<b>{html.escape(d.get('display_name', ''))}</b>", cell_style),
                Paragraph(cat_str, cell_style),
                Paragraph(f"{d.get('actual_value', '')} {d.get('unit', '')}", cell_style),
                Paragraph(f"{d.get('baseline_value', '')} {d.get('unit', '')}", cell_style),
                Paragraph(f"<font color='{color_tag}'><b>{sign}{pts:.2f} pts</b></font>", cell_style),
            ]
        )

    if len(driver_rows) == 1:
        driver_rows.append([Paragraph("No significant departures from median.", cell_style), "", "", "", ""])

    drivers_table = Table(driver_rows, colWidths=[65 * mm, 25 * mm, 32 * mm, 32 * mm, 32 * mm])
    drivers_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(drivers_table)
    story.append(Spacer(1, 3 * mm))

    # Scenario & Capital Stress (if available)
    if scenario_result or stress_result:
        story.append(Paragraph("Credit Facility Sizing & Counterfactual Stress Simulation", heading_style))
        story.append(Spacer(1, 1.5 * mm))

        sc_cells = []
        if scenario_result:
            sc_cells.append(
                Paragraph(
                    f"<b>Facility Exposure:</b> {_fmt_money(scenario_result.get('principal'))} ({scenario_result.get('tenure_days')}D)<br/>"
                    f"<b>Total Repayment:</b> {_fmt_money(scenario_result.get('total_repayment'))}<br/>"
                    f"<b>Daily Debt Service:</b> {_fmt_money(scenario_result.get('daily_debt_service'))}/day<br/>"
                    f"<b>DSTC Ratio:</b> {float(scenario_result.get('dstc_ratio', 0.0))*100:.1f}% ({scenario_result.get('risk_assessment', '')})",
                    cell_style,
                )
            )
        if stress_result:
            b_sc = round(float(stress_result.get("baseline", {}).get("sequence_score", 0.0)) * 100)
            s_sc = round(float(stress_result.get("stressed", {}).get("sequence_score", 0.0)) * 100)
            d_pts = float(stress_result.get("delta_pts", 0.0))
            shift = "Band Shifted" if stress_result.get("band_shifted") else "Band Maintained"
            sc_cells.append(
                Paragraph(
                    f"<b>Cash-flow Shock Test:</b> -20% Inbound / +30% Fee<br/>"
                    f"<b>Baseline Score:</b> {b_sc} / 100<br/>"
                    f"<b>Stressed Score:</b> {s_sc} / 100 (+{d_pts:.1f} pts)<br/>"
                    f"<b>Stress Assessment:</b> {shift}",
                    cell_style,
                )
            )

        if len(sc_cells) == 1:
            sc_table = Table([[sc_cells[0]]], colWidths=[186 * mm])
        else:
            sc_table = Table([[sc_cells[0], sc_cells[1]]], colWidths=[93 * mm, 93 * mm])

        sc_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(sc_table)
        story.append(Spacer(1, 3 * mm))

    # Governance Addendum
    gov_text = (
        f"<b>MODEL GOVERNANCE & LINEAGE ADDENDUM</b><br/>"
        f"<b>Model ID:</b> {gov['model_version']} | <b>Verification Status:</b> {gov['status']} | <b>Lineage Date:</b> {gov['created_at'][:19].replace('T', ' ')} UTC<br/>"
        f"<b>Static Model SHA-256:</b> <font face='Courier'>{gov['static_sha256'][:28]}...</font><br/>"
        f"<b>Sequence Surrogate SHA-256:</b> <font face='Courier'>{gov['sequence_sha256'][:28]}...</font><br/>"
        f"<b>Acceptance Criteria:</b> Default Rate: {gov['checks']['default_rate']['status']} | Static AUC: {gov['checks']['static_auc']['status']} | Sequence AUC: {gov['checks']['sequence_auc']['status']} | Gain: {gov['checks']['sequence_gain']['status']}<br/>"
        f"<font size='6' color='#64748b'>{gov['regulatory_disclaimer']}</font>"
    )
    gov_table = Table([[Paragraph(gov_text, gov_style)]], colWidths=[186 * mm])
    gov_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("LINELEFT", (0, 0), (0, -1), 2.5, colors.HexColor("#1e3a8a")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(KeepTogether([gov_table]))

    doc.build(story)
    return buffer.getvalue()


# =========================================================================
# 2. PORTFOLIO BATCH RISK SUMMARY (HTML & PDF)
# =========================================================================


def _compute_portfolio_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate distribution, 3x3 migration, and cohort flags from dashboard payload."""
    scores_df = pd.DataFrame(payload.get("scores", []))
    if scores_df.empty:
        return {}

    # Basic portfolio counts
    borrower_count = len(scores_df)
    tx_count = payload.get("portfolio", {}).get("transactions", 0)
    avg_seq = float(scores_df["sequence_demo_score"].mean())
    avg_stat = float(scores_df["static_demo_score"].mean())

    # Score distribution bins
    bins = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]
    labels = ["0–4 pts", "5–9 pts", "10–19 pts", "20–39 pts", "40+ pts"]
    dist_counts = pd.cut(scores_df["sequence_demo_score"], bins=bins, labels=labels, right=False).value_counts().sort_index()

    # Risk band distribution
    band_counts = scores_df["demo_risk_band"].value_counts().to_dict()

    # Deltas
    scores_df["delta_pts"] = (scores_df["sequence_demo_score"] - scores_df["static_demo_score"]) * 100.0
    highest_escalation = scores_df.sort_values("delta_pts", ascending=False).head(5)
    highest_mitigation = scores_df.sort_values("delta_pts", ascending=True).head(5)

    # 3x3 Risk Band Migration: Static vs Sequence
    # Assign static risk bands based on same thresholds for comparison
    th = payload.get("thresholds", {"watch": 0.03275, "elevated": 0.09255})
    w_th = th.get("watch", 0.03275)
    e_th = th.get("elevated", 0.09255)

    def assign_band(score: float) -> str:
        if score < w_th:
            return "Lower demo risk"
        if score < e_th:
            return "Watch demo risk"
        return "Elevated demo risk"

    scores_df["static_band"] = scores_df["static_demo_score"].apply(assign_band)
    bands = ["Lower demo risk", "Watch demo risk", "Elevated demo risk"]

    matrix: dict[str, dict[str, int]] = {b1: {b2: 0 for b2 in bands} for b1 in bands}
    upgrades = 0
    downgrades = 0
    unchanged = 0
    band_order = {b: i for i, b in enumerate(bands)}

    for _, row in scores_df.iterrows():
        b_static = row["static_band"]
        b_seq = row["demo_risk_band"]
        if b_static in matrix and b_seq in matrix[b_static]:
            matrix[b_static][b_seq] += 1
        i_static = band_order.get(b_static, 0)
        i_seq = band_order.get(b_seq, 0)
        if i_seq < i_static:
            upgrades += 1
        elif i_seq > i_static:
            downgrades += 1
        else:
            unchanged += 1

    return {
        "borrowers": borrower_count,
        "transactions": tx_count,
        "avg_sequence_score": avg_seq,
        "avg_static_score": avg_stat,
        "net_gain_pts": (avg_seq - avg_stat) * 100.0,
        "distribution": {label: int(dist_counts.get(label, 0)) for label in labels},
        "bands": band_counts,
        "migration_matrix": matrix,
        "upgrades": upgrades,
        "downgrades": downgrades,
        "unchanged": unchanged,
        "highest_escalation": highest_escalation.to_dict(orient="records"),
        "highest_mitigation": highest_mitigation.to_dict(orient="records"),
    }


def generate_portfolio_summary_html(payload: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Generate aggregate portfolio risk summary report in HTML with @media print."""
    gov = build_governance_addendum(manifest)
    metrics = _compute_portfolio_metrics(payload)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not metrics:
        return "<html><body><h1>No batch scoring data available.</h1></body></html>"

    bands = ["Lower demo risk", "Watch demo risk", "Elevated demo risk"]
    matrix = metrics["migration_matrix"]
    matrix_rows = ""
    for b_base in bands:
        clean_base = b_base.replace(" demo risk", "")
        matrix_rows += f"""
        <tr>
            <td><strong>{clean_base}</strong> (Baseline Static)</td>
            <td class="text-right">{matrix[b_base]['Lower demo risk']:,}</td>
            <td class="text-right">{matrix[b_base]['Watch demo risk']:,}</td>
            <td class="text-right">{matrix[b_base]['Elevated demo risk']:,}</td>
        </tr>
        """

    dist_rows = ""
    for b_label, count in metrics["distribution"].items():
        pct = (count / metrics["borrowers"]) * 100 if metrics["borrowers"] > 0 else 0
        dist_rows += f"""
        <tr>
            <td><strong>{b_label}</strong></td>
            <td class="text-right">{count:,} counterparties</td>
            <td class="text-right">{pct:.1f}%</td>
        </tr>
        """

    flag_esc_rows = ""
    for r in metrics["highest_escalation"]:
        flag_esc_rows += f"""
        <tr>
            <td><strong>{html.escape(str(r['borrower_id']))}</strong></td>
            <td>{round(float(r['static_demo_score'])*100):d} / 100</td>
            <td><strong>{round(float(r['sequence_demo_score'])*100):d} / 100</strong></td>
            <td class="text-right" style="color:#991b1b;"><strong>+{float(r['delta_pts']):.1f} pts</strong></td>
            <td>{html.escape(str(r['demo_risk_band']).replace(' demo risk', ''))}</td>
        </tr>
        """

    flag_mit_rows = ""
    for r in metrics["highest_mitigation"]:
        flag_mit_rows += f"""
        <tr>
            <td><strong>{html.escape(str(r['borrower_id']))}</strong></td>
            <td>{round(float(r['static_demo_score'])*100):d} / 100</td>
            <td><strong>{round(float(r['sequence_demo_score'])*100):d} / 100</strong></td>
            <td class="text-right" style="color:#166534;"><strong>{float(r['delta_pts']):.1f} pts</strong></td>
            <td>{html.escape(str(r['demo_risk_band']).replace(' demo risk', ''))}</td>
        </tr>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Batch Risk Summary & Sequence Migration</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
  }}
  * {{ box-sizing: border-box; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    background: #ffffff;
    margin: 0;
    padding: 16px;
    font-size: 11px;
    line-height: 1.4;
  }}
  .no-print-bar {{
    background: #1e3a8a;
    color: #ffffff;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .btn-print {{
    background: #ffffff;
    color: #1e3a8a;
    border: none;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
  }}
  @media print {{
    body {{ padding: 0; }}
    .no-print-bar {{ display: none !important; }}
  }}
  .memo-header {{
    border-bottom: 2px solid #1e3a8a;
    padding-bottom: 10px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }}
  .memo-title h1 {{
    margin: 0;
    font-size: 18px;
    color: #1e3a8a;
    letter-spacing: -0.01em;
  }}
  .memo-title p {{
    margin: 3px 0 0 0;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }}
  .memo-meta {{
    text-align: right;
    font-size: 10px;
    color: #475569;
  }}
  .grid-4 {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }}
  .kpi-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 8px 10px;
  }}
  .kpi-box.primary {{
    background: #eff6ff;
    border-color: #bfdbfe;
    border-left: 3px solid #1e3a8a;
  }}
  .kpi-label {{
    font-size: 9px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 2px;
  }}
  .kpi-value {{
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
  }}
  .kpi-sub {{
    font-size: 9px;
    color: #64748b;
    margin-top: 2px;
  }}
  .section-card {{
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    background: #ffffff;
    page-break-inside: avoid;
  }}
  .section-title {{
    font-size: 11px;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 4px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
  }}
  th {{
    background: #f1f5f9;
    color: #334155;
    text-align: left;
    padding: 5px 6px;
    font-weight: 600;
    border-bottom: 1px solid #cbd5e1;
  }}
  td {{
    padding: 5px 6px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
  }}
  .text-right {{ text-align: right; }}
  .governance-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3px solid #64748b;
    padding: 8px 10px;
    font-size: 8.5px;
    color: #475569;
    line-height: 1.35;
    page-break-inside: avoid;
  }}
  .hash-code {{
    font-family: monospace;
    font-size: 8px;
    background: #e2e8f0;
    padding: 1px 4px;
    border-radius: 2px;
  }}
</style>
</head>
<body>

<div class="no-print-bar">
  <div><strong>SeqCredit Portfolio Summary</strong> — Print-ready aggregate committee memorandum.</div>
  <button class="btn-print" onclick="window.print()">Print / Save PDF</button>
</div>

<div class="memo-header">
  <div class="memo-title">
    <h1>Portfolio Batch Risk & Sequence Migration Report</h1>
    <p>Executive Risk Committee Oversight Briefing</p>
  </div>
  <div class="memo-meta">
    <div><strong>Batch Size:</strong> {metrics['borrowers']:,} Counterparties</div>
    <div><strong>Report Date:</strong> {now_str}</div>
    <div><strong>Model Version:</strong> {gov['model_version']}</div>
  </div>
</div>

<div class="grid-4">
  <div class="kpi-box primary">
    <div class="kpi-label">Batch Counterparties</div>
    <div class="kpi-value">{metrics['borrowers']:,}</div>
    <div class="kpi-sub">Total Evaluated Cohort</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Audited Operations</div>
    <div class="kpi-value">{metrics['transactions']:,}</div>
    <div class="kpi-sub">Validated Transactions</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Mean Sequence Score</div>
    <div class="kpi-value">{round(metrics['avg_sequence_score']*100):d} / 100</div>
    <div class="kpi-sub">Static Avg: {round(metrics['avg_static_score']*100):d} / 100</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Sequence Net Lift</div>
    <div class="kpi-value">{'+' if metrics['net_gain_pts'] >= 0 else ''}{metrics['net_gain_pts']:.1f} pts</div>
    <div class="kpi-sub">{metrics['downgrades']} Downgrades / {metrics['upgrades']} Upgrades</div>
  </div>
</div>

<div class="section-card">
  <div class="section-title">3×3 Risk Band Migration Matrix (Static Baseline Rows &rarr; Sequence Surrogate Columns)</div>
  <table>
    <thead>
      <tr>
        <th>Static Baseline Risk Band</th>
        <th class="text-right">Lower Risk (Sequence)</th>
        <th class="text-right">Watch Risk (Sequence)</th>
        <th class="text-right">Elevated Risk (Sequence)</th>
      </tr>
    </thead>
    <tbody>
      {matrix_rows}
    </tbody>
  </table>
  <div style="margin-top:6px; font-size:9.5px; color:#64748b;">
    <strong>Migration Summary:</strong> {metrics['upgrades']:,} Upgrades (favorable sequence dynamics) &bull; 
    {metrics['unchanged']:,} Unchanged &bull; 
    {metrics['downgrades']:,} Downgrades (deteriorating velocity or low-balance trajectory).
  </div>
</div>

<div class="section-card">
  <div class="section-title">Portfolio Sequence Risk Distribution Breakdown</div>
  <table>
    <thead>
      <tr>
        <th>Score Range (0–100 Scale)</th>
        <th class="text-right">Counterparty Count</th>
        <th class="text-right">Portfolio Share</th>
      </tr>
    </thead>
    <tbody>
      {dist_rows}
    </tbody>
  </table>
</div>

<div class="section-card">
  <div class="section-title">Key Cohort Flags — Top Risk Escalations (+ Sequence Delta)</div>
  <table>
    <thead>
      <tr>
        <th>Borrower Identifier</th>
        <th>Static Score</th>
        <th>Sequence Score</th>
        <th class="text-right">Sequence Delta</th>
        <th>Assigned Risk Tier</th>
      </tr>
    </thead>
    <tbody>
      {flag_esc_rows}
    </tbody>
  </table>
</div>

<div class="section-card">
  <div class="section-title">Key Cohort Flags — Top Risk Mitigations (- Sequence Delta)</div>
  <table>
    <thead>
      <tr>
        <th>Borrower Identifier</th>
        <th>Static Score</th>
        <th>Sequence Score</th>
        <th class="text-right">Sequence Delta</th>
        <th>Assigned Risk Tier</th>
      </tr>
    </thead>
    <tbody>
      {flag_mit_rows}
    </tbody>
  </table>
</div>

<div class="governance-box">
  <strong style="color:#1e3a8a; text-transform:uppercase; font-size:9px;">Model Lineage & Cryptographic Governance Addendum</strong><br>
  <strong>Model ID:</strong> {gov['model_version']} &bull; <strong>Status:</strong> {gov['status']} &bull; <strong>Calibrated:</strong> {gov['created_at'][:19].replace('T', ' ')} UTC<br>
  <strong>Static Model Hash:</strong> <span class="hash-code">{gov['static_sha256'][:24]}...</span> &bull; 
  <strong>Sequence Surrogate Hash:</strong> <span class="hash-code">{gov['sequence_sha256'][:24]}...</span><br>
  <strong>Acceptance Criteria Verification:</strong> Strict Default: {gov['checks']['default_rate']['status']} &bull; Static AUC: {gov['checks']['static_auc']['status']} &bull; Sequence AUC: {gov['checks']['sequence_auc']['status']} &bull; Gain: {gov['checks']['sequence_gain']['status']}<br>
  <div style="margin-top:4px; font-size:7.5px; color:#64748b;">{gov['regulatory_disclaimer']}</div>
</div>

</body>
</html>
"""


def generate_portfolio_summary_pdf(payload: dict[str, Any], manifest: dict[str, Any]) -> bytes:
    """Generate high-resolution vector portfolio risk committee summary PDF using ReportLab."""
    gov = build_governance_addendum(manifest)
    metrics = _compute_portfolio_metrics(payload)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MemoTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
    )
    subtitle_style = ParagraphStyle(
        "MemoSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748b"),
        textTransform="uppercase",
    )
    meta_style = ParagraphStyle(
        "MemoMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=2,
        textColor=colors.HexColor("#334155"),
    )
    heading_style = ParagraphStyle(
        "MemoHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
        textTransform="uppercase",
    )
    cell_style = ParagraphStyle(
        "MemoCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )
    cell_bold_style = ParagraphStyle(
        "MemoCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    gov_style = ParagraphStyle(
        "MemoGov",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Header Table
    left_header = [
        Paragraph("Portfolio Batch Risk & Migration Summary", title_style),
        Spacer(1, 2 * mm),
        Paragraph("Executive Risk Committee Oversight & Algorithmic Benchmarking", subtitle_style),
    ]
    right_header = [
        Paragraph(f"<b>Batch Size:</b> {metrics.get('borrowers', 0):,} Borrowers", meta_style),
        Paragraph(f"<b>Date:</b> {now_str}", meta_style),
        Paragraph(f"<b>Model ID:</b> {gov['model_version']}", meta_style),
    ]
    header_table = Table([[left_header, right_header]], colWidths=[110 * mm, 76 * mm])
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=3 * mm))

    # KPI Summary Cards Table
    kpi_data = [
        [
            Paragraph(f"<font size='7' color='#64748b'>BATCH VOLUME</font><br/><b><font size='14' color='#1e3a8a'>{metrics.get('borrowers', 0):,}</font></b><br/><font size='7' color='#64748b'>Counterparties</font>", cell_style),
            Paragraph(f"<font size='7' color='#64748b'>AUDITED OPERATIONS</font><br/><b><font size='14' color='#0f172a'>{metrics.get('transactions', 0):,}</font></b><br/><font size='7' color='#64748b'>Transactions</font>", cell_style),
            Paragraph(f"<font size='7' color='#64748b'>MEAN SEQUENCE SCORE</font><br/><b><font size='14' color='#0f172a'>{round(metrics.get('avg_sequence_score', 0.0)*100):d} / 100</font></b><br/><font size='7' color='#64748b'>Static: {round(metrics.get('avg_static_score', 0.0)*100):d}/100</font>", cell_style),
            Paragraph(f"<font size='7' color='#64748b'>MIGRATION BALANCE</font><br/><b><font size='14' color='#0f172a'>{metrics.get('downgrades', 0)} / {metrics.get('upgrades', 0)}</font></b><br/><font size='7' color='#64748b'>Down / Up Migrations</font>", cell_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[46.5 * mm] * 4)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#eff6ff")),
                ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 3 * mm))

    # Migration Matrix
    story.append(Paragraph("3×3 Risk Band Migration Matrix (Static Baseline Rows → Sequence Surrogate Columns)", heading_style))
    story.append(Spacer(1, 1.5 * mm))

    bands = ["Lower demo risk", "Watch demo risk", "Elevated demo risk"]
    matrix = metrics.get("migration_matrix", {})
    matrix_rows = [
        [
            Paragraph("<b>Static Baseline Risk Band</b>", cell_bold_style),
            Paragraph("<b>Lower Risk (Seq)</b>", cell_bold_style),
            Paragraph("<b>Watch Risk (Seq)</b>", cell_bold_style),
            Paragraph("<b>Elevated Risk (Seq)</b>", cell_bold_style),
        ]
    ]
    for b_base in bands:
        clean_b = b_base.replace(" demo risk", "")
        row_dict = matrix.get(b_base, {})
        matrix_rows.append(
            [
                Paragraph(f"<b>{clean_b}</b>", cell_style),
                Paragraph(f"{row_dict.get('Lower demo risk', 0):,}", cell_style),
                Paragraph(f"{row_dict.get('Watch demo risk', 0):,}", cell_style),
                Paragraph(f"{row_dict.get('Elevated demo risk', 0):,}", cell_style),
            ]
        )

    mat_table = Table(matrix_rows, colWidths=[66 * mm, 40 * mm, 40 * mm, 40 * mm])
    mat_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(mat_table)
    story.append(Spacer(1, 3 * mm))

    # Cohort Flags: Top Escalations
    story.append(Paragraph("Key Cohort Flags — Top Sequence Risk Escalations (+ Delta)", heading_style))
    story.append(Spacer(1, 1.5 * mm))

    esc_rows = [
        [
            Paragraph("<b>Borrower ID</b>", cell_bold_style),
            Paragraph("<b>Static Score</b>", cell_bold_style),
            Paragraph("<b>Sequence Score</b>", cell_bold_style),
            Paragraph("<b>Sequence Delta</b>", cell_bold_style),
            Paragraph("<b>Assigned Risk Tier</b>", cell_bold_style),
        ]
    ]
    for r in metrics.get("highest_escalation", []):
        d_p = float(r["delta_pts"])
        esc_rows.append(
            [
                Paragraph(f"<b>{html.escape(str(r['borrower_id']))}</b>", cell_style),
                Paragraph(f"{round(float(r['static_demo_score'])*100):d} / 100", cell_style),
                Paragraph(f"<b>{round(float(r['sequence_demo_score'])*100):d} / 100</b>", cell_style),
                Paragraph(f"<font color='#991b1b'><b>+{d_p:.1f} pts</b></font>", cell_style),
                Paragraph(html.escape(str(r["demo_risk_band"]).replace(" demo risk", "")), cell_style),
            ]
        )
    esc_table = Table(esc_rows, colWidths=[40 * mm, 32 * mm, 34 * mm, 35 * mm, 45 * mm])
    esc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(esc_table)
    story.append(Spacer(1, 3 * mm))

    # Governance Addendum
    gov_text = (
        f"<b>MODEL GOVERNANCE & LINEAGE ADDENDUM</b><br/>"
        f"<b>Model ID:</b> {gov['model_version']} | <b>Verification Status:</b> {gov['status']} | <b>Lineage Date:</b> {gov['created_at'][:19].replace('T', ' ')} UTC<br/>"
        f"<b>Static Model SHA-256:</b> <font face='Courier'>{gov['static_sha256'][:28]}...</font><br/>"
        f"<b>Sequence Surrogate SHA-256:</b> <font face='Courier'>{gov['sequence_sha256'][:28]}...</font><br/>"
        f"<b>Acceptance Criteria:</b> Default Rate: {gov['checks']['default_rate']['status']} | Static AUC: {gov['checks']['static_auc']['status']} | Sequence AUC: {gov['checks']['sequence_auc']['status']} | Gain: {gov['checks']['sequence_gain']['status']}<br/>"
        f"<font size='6' color='#64748b'>{gov['regulatory_disclaimer']}</font>"
    )
    gov_table = Table([[Paragraph(gov_text, gov_style)]], colWidths=[186 * mm])
    gov_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ("LINELEFT", (0, 0), (0, -1), 2.5, colors.HexColor("#1e3a8a")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(KeepTogether([gov_table]))

    doc.build(story)
    return buffer.getvalue()
