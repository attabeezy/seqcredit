# SeqCredit Presentation MVP - Development Roadmap & To-Do List

This document tracks planned enhancements, feature backlogs, and architectural extensions for the `seqcredit-mvp` prototype.

---

## 📋 Backlog Overview

- [x] **1. Explainability & Feature Attribution** *(Completed)*
- [x] **2. Scenario Stress-Testing & Policy Simulations** *(Completed)*
- [x] **3. Executive Risk Committee Export (PDF / Print Report)** *(Completed)*
- [x] **4. Frontend & Presentation UX Refinements** *(Completed)*
- [x] **5. Synthetic Cohort & Model Architecture Evolution** *(Completed)*

---

## 1. Explainability & Feature Attribution

Provide risk officers and model reviewers with transparent rationale for why a borrower was scored in a specific risk band, highlighting the difference between aggregate behavior and temporal sequence dynamics.

- [x] **Top Feature Drivers Breakdown:**
  - Compute contribution scores (exact counterfactual marginal attribution against cohort median baseline) for sequence and static features.
  - Display top positive (risk-increasing) and negative (risk-mitigating) factors for any selected borrower.
- [x] **Static vs. Sequence Delta Analysis:**
  - Specifically isolate the delta: why did the sequence surrogate score the borrower higher or lower than the static model?
  - Highlight key velocity/trajectory drivers (e.g., declining balance slope, rising cash-out frequency, or recent low-balance incidents).
- [x] **Waterfall / Driver Visualizations:**
  - Implement an interactive ECharts waterfall and horizontal divergence chart showing factor contributions moving the score away from the cohort baseline.

---

## 2. Scenario Stress-Testing & Policy Simulations

Extend the what-if loan scenario tool to evaluate borrower resilience under varying credit terms and economic pressures.

- [x] **Multi-Term Sizing Simulation:**
  - Allow simulating loan tenures (e.g., 7-day, 14-day, 30-day micro-loans) alongside requested principal.
  - Calculate debt-service-to-cashflow ratios against observed synthetic liquidity.
- [x] **Income Shock & Cash-Flow Stress Tests:**
  - Interactive sliders to simulate shocks: -20% inbound transfer drop, +30% fee burden, or prolonged inactivity periods.
  - Re-evaluate the dynamic risk index under stressed conditions in real-time.
- [x] **Threshold & Underwriting Policy Sandbox:**
  - Interactive policy cutoff sliders (e.g., adjusting "Watch" and "Elevated" score thresholds).
  - Live matrix showing portfolio acceptance rate, expected strict default capture, and risk band shifts.

---

## 3. Executive Risk Committee Export (PDF / Print Report)

Enable one-click generation of professional institutional briefing documents for risk committees and external stakeholders.

- [x] **Institutional Single-Borrower Memo:**
  - Clean, print-ready 1-page summary sheet containing borrower trajectory, risk gauge, top drivers, and scenario analysis.
  - Optimized CSS print media styles (`@media print`) and PDF download capability.
- [x] **Portfolio Batch Risk Summary:**
  - Downloadable aggregate report covering batch size, score distributions, risk band migration from static to sequence, and key cohort flags.
- [x] **Model Governance & Calibration Addendum:**
  - Automatically append model lineage metadata, hash verification, acceptance criteria compliance, and non-lending disclaimer disclaimers to exported reports.

---

## 4. Frontend & Presentation UX Refinements

Polish the Streamlit app for seamless presentation and demonstration workflows.

- [x] **Presentation Mode Toggle:**
  - Fullscreen, high-contrast, uncluttered display mode tailored for conference projectors and slide demos.
- [x] **Consolidated Streamlit Architecture:**
  - Standardized on Streamlit as the sole presentation dashboard, removing legacy standalone web server.
- [x] **Batch Comparison & Cohort Filtering:**
  - Add cohort filtering controls (e.g., filter by transaction volume, balance tier, or trajectory type) across the portfolio view.
- [x] **Sample Dataset Presets:**
  - Add quick-switch sample presets (e.g., "High-risk deteriorating borrower", "Low-risk consistent borrower", "High-volume volatile merchant").

---

## 5. Synthetic Cohort & Model Architecture Evolution

Enhance the underlying synthetic generator and surrogate models to showcase richer mobile financial services behaviors.

- [x] **Richer MFS Behavioral Archetypes:**
  - Introduce realistic synthetic behavioral archetypes: salary cyclicality, seasonal merchant spikes, payday spending rushes, and agent rebalancing.
- [x] **Calibration & Reliability Visualizations:**
  - Add reliability diagrams (calibration curves) and Brier score breakdowns directly in the model comparison tab.
- [x] **Sequence Length & Observation Window Sensitivity:**
  - Allow testing how the surrogate behaves across different history lengths (e.g., 10 vs 30 vs 50 transactions).
- [x] **Model Contract Versioning:**
  - Support side-by-side evaluation of multiple artifact versions (e.g., `v1` vs `v2`) through the manifest interface.
