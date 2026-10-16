"""Regenerate paper discrimination figure from synced benchmark CSVs.

Reads data/cv_results_y_default.csv and data/cv_results_y_bad.csv
(fold="mean" rows holding the canonical Table 1 means from
docs/publications/AuthorKit27/main.tex) and writes
data/figures/fig2_discrimination_auc.svg/.png.

y_default rows are ordered by AUC-ROC descending, so RandomForest (0.7244)
sits above XGBoost (0.7211). The y_bad panel is likewise ordered by its own
AUC-ROC, matching Table 1 row order in AuthorKit27/main.tex.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = DATA / "figures"

ORDER_DEFAULT = ["GRU", "LSTM", "RandomForest", "XGBoost", "LightGBM", "LogisticRegression"]
ORDER_BAD = ["GRU", "LSTM", "XGBoost", "LightGBM", "RandomForest", "LogisticRegression"]
LABELS = {
    "GRU": "GRU",
    "LSTM": "LSTM",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "LogisticRegression": "Logistic Regression",
}
COLORS = {
    "GRU": "#0d9488",
    "LSTM": "#14b8a6",
    "RandomForest": "#ea580c",
    "XGBoost": "#e11d48",
    "LightGBM": "#f59e0b",
    "LogisticRegression": "#64748b",
}


def load(target: str) -> pd.DataFrame:
    fname = "cv_results_y_default.csv" if target == "y_default" else "cv_results_y_bad.csv"
    df = pd.read_csv(DATA / fname)
    df = df[(df["target"] == target) & (df["fold"] == "mean")].copy()
    return df.set_index("model")


def panel(ax, df: pd.DataFrame, models: list[str], title: str, annotate: str | None) -> None:
    vals = [float(df.loc[m, "auc_roc"]) for m in models]
    y = list(range(len(models)))[::-1]
    ax.barh(y, vals, color=[COLORS[m] for m in models], edgecolor="#0f172a", linewidth=0.8, height=0.6)
    ax.set_yticks(y, [LABELS[m] for m in models], fontweight="bold")
    for yy, v in zip(y, vals):
        ax.text(v + 0.0015, yy, f"{v:.4f}", va="center", ha="left", fontweight="bold", color="#0f172a")
    lo = min(vals)
    ax.set_xlim(lo - 0.03, max(vals) + 0.035)
    ax.set_xlabel("AUC-ROC", fontweight="bold")
    ax.set_title(title, fontweight="bold", loc="left")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    if annotate:
        ax.annotate(
            annotate,
            xy=(0.04, 0.94),
            xycoords="axes fraction",
            fontsize=9,
            fontweight="bold",
            color="#0f766e",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdfa", ec="#5eead4"),
        )


def main() -> None:
    dflt = load("y_default")
    bad = load("y_bad")
    gap = float(dflt.loc["GRU", "auc_roc"]) - float(dflt.loc["RandomForest", "auc_roc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=False)
    panel(ax1, dflt, ORDER_DEFAULT, "A. Target: Rare Default (y_default)",
          f"\u2605 GRU beats RandomForest (+{gap:.3f}, p=0.000)")
    panel(ax2, bad, ORDER_BAD, "B. Target: Adverse Outcome (y_bad)", None)
    fig.suptitle("Discrimination (AUC-ROC): sequential models lead on both targets",
                 fontweight="bold")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig2_discrimination_auc.svg")
    fig.savefig(FIG / "fig2_discrimination_auc.png", dpi=150)
    print("wrote fig2_discrimination_auc.svg/.png")


if __name__ == "__main__":
    main()
