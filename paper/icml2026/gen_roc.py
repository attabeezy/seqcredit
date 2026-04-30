"""Render roc_curves.pdf from roc_data.json printed by the Databricks notebook.

Usage
-----
1. Run Cell 22 on Databricks after Cell 21 has produced cv_oof_preds.npz.
2. Copy the single JSON line between === ROC_JSON_START === and === ROC_JSON_END ===.
3. Paste it into  paper/icml2026/roc_data.json  (the whole file is just that one line).
4. From the repo root:
       python paper/icml2026/gen_roc.py
   Output: paper/icml2026/roc_curves.pdf  (and roc_curves.png)
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
data_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "roc_data.json"

if not data_path.exists():
    sys.exit(f"ERROR: {data_path} not found.\nCreate it by pasting the JSON line from Cell 22 output.")

with open(data_path, encoding="utf-8") as f:
    roc_data = json.load(f)

MODELS = ["LogisticRegression", "XGBoost", "RandomForest", "LightGBM"]
LABELS = {"LogisticRegression": "LR", "XGBoost": "XGBoost",
          "RandomForest": "RF", "LightGBM": "LightGBM"}
COLORS = {"LogisticRegression": "#e41a1c", "XGBoost": "#377eb8",
          "RandomForest": "#4daf4a", "LightGBM": "#984ea3"}

TARGETS = [
    ("y_default", r"$y_{\mathrm{default}}$ (strict default)"),
    ("y_bad",     r"$y_{\mathrm{bad}}$ (default or penalty)"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (target, title) in zip(axes, TARGETS):
    target_data = roc_data.get(target, {})
    for model in MODELS:
        d = target_data.get(model)
        if d:
            ax.plot(d["fpr"], d["tpr"],
                    color=COLORS[model], lw=1.6,
                    label=f"{LABELS[model]} (AUC = {d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="_diagonal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()

out_pdf = HERE / "roc_curves.pdf"
out_png = HERE / "roc_curves.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
