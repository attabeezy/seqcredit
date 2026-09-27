"""Batch scoring contract for the synthetic presentation dashboard."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

# The Windows demo environment can emit a verbose joblib CPU-discovery warning
# while loading an otherwise valid frozen artifact. Keep presentation startup
# clean; schema and model-contract failures remain hard exceptions below.
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd

from seqcredit_mvp.features import build_feature_tables


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"


class DemoScorer:
    """Load frozen synthetic artifacts and score a transaction batch."""

    def __init__(
        self,
        artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
        version: str | None = None,
    ):
        base_dir = Path(artifact_dir)
        target_dir = base_dir
        if version:
            candidates = [
                base_dir / version,
                base_dir / f"synthetic-demo-{version}",
                base_dir / f"v{version.lstrip('v')}",
            ]
            found = False
            for cand in candidates:
                if cand.is_dir() and (cand / "manifest.json").exists():
                    target_dir = cand
                    found = True
                    break
            if not found:
                manifest_path = base_dir / "manifest.json"
                if manifest_path.exists():
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if m.get("model_version") in (version, f"synthetic-demo-{version}"):
                        target_dir = base_dir
                        found = True
                if not found:
                    raise FileNotFoundError(
                        f"Model version '{version}' not found in artifact directory '{base_dir}'."
                    )

        manifest_path = target_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Demo artifacts are missing. Run 'python -m seqcredit_mvp.calibrate' first."
            )
        self.artifact_dir = target_dir
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "passed":
            raise ValueError("The synthetic model manifest did not pass its calibration gate.")
        self.static_model = joblib.load(target_dir / "static_model.joblib")
        self.sequence_model = joblib.load(target_dir / "sequence_model.joblib")

    @classmethod
    def available_versions(cls, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> list[str]:
        """Discover available model artifact versions within the artifact directory."""
        base_dir = Path(artifact_dir)
        versions: list[str] = []
        if (base_dir / "manifest.json").exists():
            try:
                m = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
                v = m.get("model_version", "synthetic-demo-v1")
                versions.append(v)
            except Exception:
                pass
        if base_dir.is_dir():
            for sub in sorted(base_dir.iterdir()):
                if sub.is_dir() and (sub / "manifest.json").exists():
                    try:
                        m = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
                        v = m.get("model_version", sub.name)
                        if v not in versions:
                            versions.append(v)
                    except Exception:
                        pass
        return versions or ["synthetic-demo-v1"]

    def score(self, transactions: pd.DataFrame) -> pd.DataFrame:
        static, sequence = build_feature_tables(transactions)
        expected_static = self.manifest["models"]["static"]["features"]
        expected_sequence = self.manifest["models"]["sequence"]["features"]
        static = static.loc[:, expected_static]
        sequence = sequence.loc[:, expected_sequence]

        static_scores = self.static_model.predict_proba(static)[:, 1]
        sequence_scores = self.sequence_model.predict_proba(sequence)[:, 1]
        thresholds = self.manifest["demo_risk_band_thresholds"]

        def band(score: float) -> str:
            if score >= thresholds["elevated"]:
                return "Elevated demo risk"
            if score >= thresholds["watch"]:
                return "Watch demo risk"
            return "Lower demo risk"

        scored_at = datetime.now(timezone.utc).isoformat()
        result = pd.DataFrame(
            {
                "borrower_id": sequence.index,
                "static_demo_score": static_scores,
                "sequence_demo_score": sequence_scores,
                "demo_risk_band": [band(value) for value in sequence_scores],
                "model_version": self.manifest["model_version"],
                "scored_at_utc": scored_at,
                "synthetic_demo": True,
                "decision": "No lending decision",
            }
        )
        return result.sort_values("sequence_demo_score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "scored_sample.csv")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    transactions = pd.read_csv(args.input_csv)
    scored = DemoScorer(args.artifacts).score(transactions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored):,} fabricated borrowers -> {args.output}")


if __name__ == "__main__":
    main()
