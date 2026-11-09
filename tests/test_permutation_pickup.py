"""Check incremental pickup verification without contacting the backend."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/colab"))

from permutation_pickup import missing, sha


def _manifest(dest: Path, files: dict) -> dict:
    return {"status": "passed", "files": {name: sha(dest / name) for name in files}}


def test_missing_detects_absent_and_tampered(tmp_path):
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation.json").write_text("{}", encoding="utf-8")
    manifest = _manifest(tmp_path, {"metrics.json": True, "validation.json": True})
    assert missing(manifest, tmp_path) == []
    (tmp_path / "validation.json").write_text('{"x": 1}', encoding="utf-8")
    assert missing(manifest, tmp_path) == ["validation.json"]
    (tmp_path / "predictions.csv").write_text("a\n", encoding="utf-8")
    manifest["files"]["predictions.csv"] = sha(tmp_path / "predictions.csv")
    assert missing(manifest, tmp_path) == ["validation.json"]
    (tmp_path / "validation.json").unlink()
    assert missing(manifest, tmp_path) == ["validation.json"]
