"""Run unit checks and a small end-to-end integration test."""

from pathlib import Path
import subprocess
import sys

from workflow import check


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=Path(__file__).resolve().parents[1], check=True)
    check(borrowers=1000, epochs=2)
