"""Build or verify data/yara_telecel_dump.csv.gz. No options required."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src/scripts"))

from dataset_io import build


if __name__ == "__main__":
    build()
