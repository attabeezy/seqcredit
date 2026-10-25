"""Train and compare the fixed models on the shared dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src/scripts"))

from competition import main


if __name__ == "__main__":
    main()
