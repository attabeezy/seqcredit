"""Root Streamlit entrypoint for the SeqCredit presentation app."""

from __future__ import annotations

import sys
from pathlib import Path


APP_SOURCE = Path(__file__).resolve().parent / "app" / "src"
if str(APP_SOURCE) not in sys.path:
    sys.path.insert(0, str(APP_SOURCE))

from seqcredit_mvp.streamlit_app import main


if __name__ == "__main__":
    main()
