"""Make the repo-root `finance_forecast` module importable from tests/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
