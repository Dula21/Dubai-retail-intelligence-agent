"""
tests/conftest.py
-----------------
Adds /app/backend to sys.path so pytest can resolve `agents` and `services`
imports without requiring the backend package to be installed.

This is needed because:
  - Tests live at /app/tests/
  - Source modules live at /app/backend/agents/ and /app/backend/services/
  - pytest's rootdir is /app — backend/ is not on sys.path by default
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))