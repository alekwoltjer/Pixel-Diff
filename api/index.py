import sys
from pathlib import Path

# Make the project root importable so app.py and blueprint_diff.py are found
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app  # noqa: F401 — Vercel looks for `app` in this module
