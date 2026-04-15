"""Experiment scripts for pain detection."""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `pain_detection` is importable
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
