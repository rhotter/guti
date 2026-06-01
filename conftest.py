"""Pytest configuration.

Adds the ``scripts/`` directory to ``sys.path`` so tests can import the
analysis/driver scripts (e.g. ``compute_information_maps``,
``export_svd_json``, ``recompute_meg_variants``) by bare module name, the same
way they did when those scripts lived at the repo root.
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent / "scripts"
if _SCRIPTS_DIR.is_dir():
    sys.path.insert(0, str(_SCRIPTS_DIR))
