"""Architectural invariant (ADR-0003): only `ui/` imports Streamlit.

Issue #005 acceptance: "No module outside `ui/` imports Streamlit (enforced
by a linting rule or import guard)." This test is the guard.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
STREAMLIT_IMPORT = re.compile(r"^\s*(?:import\s+streamlit\b|from\s+streamlit\b)", re.M)

_SKIP_DIRS = {".venv", ".git", "build", "dist", "ui"}


def test_streamlit_only_imported_inside_ui() -> None:
    offenders: list[str] = []
    self_path = pathlib.Path(__file__).resolve()

    for py in ROOT.rglob("*.py"):
        if py.resolve() == self_path:
            continue
        if any(part in _SKIP_DIRS for part in py.relative_to(ROOT).parts):
            continue
        if STREAMLIT_IMPORT.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(ROOT)))

    assert not offenders, (
        "Streamlit imported outside ui/ (violates ADR-0003): " + ", ".join(offenders)
    )
