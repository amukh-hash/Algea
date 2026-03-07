"""test_runtime_todo_gate — CI guard against unfinished code in production paths.

Recursively scans all production-critical directories for TODO, FIXME, XXX,
and bare NotImplementedError. Files in the allow-list are excluded.

Phase 6 hardening: expanded scope to cover backend/app/ and algae/ fully.
The experimental/ directory is excluded since it is explicitly for prototypes.
"""
from pathlib import Path
import re

import pytest


# Markers that indicate unfinished code
_DENY_PATTERN = re.compile(r"\bTODO\b|\bFIXME\b|\bXXX\b|NotImplementedError")

# Paths (substrings) that are explicitly allowed to contain markers
_ALLOW_SUBSTRINGS = {
    "tests",
    "docs",
    "experimental",           # algae/models/experimental/ — prototypes OK
    "base.py",                # abstract interfaces (foundation/base.py, env_base.py, broker_base.py)
    "vrp_strategy.py",        # train() deliberately raises NIE
    "runtime_mode.py",        # defines error classes
    "chronos2.py",            # shim with deliberate NIE on train()
}

# Production-critical directories to scan
_SCAN_ROOTS = [
    Path("backend/app/orchestrator"),
    Path("backend/app/api"),
    Path("backend/app/jobs"),
    Path("backend/app/ml_platform"),
    Path("backend/app/core"),
    Path("algae/trading"),
    Path("algae/execution"),
    Path("algae/data"),
    Path("algae/models"),
]


def test_runtime_todo_gate():
    """No TODO/FIXME/XXX/NotImplementedError in production paths."""
    violations: list[str] = []

    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            s = str(p)
            if any(a in s for a in _ALLOW_SUBSTRINGS):
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(txt.splitlines(), 1):
                # Skip comments that are just inline documentation
                stripped = line.strip()
                if stripped.startswith("#") and "TODO" not in stripped:
                    continue
                if _DENY_PATTERN.search(line):
                    violations.append(f"{p}:{lineno}: {stripped[:80]}")

    assert not violations, (
        f"Found {len(violations)} disallowed marker(s) in production paths:\n"
        + "\n".join(violations[:20])
        + ("\n..." if len(violations) > 20 else "")
    )
