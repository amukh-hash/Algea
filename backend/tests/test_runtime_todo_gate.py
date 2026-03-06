from pathlib import Path
import re


def test_runtime_todo_gate():
    deny = re.compile(r"TODO|FIXME|XXX|NotImplementedError")
    allow = {"tests", "docs", "deprecated", "lagllama.py", "foundation/base.py", "broker_base.py"}
    roots = [Path("backend/app/orchestrator"), Path("backend/app/api"), Path("algae/trading")]
    for root in roots:
        for p in root.rglob("*.py"):
            s = str(p)
            if any(a in s for a in allow):
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            assert not deny.search(txt), f"disallowed marker in runtime path: {p}"
