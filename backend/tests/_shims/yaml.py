from __future__ import annotations


def safe_load(text: str):
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0] != "jobs:":
        return {}
    jobs = []
    current = None
    for ln in lines[1:]:
        if ln.startswith("  - "):
            if current:
                jobs.append(current)
            current = {}
            ln = ln[4:]
            if ln:
                k, v = [x.strip() for x in ln.split(":", 1)]
                current[k] = _parse(v)
        elif ln.startswith("    ") and current is not None:
            k, v = [x.strip() for x in ln.strip().split(":", 1)]
            current[k] = _parse(v)
    if current:
        jobs.append(current)
    return {"jobs": jobs}


def _parse(v: str):
    if v == "[]":
        return []
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [part.strip() for part in inner.split(",")]
    if v in ("true", "false"):
        return v == "true"
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v
