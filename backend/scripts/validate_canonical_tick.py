"""Post-tick validation for Intent Supremacy burn-in.

Run after each orchestrator tick to validate canonical artifacts.
Usage:
    python -m backend.scripts.validate_canonical_tick [artifact_root] [run_id] [asof_date]

Or import and call from orchestrator hook:
    from backend.scripts.validate_canonical_tick import validate_tick
    validate_tick(artifact_root, run_id, asof_date)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_tick(
    artifact_root: Path,
    run_id: str = "",
    asof_date: str = "",
    *,
    stage: int = 1,
) -> dict:
    """Validate canonical artifacts for the current tick.

    Returns a report dict with check results and any warnings/errors.
    """
    checks: list[dict] = []
    warnings: list[str] = []
    errors: list[str] = []

    # ── Stage 1+: Sleeve decisions ────────────────────────────────────
    default_sleeves = ["core", "vrp", "selector"]
    sleeves_dir = artifact_root / "sleeves"

    for sleeve in default_sleeves:
        dp = sleeves_dir / sleeve / "decision.json"
        if not dp.exists():
            errors.append(f"MISSING: {dp}")
            checks.append({"check": f"sleeve_decision_{sleeve}", "status": "FAIL", "reason": "missing"})
            continue

        try:
            d = _read_json(dp)
        except Exception as e:
            errors.append(f"CORRUPT: {dp}: {e}")
            checks.append({"check": f"sleeve_decision_{sleeve}", "status": "FAIL", "reason": str(e)})
            continue

        status = d.get("status", "unknown")
        n_intents = len(d.get("intents", []))

        # Schema version
        if d.get("schema_version") != "sleeve_decision.v1":
            errors.append(f"{sleeve}: bad schema_version={d.get('schema_version')}")

        # run_id match
        if run_id and d.get("run_id") != run_id:
            warnings.append(f"{sleeve}: run_id mismatch: {d.get('run_id')} != {run_id}")

        # OK must have intents
        if status == "ok" and n_intents == 0:
            errors.append(f"{sleeve}: status=OK but 0 intents (invariant violation)")

        # HALTED/DISABLED/FAILED must NOT have intents
        if status in ("halted", "disabled", "failed") and n_intents > 0:
            errors.append(f"{sleeve}: status={status} but {n_intents} intents (invariant violation)")

        # Diagnostics populated
        diag = d.get("diagnostics", {})
        if not diag.get("source_branch"):
            warnings.append(f"{sleeve}: diagnostics.source_branch empty")

        checks.append({
            "check": f"sleeve_decision_{sleeve}",
            "status": "OK",
            "sleeve_status": status,
            "n_intents": n_intents,
            "source_branch": diag.get("source_branch", ""),
        })

    # ── Stage 1: Legacy parity (compat artifacts still written) ───────
    for sleeve in default_sleeves:
        tgt = artifact_root / "targets" / f"{sleeve}_targets.json"
        sig = artifact_root / "signals" / f"{sleeve}_signals.json"
        if not tgt.exists():
            warnings.append(f"MISSING legacy target: {tgt}")
        if not sig.exists():
            warnings.append(f"MISSING legacy signal: {sig}")

    # ── Stage 2+: Canonical intents ───────────────────────────────────
    if stage >= 2:
        ci_path = artifact_root / "canonical_intents.json"
        if not ci_path.exists():
            errors.append(f"MISSING: {ci_path}")
        else:
            ci = _read_json(ci_path)
            col_id = ci.get("collation_id", "")
            if not col_id.startswith("col-"):
                errors.append(f"invalid collation_id: {col_id}")
            if ci.get("schema_version") != "canonical_intents.v1":
                errors.append(f"bad canonical schema: {ci.get('schema_version')}")
            if run_id and ci.get("run_id") != run_id:
                errors.append(f"canonical_intents run_id mismatch: {ci.get('run_id')} != {run_id}")
            if asof_date and ci.get("asof_date") != asof_date:
                errors.append(f"canonical_intents asof_date mismatch: {ci.get('asof_date')} != {asof_date}")

            # Per-sleeve inclusion field present
            for sleeve, ss in ci.get("sleeve_statuses", {}).items():
                if "inclusion" not in ss:
                    warnings.append(f"canonical: {sleeve} missing inclusion field")

            checks.append({
                "check": "canonical_intents",
                "status": "OK" if not errors else "FAIL",
                "collation_id": col_id,
                "total_intents": ci.get("total_intents", 0),
            })

    # ── Stage 2+: Risk report identity ────────────────────────────────
    if stage >= 2:
        rr_path = artifact_root / "reports" / "risk_checks.json"
        if rr_path.exists():
            rr = _read_json(rr_path)
            family = rr.get("input_contract_family", "")
            if family != "canonical_intents":
                warnings.append(f"risk input_family={family}, expected canonical_intents")
            refs = rr.get("input_artifact_refs", {})
            if "collation_id" not in refs:
                warnings.append("risk report missing collation_id in input_artifact_refs")
            checks.append({
                "check": "risk_report_identity",
                "status": "OK",
                "input_family": family,
                "collation_id": refs.get("collation_id", ""),
            })

    # ── Stage 3: Planner identity ─────────────────────────────────────
    if stage >= 3 and (artifact_root / "reports" / "risk_checks.json").exists():
        rr = _read_json(artifact_root / "reports" / "risk_checks.json")
        ri_col = rr.get("input_artifact_refs", {}).get("collation_id", "")
        ci = _read_json(artifact_root / "canonical_intents.json") if (artifact_root / "canonical_intents.json").exists() else {}
        pi_col = ci.get("collation_id", "")
        if ri_col and pi_col and ri_col != pi_col:
            errors.append(f"COLLATION_ID MISMATCH: risk={ri_col} planner={pi_col}")

    # ── Build report ──────────────────────────────────────────────────
    overall = "PASS" if not errors else "FAIL"
    report = {
        "validated_at": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "run_id": run_id,
        "asof_date": asof_date,
        "overall": overall,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "n_checks": len(checks),
        "n_warnings": len(warnings),
        "n_errors": len(errors),
    }

    # Write report
    report_path = artifact_root / "reports" / "canonical_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.validate_canonical_tick <artifact_root> [run_id] [asof_date] [stage]")
        sys.exit(1)

    root = Path(sys.argv[1])
    rid = sys.argv[2] if len(sys.argv) > 2 else ""
    adate = sys.argv[3] if len(sys.argv) > 3 else ""
    stg = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    result = validate_tick(root, rid, adate, stage=stg)

    if result["errors"]:
        print(f"❌ FAIL — {result['n_errors']} errors, {result['n_warnings']} warnings")
        for e in result["errors"]:
            print(f"  ERROR: {e}")
    elif result["warnings"]:
        print(f"⚠️  PASS with {result['n_warnings']} warnings")
        for w in result["warnings"]:
            print(f"  WARN: {w}")
    else:
        print(f"✅ PASS — {result['n_checks']} checks, 0 errors, 0 warnings")

    sys.exit(1 if result["errors"] else 0)
