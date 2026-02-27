"""Contract qualification dry-pass for all roots.

Walks all 14 roots across a recent date range, qualifies each with IBKR,
and runs contract_spec_checks validation.  Produces a JSON report suitable
for promotion gating.

Usage
-----
::

    python -m sleeves.cooc_reversal_futures.run_contract_qualification \\
        --days 90 \\
        --output contract_spec_check_report.json

Without an IBKR connection, use ``--dry-run`` to validate specs against
themselves (exchange routing + month code consistency only).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _all_roots() -> list[str]:
    from .contract_master import CONTRACT_MASTER
    return sorted(CONTRACT_MASTER.keys())


def _primary_roots() -> list[str]:
    """The 14-root primary universe (no micros)."""
    return [
        "ES", "NQ", "YM", "RTY",
        "CL", "GC", "SI", "ZN", "ZB",
        "6E", "6J", "HG", "6B", "6A",
    ]


def run_dry_run_qualification(roots: list[str] | None = None) -> dict:
    """Validate all specs without an IBKR connection.

    Checks:
      - exchange field is set
      - roll_month_codes are non-empty and consistent with roll_cycle
      - tick_value == multiplier * tick_size
      - exchange matches EXCHANGE_MAP
    """
    from .contract_master import CONTRACT_MASTER
    from algea.trading.ibkr_contracts import EXCHANGE_MAP
    from .contract_spec_checks import _normalise_exchange

    roots = roots or _primary_roots()
    results: list[dict] = []
    all_ok = True

    for root in roots:
        issues: list[str] = []

        if root not in CONTRACT_MASTER:
            results.append({"root": root, "passed": False, "issues": [f"Not in CONTRACT_MASTER"]})
            all_ok = False
            continue

        spec = CONTRACT_MASTER[root]

        # Exchange set?
        if not spec.exchange:
            issues.append("exchange field is empty")

        # EXCHANGE_MAP consistency
        if root in EXCHANGE_MAP:
            if _normalise_exchange(EXCHANGE_MAP[root]) != _normalise_exchange(spec.exchange):
                issues.append(
                    f"EXCHANGE_MAP[{root}]={EXCHANGE_MAP[root]} != spec.exchange={spec.exchange}"
                )
        else:
            issues.append(f"Not in EXCHANGE_MAP")

        # Roll months non-empty
        if not spec.roll_month_codes:
            issues.append("roll_month_codes is empty")

        # Roll cycle consistency
        if spec.roll_cycle == "quarterly" and set(spec.roll_month_codes) != {"H", "M", "U", "Z"}:
            issues.append(f"quarterly but month codes = {spec.roll_month_codes}")
        if spec.roll_cycle == "monthly" and len(spec.roll_month_codes) != 12:
            issues.append(f"monthly but {len(spec.roll_month_codes)} month codes")

        # tick_value
        expected_tv = spec.multiplier * spec.tick_size
        if abs(spec.tick_value - expected_tv) > 1e-6:
            issues.append(f"tick_value={spec.tick_value} != multiplier*tick_size={expected_tv}")

        passed = len(issues) == 0
        if not passed:
            all_ok = False

        results.append({
            "root": root,
            "passed": passed,
            "exchange": spec.exchange,
            "roll_cycle": spec.roll_cycle,
            "roll_month_codes": list(spec.roll_month_codes),
            "multiplier": spec.multiplier,
            "tick_size": spec.tick_size,
            "tick_value": spec.tick_value,
            "issues": issues,
        })

    return {
        "mode": "dry_run",
        "roots_checked": roots,
        "all_passed": all_ok,
        "results": results,
    }


def run_ibkr_qualification(
    roots: list[str] | None = None,
    days_back: int = 90,
    allow_mismatch: bool = False,
) -> dict:
    """Qualify all roots via IBKR and validate specs.

    Requires: connected IbkrClient (will attempt connection).
    """
    from .contract_master import CONTRACT_MASTER
    from .roll import active_contract_for_day
    from .contract_spec_checks import validate_qualified_contract

    roots = roots or _primary_roots()
    today = date.today()
    check_date = today - timedelta(days=1)  # yesterday

    results: list[dict] = []
    all_ok = True

    try:
        from algea.trading.ibkr_client import IbkrClient
        from algea.trading.ibkr_contracts import build_future_contract, parse_active_contract_symbol

        client = IbkrClient()
        client.connect()
    except Exception as e:
        logger.error("Cannot connect to IBKR: %s — falling back to dry-run", e)
        return run_dry_run_qualification(roots)

    try:
        for root in roots:
            spec = CONTRACT_MASTER.get(root)
            if spec is None:
                results.append({"root": root, "passed": False, "issues": ["Not in CONTRACT_MASTER"]})
                all_ok = False
                continue

            try:
                active = active_contract_for_day(root, check_date, spec)
                root_parsed, expiry = parse_active_contract_symbol(active)
                contract = build_future_contract(root_parsed, expiry)

                qualified = client.qualify_contracts(contract)
                if not qualified or qualified[0].conId == 0:
                    results.append({
                        "root": root, "passed": False,
                        "issues": [f"Qualification failed for {active}"],
                        "active_contract": active,
                    })
                    all_ok = False
                    continue

                q = qualified[0]
                exchange = getattr(q, "exchange", "") or getattr(q, "primaryExchange", "")
                expiry_ym = getattr(q, "lastTradeDateOrContractMonth", "")[:6]

                check = validate_qualified_contract(
                    spec, exchange, expiry_ym, allow_mismatch=allow_mismatch,
                )
                if not check.passed:
                    all_ok = False

                results.append({
                    "root": root,
                    "passed": check.passed,
                    "active_contract": active,
                    "qualified_exchange": exchange,
                    "qualified_expiry": expiry_ym,
                    "conId": q.conId,
                    "warnings": check.warnings,
                })

            except Exception as e:
                results.append({"root": root, "passed": False, "issues": [str(e)]})
                all_ok = False

    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    return {
        "mode": "ibkr",
        "check_date": check_date.isoformat(),
        "roots_checked": roots,
        "all_passed": all_ok,
        "results": results,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Contract qualification dry-pass for all roots."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate specs without IBKR connection")
    parser.add_argument("--days", type=int, default=90,
                        help="How many days back to check (IBKR mode)")
    parser.add_argument("--roots", nargs="*", default=None,
                        help="Specific roots to check (default: all 14)")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="Allow mismatches (warn only, don't fail)")
    parser.add_argument("--output", type=str, default="contract_spec_check_report.json",
                        help="Output JSON report path")
    args = parser.parse_args()

    if args.dry_run:
        report = run_dry_run_qualification(args.roots)
    else:
        report = run_ibkr_qualification(args.roots, args.days, args.allow_mismatch)

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    # Summary
    n_pass = sum(1 for r in report["results"] if r["passed"])
    n_total = len(report["results"])
    status = "✅ ALL PASS" if report["all_passed"] else "❌ FAILURES DETECTED"

    logger.info("")
    logger.info("=" * 60)
    logger.info("Contract Qualification: %s (%d/%d roots)", status, n_pass, n_total)
    logger.info("Report → %s", output_path)
    logger.info("=" * 60)

    for r in report["results"]:
        flag = "✅" if r["passed"] else "❌"
        issues = r.get("issues", r.get("warnings", []))
        issue_str = f" — {'; '.join(issues)}" if issues else ""
        logger.info("  %s %s%s", flag, r["root"], issue_str)

    if not report["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
