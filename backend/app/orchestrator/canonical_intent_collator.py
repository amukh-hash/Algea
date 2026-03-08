"""Canonical intent collator for Phase 4 + 4.1 hardening.

Reads all ``SleeveDecision`` artifacts from ``sleeves/<name>/decision.json``,
flattens them into a single canonical intent set, and writes
``canonical_intents.json`` to the artifact root.

The collated intent set is the single source of truth for both
``handle_risk_checks_global`` (via ``FF_CANONICAL_RISK_ENGINE``) and
``handle_order_build_and_route`` (via ``FF_CANONICAL_PLANNER``).

This guarantees the **risk == planner invariant**: both consumers see
the exact same intent set, identified by the same ``collation_id``.

## Inclusion policy (Mod 1)

For each sleeve decision:

- ``OK``       → include intents; sleeve counts as successfully evaluated
- ``HALTED``   → include no intents; sleeve counts as successfully evaluated
                  (intentional no-trade; does NOT block collation)
- ``DISABLED`` → include no intents; sleeve excluded by configuration
                  (does NOT affect expected sleeve set; does NOT block)
- ``FAILED``   → hard fail the collation

## `collation_id` stability (Mod 2)

The ``collation_id`` is a deterministic hash over semantic-only fields:

    hash(run_id | sorted(sleeve:symbol:weight:phase:multiplier))

Diagnostic-only fields (generated_by, source_branch, etc.) are excluded
so logically identical intent sets produce identical IDs regardless of
metadata changes. Reordering sleeve reads also produces the same ID.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Mod 2: stable collation_id from semantic-only fields ──────────────


def _intent_sort_key(intent: dict[str, Any]) -> str:
    """Build a stable sort key from semantic-only intent fields."""
    return "|".join([
        str(intent.get("sleeve", "")),
        str(intent.get("symbol", "")),
        str(intent.get("target_weight", 0.0)),
        str(intent.get("execution_phase", "")),
        str(intent.get("multiplier", 1.0)),
    ])


def _compute_collation_id(
    run_id: str,
    intents: list[dict[str, Any]],
) -> str:
    """Deterministic hash over semantic intent fields only.

    Excluded: intent_id, generated_by, source_branch, diagnostics.
    Invariant: reordering sleeve decision reads produces the same ID.
    """
    semantic_keys = sorted(_intent_sort_key(i) for i in intents)
    payload = "|".join([run_id] + semantic_keys)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"col-{digest}"


# ── Main collation ────────────────────────────────────────────────────


def collate_sleeve_decisions(
    artifact_root: Path,
    *,
    expected_sleeves: list[str],
    run_id: str = "",
    asof_date: str = "",
    fail_on_missing: bool = True,
) -> dict[str, Any]:
    """Read all SleeveDecision artifacts and collate into a unified intent set.

    Inclusion policy (Mod 1):
      - OK:       include intents
      - HALTED:   counts as evaluated, 0 intents (intentional no-trade)
      - DISABLED: excluded by config, 0 intents (does not block)
      - FAILED:   hard fail the entire collation

    Returns a dict with collation_id, intents, sleeve_statuses, etc.
    """
    sleeves_dir = artifact_root / "sleeves"
    all_intents: list[dict[str, Any]] = []
    sleeve_statuses: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    disabled: list[str] = []
    halted: list[str] = []
    ok: list[str] = []
    missing: list[str] = []

    for sleeve in expected_sleeves:
        decision_path = sleeves_dir / sleeve / "decision.json"
        if not decision_path.exists():
            missing.append(sleeve)
            sleeve_statuses[sleeve] = {"status": "missing", "n_intents": 0, "inclusion": "blocked"}
            continue

        try:
            decision = _read_json(decision_path)
        except Exception as exc:
            logger.error("Failed to read sleeve decision for %s: %s", sleeve, exc)
            failed.append(sleeve)
            sleeve_statuses[sleeve] = {
                "status": "read_error", "n_intents": 0, "error": str(exc), "inclusion": "failed",
            }
            continue

        status = str(decision.get("status", "failed")).lower()
        intents = decision.get("intents", [])
        if not isinstance(intents, list):
            intents = []

        # ── Mod 1: explicit inclusion semantics ───────────────────
        if status == "ok":
            ok.append(sleeve)
            all_intents.extend(intents)
            inclusion = "included"
        elif status == "halted":
            # Intentional no-trade; successfully evaluated, 0 intents
            halted.append(sleeve)
            inclusion = "evaluated_empty"
        elif status == "disabled":
            # Excluded by config; does not affect expected sleeve set
            disabled.append(sleeve)
            inclusion = "excluded_by_config"
        elif status == "failed":
            failed.append(sleeve)
            inclusion = "failed"
        else:
            # Unknown status → treat as failure
            failed.append(sleeve)
            inclusion = "failed"

        sleeve_statuses[sleeve] = {
            "status": status,
            "n_intents": len(intents) if status == "ok" else 0,
            "inclusion": inclusion,
            "run_id": decision.get("run_id", ""),
            "generated_by": decision.get("generated_by", ""),
            "source_branch": (decision.get("diagnostics") or {}).get("source_branch", ""),
        }

    if fail_on_missing and missing:
        raise RuntimeError(
            f"canonical_intent_collation: missing sleeve decisions for {missing}. "
            "Did the sleeve handlers run with FF_CANONICAL_SLEEVE_OUTPUTS=1?"
        )

    if failed:
        raise RuntimeError(
            f"canonical_intent_collation: failed sleeves {failed}. "
            "Cannot proceed with failed sleeve decisions."
        )

    # ── Mod 2: deterministic collation_id from semantic fields ────
    collation_id = _compute_collation_id(run_id, all_intents)

    return {
        "collation_id": collation_id,
        "schema_version": "canonical_intents.v1",
        "run_id": run_id,
        "asof_date": asof_date,
        "collated_at": datetime.now(timezone.utc).isoformat(),
        "expected_sleeves": expected_sleeves,
        "intents": all_intents,
        "sleeve_statuses": sleeve_statuses,
        "ok_sleeves": ok,
        "failed_sleeves": failed,
        "disabled_sleeves": disabled,
        "halted_sleeves": halted,
        "missing_sleeves": missing,
        "total_intents": len(all_intents),
        "inclusion_policy": {
            "ok": "include_intents",
            "halted": "evaluated_empty",
            "disabled": "excluded_by_config",
            "failed": "hard_fail",
            "missing": "hard_fail_if_expected",
        },
    }


def write_canonical_intents(
    collation: dict[str, Any],
    artifact_root: Path,
) -> Path:
    """Write the collated canonical intents to canonical_intents.json."""
    intents_path = artifact_root / "canonical_intents.json"
    intents_path.parent.mkdir(parents=True, exist_ok=True)
    intents_path.write_text(
        json.dumps(collation, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return intents_path


# ── Mod 5: planner-side lineage validation ────────────────────────────


def validate_canonical_intents_freshness(
    collation: dict[str, Any],
    *,
    expected_run_id: str,
    expected_asof_date: str,
) -> None:
    """Validate that canonical_intents.json is fresh for the current tick.

    Raises RuntimeError if:
      - run_id mismatch (stale from prior tick)
      - asof_date mismatch (stale from prior day)
      - schema_version unrecognized
    """
    schema = collation.get("schema_version", "")
    if schema and schema != "canonical_intents.v1":
        raise RuntimeError(
            f"canonical planner: unexpected schema_version={schema}, "
            f"expected canonical_intents.v1"
        )

    col_run_id = str(collation.get("run_id", ""))
    if expected_run_id and col_run_id and col_run_id != expected_run_id:
        raise RuntimeError(
            f"canonical planner: stale canonical_intents.json — "
            f"run_id={col_run_id} ≠ expected={expected_run_id}"
        )

    col_asof = str(collation.get("asof_date", ""))
    if expected_asof_date and col_asof and col_asof != expected_asof_date:
        raise RuntimeError(
            f"canonical planner: stale canonical_intents.json — "
            f"asof_date={col_asof} ≠ expected={expected_asof_date}"
        )


# ── Shared aggregation (unchanged) ───────────────────────────────────


def intents_to_combined_weights(
    intents: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate canonical intents into per-symbol combined weights.

    This is the core "intent → portfolio" transformation used by both
    risk checks and order planning.
    """
    combined: dict[str, float] = {}
    for intent in intents:
        symbol = str(intent.get("symbol", "")).strip()
        if not symbol:
            continue
        weight = float(intent.get("target_weight", 0.0))
        combined[symbol] = combined.get(symbol, 0.0) + weight
    return combined


def intents_to_per_sleeve_metrics(
    intents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-sleeve gross/net/symbols from canonical intents."""
    per_sleeve: dict[str, dict[str, Any]] = {}
    for intent in intents:
        sleeve = str(intent.get("sleeve", "unknown"))
        weight = float(intent.get("target_weight", 0.0))
        symbol = str(intent.get("symbol", "")).strip()

        if sleeve not in per_sleeve:
            per_sleeve[sleeve] = {"gross": 0.0, "net": 0.0, "num_symbols": 0, "intent_ids": []}
        per_sleeve[sleeve]["gross"] += abs(weight)
        per_sleeve[sleeve]["net"] += weight
        if symbol and abs(weight) > 0:
            per_sleeve[sleeve]["num_symbols"] += 1
        per_sleeve[sleeve]["intent_ids"].append(str(intent.get("intent_id", "")))

    # Round
    for stats in per_sleeve.values():
        stats["gross"] = round(float(stats["gross"]), 8)
        stats["net"] = round(float(stats["net"]), 8)

    return per_sleeve
