from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.schemas import ExecutionPhase, TargetIntent
from .intent_derivation_policy import POLICY_VERSION, resolve_multiplier, resolve_policy

logger = logging.getLogger(__name__)


@dataclass
class TranslationArtifacts:
    translated_intents_path: str
    parity_report_path: str
    n_translated: int


def _to_phase(value: Any, default: ExecutionPhase) -> ExecutionPhase:
    if value is None:
        return default
    if isinstance(value, ExecutionPhase):
        return value
    return ExecutionPhase(str(value))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def translate_targets_to_intents(
    *,
    artifact_root: Path,
    asof_date: str,
    target_paths: dict[str, Path],
    allocator_scales: dict[str, float],
) -> TranslationArtifacts:
    """Translate target artifacts into canonical intents for PR-3 compatibility artifacts.

    This does NOT switch execution to intent-canonical. It only emits translated
    compatibility artifacts plus a parity report for migration visibility.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    intents_dir = artifact_root / "intents"
    reports_dir = artifact_root / "reports"
    translated_path = intents_dir / "translated_from_targets_intents.json"
    parity_path = reports_dir / "intent_translation_parity.json"

    translated_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sleeves_processed: list[str] = []
    native_intents_present: dict[str, bool] = {}
    source_target_counts: dict[str, int] = {}
    translated_counts: dict[str, int] = {}
    scaled_target_sums: dict[str, float] = {}
    translated_weight_sums: dict[str, float] = {}

    for sleeve, tpath in sorted(target_paths.items()):
        sleeves_processed.append(sleeve)
        source_target_counts[sleeve] = 0
        translated_counts[sleeve] = 0
        scaled_target_sums[sleeve] = 0.0
        translated_weight_sums[sleeve] = 0.0
        scale = float(allocator_scales.get(sleeve, 1.0))

        native_intents_path = intents_dir / f"{sleeve}_intents.json"
        native_intents_present[sleeve] = native_intents_path.exists()

        if native_intents_path.exists():
            raw_native = _load_json(native_intents_path)
            native_entries = raw_native if isinstance(raw_native, list) else raw_native.get("intents", [])
            for idx, entry in enumerate(native_entries):
                try:
                    payload = dict(entry)
                    if "asof_date" not in payload:
                        payload["asof_date"] = asof_date
                    ti = TargetIntent(**payload)
                    translated_rows.append(
                        {
                            "intent": ti.model_dump(),
                            "trace": {
                                "source_sleeve": sleeve,
                                "source_artifact_path": str(native_intents_path),
                                "source_row_index": idx,
                                "source_kind": "native_intent",
                                "policy_version": POLICY_VERSION,
                                "allocator_scale_applied": 1.0,
                                "translation_timestamp": now_iso,
                            },
                        }
                    )
                    translated_counts[sleeve] += 1
                    translated_weight_sums[sleeve] += float(ti.target_weight)
                except Exception as exc:
                    failures.append({
                        "sleeve": sleeve,
                        "source_artifact": str(native_intents_path),
                        "row_index": idx,
                        "code": "NATIVE_INTENT_INVALID",
                        "error": str(exc),
                    })
            continue

        raw = _load_json(tpath)
        targets = raw.get("targets", []) if isinstance(raw, dict) else []
        source_target_counts[sleeve] = len(targets)

        if sleeve == "statarb":
            untranslatable = [
                i for i, row in enumerate(targets)
                if not str(row.get("symbol", "")).strip() or "target_weight" not in row
            ]
            if untranslatable:
                msg = (
                    "UNTRANSLATABLE_TARGET: statarb target artifact contains pair-only rows "
                    f"without symbol-level weights at indices={untranslatable}"
                )
                failures.append({
                    "sleeve": sleeve,
                    "source_artifact": str(tpath),
                    "code": "UNTRANSLATABLE_TARGET",
                    "error": msg,
                })
                raise RuntimeError(msg)

        for idx, row in enumerate(targets):
            try:
                symbol = str(row.get("symbol", "")).strip()
                if not symbol:
                    continue
                if "target_weight" not in row:
                    continue
                policy = resolve_policy(sleeve, row)
                scaled_weight = float(row.get("target_weight", 0.0)) * scale
                multiplier = resolve_multiplier(sleeve, symbol, row)
                phase = _to_phase(row.get("execution_phase"), policy.default_execution_phase)
                asset_class = str(row.get("asset_class", policy.asset_class)).upper()

                ti = TargetIntent(
                    asof_date=asof_date,
                    sleeve=sleeve,
                    symbol=symbol,
                    asset_class=asset_class,
                    target_weight=scaled_weight,
                    execution_phase=phase,
                    multiplier=multiplier,
                    dte=int(row.get("dte", -1)),
                )
                translated_rows.append(
                    {
                        "intent": ti.model_dump(),
                        "trace": {
                            "source_sleeve": sleeve,
                            "source_artifact_path": str(tpath),
                            "source_row_index": idx,
                            "source_kind": "target_translation",
                            "policy_version": POLICY_VERSION,
                            "allocator_scale_applied": scale,
                            "translation_timestamp": now_iso,
                        },
                    }
                )
                translated_counts[sleeve] += 1
                scaled_target_sums[sleeve] += scaled_weight
                translated_weight_sums[sleeve] += float(ti.target_weight)
            except Exception as exc:
                failures.append({
                    "sleeve": sleeve,
                    "source_artifact": str(tpath),
                    "row_index": idx,
                    "code": "TRANSLATION_ERROR",
                    "error": str(exc),
                })
                raise

    parity = {
        "status": "ok" if not failures else "failed",
        "generated_at": now_iso,
        "policy_version": POLICY_VERSION,
        "sleeves_processed": sleeves_processed,
        "source_target_counts": source_target_counts,
        "translated_intent_counts": translated_counts,
        "native_intents_present": native_intents_present,
        "allocator_scale_applied": {s: float(allocator_scales.get(s, 1.0)) for s in sleeves_processed},
        "scaled_target_weight_sums": scaled_target_sums,
        "translated_weight_sums": translated_weight_sums,
        "semantic_mismatches": [
            {
                "sleeve": s,
                "code": "WEIGHT_SUM_MISMATCH",
                "scaled_target_sum": scaled_target_sums[s],
                "translated_sum": translated_weight_sums[s],
            }
            for s in sleeves_processed
            if abs(float(scaled_target_sums[s]) - float(translated_weight_sums[s])) > 1e-9
            and not native_intents_present.get(s, False)
        ],
        "translation_failures": failures,
    }

    translated_payload = {
        "schema_version": "translated_intents.v1",
        "asof_date": asof_date,
        "policy_version": POLICY_VERSION,
        "generated_at": now_iso,
        "intents": translated_rows,
    }

    _write_json(translated_path, translated_payload)
    _write_json(parity_path, parity)

    if failures:
        logger.error("Intent translation completed with failures: %s", failures)
    else:
        logger.info("Intent translation completed: %d intents", len(translated_rows))

    return TranslationArtifacts(
        translated_intents_path=str(translated_path),
        parity_report_path=str(parity_path),
        n_translated=len(translated_rows),
    )
