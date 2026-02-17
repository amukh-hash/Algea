#!/usr/bin/env python
"""Ingest existing report JSON files into the telemetry database.

Scans runs/, data_lake/, and backend/data/ for *report*.json files,
classifies them, extracts time-series and summary metrics, and emits
them via TelemetryEmitter so they appear on the dashboard.

**Family runs**: daily three_sleeve reports are merged into a single
Run (family run) with one metric point per day, producing true curves.

Usage::

    python backend/scripts/ingest_real_reports.py [--dry-run] [--reset]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.api.telemetry_routes import storage
from backend.app.telemetry.emitter import TelemetryEmitter
from backend.app.telemetry.schemas import (
    ArtifactKind,
    EventLevel,
    EventType,
    MetricPoint,
    Run,
    RunStatus,
    RunType,
)

logger = logging.getLogger(__name__)

# ── Classification ──────────────────────────────────────────────────────────

REPORT_PATTERNS = [
    ("three_sleeve_report", RunType.sleeve_paper),
    ("validation_report", RunType.backtest),
    ("risk_calibration_report", RunType.backtest),
    ("selector_full_report", RunType.train),
    ("preflight_report", RunType.backtest),
    ("guard_report", RunType.sleeve_paper),
    ("val_report", RunType.backtest),
]

# Report kinds that use family aggregation (many daily files → one run)
FAMILY_KINDS = {"three_sleeve_report"}


def classify_report(path: Path) -> tuple[str, RunType]:
    """Return (report_kind, run_type) based on filename."""
    name = path.stem.lower()
    for pattern, run_type in REPORT_PATTERNS:
        if pattern in name:
            return pattern, run_type
    return "other", RunType.backtest


def make_run_name(path: Path, kind: str) -> str:
    """Generate a human-readable run name from path and kind."""
    parts = path.relative_to(ROOT).parts
    context_parts = [p for p in parts[:-1] if p not in {"validation", "production_pack", "reports"}]
    context = "/".join(context_parts[-3:])
    kind_label = kind.replace("_", " ").title()
    return f"{kind_label} — {context}"


def idempotency_key(path: Path) -> str:
    """Deterministic run_id so re-running doesn't create duplicates."""
    rel = str(path.relative_to(ROOT))
    return hashlib.sha256(rel.encode()).hexdigest()[:24]


def family_run_id(family_key: str) -> str:
    """Deterministic run_id for a family of reports."""
    return hashlib.sha256(f"family:{family_key}".encode()).hexdigest()[:24]


# ── Timestamp helpers ───────────────────────────────────────────────────────

def _parse_report_timestamp(data: dict[str, Any], path: Path) -> datetime:
    """Extract a timestamp from report content, then path, then file mtime."""
    # 1. Try report-level timestamp field
    if "timestamp" in data and data["timestamp"]:
        try:
            return datetime.fromisoformat(str(data["timestamp"])).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    # 2. Try asof field (date only → noon UTC)
    if "asof" in data and data["asof"]:
        try:
            return datetime.strptime(str(data["asof"]), "%Y-%m-%d").replace(
                hour=12, tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            pass
    # 3. Try path components
    return _timestamp_from_path(path)


def _timestamp_from_path(path: Path) -> datetime:
    """Try to extract a date from path components, fallback to file mtime."""
    for part in path.parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            try:
                return datetime.strptime(part, "%Y-%m-%d").replace(hour=12, tzinfo=timezone.utc)
            except ValueError:
                pass
        if len(part) >= 15 and part[8] == "_":
            try:
                return datetime.strptime(part[:15], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


# ── Three-sleeve family extraction (Section C) ─────────────────────────────

def _extract_three_sleeve_metrics(
    data: dict[str, Any], ts: datetime, run_id: str
) -> list[MetricPoint]:
    """Extract C1 metrics from a single three_sleeve_report.json instance."""
    points: list[MetricPoint] = []
    labels = {
        "mode": str(data.get("mode", "unknown")),
        "vrp_mode": str(data.get("vrp_mode", "unknown")),
        "asof": str(data.get("asof", "")),
    }

    def _emit(key: str, value: float | None, with_labels: bool = True) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        points.append(MetricPoint(
            run_id=run_id, ts=ts, key=key, value=float(value),
            labels=labels if with_labels else {},
        ))

    # Account-level
    acct = data.get("account", {})
    _emit("equity", acct.get("equity"))
    _emit("cash", acct.get("cash"))
    _emit("buying_power", acct.get("buying_power"))

    # Sleeve capital
    sc = data.get("sleeve_capital", {})
    total = 0.0
    for sleeve_name in ("core", "vrp", "selector"):
        val = sc.get(sleeve_name)
        if val is not None:
            _emit(f"sleeve_capital.{sleeve_name}", val, with_labels=False)
            total += float(val)
    if sc:
        _emit("sleeve_capital.total", total, with_labels=False)

    # Per-sleeve state
    sleeves = data.get("sleeves", {})
    for sleeve_name in ("core", "vrp", "selector"):
        sd = sleeves.get(sleeve_name, {})
        if not isinstance(sd, dict):
            continue
        orders = sd.get("orders", [])
        _emit(f"sleeve.{sleeve_name}.orders_count", len(orders) if isinstance(orders, list) else 0, with_labels=False)

    # Selector intents
    sel = sleeves.get("selector", {})
    if isinstance(sel, dict):
        intents = sel.get("intents", [])
        if isinstance(intents, list):
            _emit("sleeve.selector.intents_count", len(intents), with_labels=False)
            notional_sum = sum(
                float(i.get("estimated_notional", 0)) for i in intents if isinstance(i, dict)
            )
            _emit("sleeve.selector.intent_notional_sum", notional_sum, with_labels=False)
            weight_sum = sum(
                abs(float(i.get("weight", 0))) for i in intents if isinstance(i, dict)
            )
            _emit("sleeve.selector.intent_abs_weight_sum", weight_sum, with_labels=False)
        num_longs = sel.get("num_longs")
        if num_longs is not None:
            _emit("sleeve.selector.num_longs", num_longs, with_labels=False)
        num_shorts = sel.get("num_shorts")
        if num_shorts is not None:
            _emit("sleeve.selector.num_shorts", num_shorts, with_labels=False)

    return points


def _extract_three_sleeve_events(
    data: dict[str, Any], ts: datetime, run_id: str
) -> list[tuple[EventLevel, EventType, str, dict[str, Any]]]:
    """Extract C2 events from a single three_sleeve_report.json instance."""
    events: list[tuple[EventLevel, EventType, str, dict[str, Any]]] = []
    asof = str(data.get("asof", ""))
    mode = str(data.get("mode", ""))
    vrp_mode = str(data.get("vrp_mode", ""))
    sleeves = data.get("sleeves", {})

    # Compact selector intent summary
    sel = sleeves.get("selector", {}) if isinstance(sleeves, dict) else {}
    intents = sel.get("intents", []) if isinstance(sel, dict) else []
    intent_summary = {}
    if isinstance(intents, list) and intents:
        longs = sorted(
            [i for i in intents if isinstance(i, dict) and i.get("side") == "buy"],
            key=lambda x: -abs(float(x.get("weight", 0))),
        )[:5]
        shorts = sorted(
            [i for i in intents if isinstance(i, dict) and i.get("side") == "sell"],
            key=lambda x: -abs(float(x.get("weight", 0))),
        )[:5]
        intent_summary = {
            "top_longs": [{"sym": i.get("symbol"), "w": i.get("weight"), "n": i.get("estimated_notional")} for i in longs],
            "top_shorts": [{"sym": i.get("symbol"), "w": i.get("weight"), "n": i.get("estimated_notional")} for i in shorts],
            "total_intents": len(intents),
            "notional_sum": sum(float(i.get("estimated_notional", 0)) for i in intents if isinstance(i, dict)),
        }

    # Sleeve statuses
    sleeve_statuses = {}
    for name, sd in (sleeves.items() if isinstance(sleeves, dict) else []):
        if isinstance(sd, dict):
            sleeve_statuses[name] = sd.get("status", "unknown")

    # DECISION_MADE
    events.append((
        EventLevel.info,
        EventType.DECISION_MADE,
        f"[{asof}] mode={mode} vrp_mode={vrp_mode} statuses={sleeve_statuses}",
        {
            "asof": asof,
            "mode": mode,
            "vrp_mode": vrp_mode,
            "sleeve_capital": data.get("sleeve_capital", {}),
            "sleeve_statuses": sleeve_statuses,
            "selector_intents": intent_summary,
        },
    ))

    # RISK warnings
    for name, sd in (sleeves.items() if isinstance(sleeves, dict) else []):
        if isinstance(sd, dict):
            if sd.get("status") == "inputs_missing":
                events.append((
                    EventLevel.warn,
                    EventType.RISK_LIMIT,
                    f"[{asof}] Sleeve '{name}' has inputs_missing",
                    {"sleeve": name, "status": "inputs_missing", "asof": asof},
                ))

    # VRP sizing warning
    vrp_data = sleeves.get("vrp", {}) if isinstance(sleeves, dict) else {}
    if isinstance(vrp_data, dict) and vrp_data.get("sizing_warning"):
        events.append((
            EventLevel.warn,
            EventType.RISK_LIMIT,
            f"[{asof}] VRP sizing warning: {vrp_data['sizing_warning']}",
            {"sleeve": "vrp", "sizing_warning": vrp_data["sizing_warning"], "asof": asof},
        ))

    # Submitted but empty orders
    for name, sd in (sleeves.items() if isinstance(sleeves, dict) else []):
        if isinstance(sd, dict):
            if sd.get("status") == "submitted" and not sd.get("orders"):
                events.append((
                    EventLevel.info,
                    EventType.ORDER_SUBMITTED,
                    f"[{asof}] Sleeve '{name}' submitted but fills not yet written back",
                    {"sleeve": name, "asof": asof, "mode": sd.get("mode", "")},
                ))

    return events


# ── Generic series detection (Section D) ────────────────────────────────────

MAX_KEYS_PER_REPORT = 50

def _detect_generic_series(
    data: dict[str, Any], kind: str, base_ts: datetime, run_id: str
) -> list[MetricPoint]:
    """Best-effort detection of embedded time-series arrays in any report JSON."""
    points: list[MetricPoint] = []
    keys_emitted: set[str] = set()
    prefix = kind.split("_")[0] if "_" in kind else kind[:4]  # e.g. "val", "risk", "sel"

    def _add_series(key: str, ts_values: list[tuple[datetime, float]]) -> None:
        nonlocal keys_emitted
        namespaced = f"{prefix}.{key}"
        if namespaced in keys_emitted or len(keys_emitted) >= MAX_KEYS_PER_REPORT:
            return
        keys_emitted.add(namespaced)
        for ts, val in ts_values:
            if not math.isnan(val) and not math.isinf(val):
                points.append(MetricPoint(run_id=run_id, ts=ts, key=namespaced, value=val, labels={}))

    def _scan_obj(obj: dict[str, Any], depth: int = 0) -> None:
        if depth > 3 or len(keys_emitted) >= MAX_KEYS_PER_REPORT:
            return

        # Pattern 1: Parallel arrays {timestamps: [...], <name>: [...]}
        ts_keys = [k for k in obj if k.lower() in ("timestamps", "dates", "ts", "time", "date", "index")]
        if ts_keys:
            ts_array = obj[ts_keys[0]]
            if isinstance(ts_array, list) and len(ts_array) >= 2:
                parsed_ts = _parse_ts_array(ts_array, base_ts)
                if parsed_ts:
                    for name, values in obj.items():
                        if name == ts_keys[0]:
                            continue
                        if isinstance(values, list) and len(values) == len(parsed_ts):
                            if all(isinstance(v, (int, float)) for v in values[:10]):
                                _add_series(name, list(zip(parsed_ts, [float(v) for v in values])))

        # Pattern 2: List of objects with ts/date + numeric fields
        for name, val in obj.items():
            if not isinstance(val, list) or len(val) < 2:
                continue
            if all(isinstance(item, dict) for item in val[:5]):
                # Check first item for timestamp-like key
                first = val[0]
                ts_field = None
                for candidate in ("ts", "timestamp", "date", "time", "step", "epoch"):
                    if candidate in first:
                        ts_field = candidate
                        break
                if ts_field:
                    numeric_keys = [
                        k for k in first
                        if k != ts_field and isinstance(first.get(k), (int, float))
                    ][:10]
                    for nk in numeric_keys:
                        ts_values: list[tuple[datetime, float]] = []
                        for item in val:
                            if not isinstance(item, dict):
                                continue
                            ts_val = _parse_single_ts(item.get(ts_field), base_ts, len(ts_values))
                            num_val = item.get(nk)
                            if ts_val and isinstance(num_val, (int, float)):
                                ts_values.append((ts_val, float(num_val)))
                        if len(ts_values) >= 2:
                            series_name = f"{name}.{nk}" if name != "data" else nk
                            _add_series(series_name, ts_values)

            # Recurse into nested dicts
            elif isinstance(val, dict):
                _scan_obj(val, depth + 1)

        # Also recurse into dict children
        for name, val in obj.items():
            if isinstance(val, dict) and depth < 3:
                _scan_obj(val, depth + 1)

    _scan_obj(data)
    return points


def _parse_ts_array(arr: list, base_ts: datetime) -> list[datetime] | None:
    """Parse an array of timestamp-like values."""
    results: list[datetime] = []
    for i, v in enumerate(arr):
        ts = _parse_single_ts(v, base_ts, i)
        if ts:
            results.append(ts)
        else:
            return None
    return results


def _parse_single_ts(v: Any, base_ts: datetime, index: int) -> datetime | None:
    """Parse a single timestamp value."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        # Epoch seconds or step index
        if v > 1e9:  # epoch seconds
            return datetime.fromtimestamp(v, tz=timezone.utc)
        else:  # step index
            return base_ts.replace(second=0, microsecond=0).replace(
                second=min(59, int(v) % 60),
                minute=min(59, int(v) // 60 % 60),
                hour=min(23, int(v) // 3600 % 24),
            )
    if isinstance(v, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(v).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    return None


# ── Existing time-series extraction (kept for backcompat) ───────────────────

def _extract_time_series(data: dict[str, Any], base_ts: datetime) -> list[tuple[str, list[tuple[datetime, float]]]]:
    """Look for known array-like time-series in the report."""
    series: list[tuple[str, list[tuple[datetime, float]]]] = []

    if "equity_curve" in data:
        ec = data["equity_curve"]
        if isinstance(ec, list) and len(ec) > 0:
            points = []
            for item in ec:
                if isinstance(item, dict):
                    ts_val = item.get("ts") or item.get("timestamp") or item.get("date")
                    val = item.get("equity") or item.get("value") or item.get("pnl")
                    if ts_val is not None and val is not None:
                        try:
                            ts = datetime.fromisoformat(str(ts_val)).replace(tzinfo=timezone.utc) if isinstance(ts_val, str) else datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
                            points.append((ts, float(val)))
                        except (ValueError, TypeError):
                            pass
            if points:
                series.append(("equity", points))

    if "loss_history" in data:
        lh = data["loss_history"]
        if isinstance(lh, list) and len(lh) > 0:
            train_points, val_points = [], []
            for i, item in enumerate(lh):
                if isinstance(item, dict):
                    ts_val = item.get("ts") or item.get("timestamp")
                    if ts_val:
                        try:
                            ts = datetime.fromisoformat(str(ts_val)).replace(tzinfo=timezone.utc)
                        except (ValueError, TypeError):
                            ts = datetime(2024, 1, 1, tzinfo=timezone.utc).replace(second=i)
                    else:
                        ts = datetime(2024, 1, 1, tzinfo=timezone.utc).replace(second=i)
                    if "train_loss" in item:
                        train_points.append((ts, float(item["train_loss"])))
                    if "val_loss" in item:
                        val_points.append((ts, float(item["val_loss"])))
            if train_points:
                series.append(("train_loss", train_points))
            if val_points:
                series.append(("val_loss", val_points))

    if "exposure_series" in data:
        es = data["exposure_series"]
        if isinstance(es, list) and len(es) > 0:
            gross_pts, net_pts = [], []
            for item in es:
                if isinstance(item, dict):
                    ts_val = item.get("ts") or item.get("timestamp") or item.get("date")
                    if ts_val:
                        try:
                            ts = datetime.fromisoformat(str(ts_val)).replace(tzinfo=timezone.utc)
                        except (ValueError, TypeError):
                            continue
                        if "gross" in item:
                            gross_pts.append((ts, float(item["gross"])))
                        if "net" in item:
                            net_pts.append((ts, float(item["net"])))
            if gross_pts:
                series.append(("gross_exposure", gross_pts))
            if net_pts:
                series.append(("net_exposure", net_pts))

    return series


def _extract_summary_metrics(data: dict[str, Any], kind: str) -> dict[str, float]:
    """Extract scalar summary metrics from the report content."""
    metrics: dict[str, float] = {}

    if "model_ic" in data:
        metrics["model_ic"] = float(data["model_ic"])
    if "baseline_ic" in data:
        metrics["baseline_ic"] = float(data["baseline_ic"])
    if "all_passed" in data:
        metrics["all_passed"] = 1.0 if data["all_passed"] else 0.0
    if "spearman_correlation" in data:
        metrics["spearman_correlation"] = float(data["spearman_correlation"])
    if "spearman_pvalue" in data:
        metrics["spearman_pvalue"] = float(data["spearman_pvalue"])

    # Three-sleeve (kept for per-file mode fallback)
    if kind != "three_sleeve_report":
        if "account" in data and isinstance(data["account"], dict):
            for k in ("equity", "cash", "buying_power"):
                if k in data["account"]:
                    metrics[f"account_{k}"] = float(data["account"][k])
        if "sleeve_capital" in data and isinstance(data["sleeve_capital"], dict):
            for sleeve, cap in data["sleeve_capital"].items():
                metrics[f"capital_{sleeve}"] = float(cap)

    if "accuracy" in data:
        metrics["accuracy"] = float(data["accuracy"])
    if "auc" in data:
        metrics["auc"] = float(data["auc"])

    return metrics


def _extract_events(data: dict[str, Any], kind: str) -> list[tuple[EventLevel, EventType, str, dict]]:
    """Extract notable events from report content."""
    events: list[tuple[EventLevel, EventType, str, dict]] = []

    if "gates" in data and isinstance(data["gates"], list):
        for gate in data["gates"]:
            if isinstance(gate, dict):
                name = gate.get("name", "unknown")
                passed = gate.get("passed", False)
                detail = gate.get("detail", "")
                level = EventLevel.info if passed else EventLevel.warn
                etype = EventType.EVAL_COMPLETE if passed else EventType.GATE_TRIPPED
                events.append((level, etype, f"Gate '{name}': {'PASS' if passed else 'FAIL'} — {detail}", gate))

    # Non-family sleeve events (per-file mode only)
    if kind != "three_sleeve_report" and "sleeves" in data and isinstance(data["sleeves"], dict):
        for sleeve_name, sleeve_data in data["sleeves"].items():
            if isinstance(sleeve_data, dict):
                status = sleeve_data.get("status", "unknown")
                msg = sleeve_data.get("message", "")
                events.append((
                    EventLevel.info,
                    EventType.ORDER_SUBMITTED if status == "submitted" else EventType.DECISION_MADE,
                    f"Sleeve '{sleeve_name}': {status} — {msg[:200]}",
                    {"sleeve": sleeve_name, "status": status},
                ))

    return events


# ── Report discovery ────────────────────────────────────────────────────────

def discover_reports(root: Path) -> list[Path]:
    """Find all *report*.json files under known directories."""
    dirs = [root / "runs", root / "data_lake", root / "backend" / "data", root / "backend" / "reports"]
    top_level = list(root.glob("*report*.json"))
    reports: list[Path] = list(top_level)
    for d in dirs:
        if d.exists():
            reports.extend(d.rglob("*report*.json"))
    seen: set[str] = set()
    unique: list[Path] = []
    for p in reports:
        key = str(p.resolve())
        # Skip files inside backend/artifacts (those are copies)
        if "artifacts" in p.parts:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique)


# ── Family run ingestion ────────────────────────────────────────────────────

def ingest_family(
    emitter: TelemetryEmitter,
    family_key: str,
    kind: str,
    run_type: RunType,
    reports: list[tuple[Path, dict[str, Any]]],
    dry_run: bool = False,
) -> str | None:
    """Ingest a group of daily reports into a single family run."""
    run_id = family_run_id(family_key)

    if dry_run:
        logger.info(
            "[DRY RUN] Would create family run '%s' (%s) with %d daily reports",
            family_key, run_id[:12], len(reports),
        )
        for path, data in reports:
            asof = data.get("asof", "?")
            logger.info("  └─ %s (asof=%s)", path.name, asof)
        return run_id

    # Sort reports by timestamp
    reports_with_ts = []
    for path, data in reports:
        ts = _parse_report_timestamp(data, path)
        reports_with_ts.append((path, data, ts))
    reports_with_ts.sort(key=lambda x: x[2])

    # Create/update the family run
    earliest_ts = reports_with_ts[0][2]
    latest_ts = reports_with_ts[-1][2]

    run = Run(
        run_id=run_id,
        run_type=run_type,
        name=f"Three Sleeve (Paper) — Family",
        sleeve_name="three_sleeve",
        status=RunStatus.completed,
        started_at=earliest_ts,
        ended_at=latest_ts,
        git_sha="historical",
        config_hash=hashlib.sha256(family_key.encode()).hexdigest()[:12],
        data_version="ingested",
        tags=[kind, "ingested", "family"],
        meta={
            "family_key": family_key,
            "source_type": "report_ingest",
            "family_members": len(reports),
        },
    )
    storage.upsert_run(run)

    # Collect all metric points and events
    all_points: list[MetricPoint] = []
    all_events: list[tuple[datetime, EventLevel, EventType, str, dict[str, Any]]] = []
    artifact_count = 0

    for path, data, ts in reports_with_ts:
        # C1: Metrics
        points = _extract_three_sleeve_metrics(data, ts, run_id)
        all_points.extend(points)

        # C2: Events
        events = _extract_three_sleeve_events(data, ts, run_id)
        for level, etype, msg, payload in events:
            all_events.append((ts, level, etype, msg, payload))

        # C3: Artifact
        asof = data.get("asof", ts.strftime("%Y-%m-%d"))
        emitter.register_artifact(
            run_id, str(path), kind="report", mime="application/json",
            meta={"asof": asof},
        )
        artifact_count += 1

    # Batch insert metrics (dedup via INSERT OR IGNORE)
    inserted = storage.insert_metrics_batch(all_points)

    # Insert events
    for ts, level, etype, msg, payload in all_events:
        emitter.emit_event(run_id, level, etype, msg, ts=ts, payload=payload)

    logger.info(
        "Family '%s': %d daily reports → %d metric points (%d new), %d events, %d artifacts",
        family_key, len(reports), len(all_points), inserted, len(all_events), artifact_count,
    )
    return run_id


# ── Per-file ingestion (non-family) ────────────────────────────────────────

def ingest_report(emitter: TelemetryEmitter, path: Path, dry_run: bool = False) -> str | None:
    """Ingest a single report file. Returns run_id or None if skipped."""
    kind, run_type = classify_report(path)
    run_id = idempotency_key(path)

    existing = storage.get_run(run_id)
    if existing:
        logger.debug("Skipping already-ingested: %s", path)
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("Skipping unreadable file %s: %s", path, e)
        return None

    run_name = make_run_name(path, kind)
    base_ts = _parse_report_timestamp(data, path)

    if dry_run:
        logger.info("[DRY RUN] Would ingest: %s → %s (%s)", path.name, run_name, run_type.value)
        return run_id

    run = Run(
        run_id=run_id,
        run_type=run_type,
        name=run_name,
        sleeve_name=data.get("sleeve_name") or (kind if "sleeve" in kind else None),
        status=RunStatus.starting,
        started_at=base_ts,
        git_sha="historical",
        config_hash=hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12],
        data_version="ingested",
        tags=[kind, "ingested"],
        meta={"source_path": str(path.relative_to(ROOT)), "report_kind": kind, "source_type": "report_ingest"},
    )
    storage.upsert_run(run)
    emitter.set_status(run_id, RunStatus.running)

    # Known time-series
    ts_series = _extract_time_series(data, base_ts)
    ts_count = 0
    batch_points: list[MetricPoint] = []
    for key, ts_points in ts_series:
        for ts, value in ts_points:
            batch_points.append(MetricPoint(run_id=run_id, ts=ts, key=key, value=value, labels={}))
            ts_count += 1

    # Generic series detection (D1)
    generic_points = _detect_generic_series(data, kind, base_ts, run_id)
    batch_points.extend(generic_points)

    # Summary metrics at base_ts
    summary = _extract_summary_metrics(data, kind)
    for key, value in summary.items():
        batch_points.append(MetricPoint(run_id=run_id, ts=base_ts, key=key, value=value, labels={}))

    # Batch insert all metrics
    inserted = storage.insert_metrics_batch(batch_points)

    # Events
    events = _extract_events(data, kind)
    for level, etype, msg, payload in events:
        emitter.emit_event(run_id, level, etype, msg, ts=base_ts, payload=payload)

    # Artifact
    emitter.register_artifact(run_id, str(path), kind="report", mime="application/json")

    emitter.set_status(run_id, RunStatus.completed)

    logger.info(
        "Ingested: %s → %s (%d metric points, %d new, %d events)",
        path.name, run_name, len(batch_points), inserted, len(events),
    )
    return run_id


# ── Reset ───────────────────────────────────────────────────────────────────

def reset_ingested(emitter: TelemetryEmitter) -> int:
    """Delete all runs tagged 'ingested' and their associated data."""
    runs, total = storage.list_runs({}, limit=10000, offset=0)
    deleted = 0
    for run in runs:
        if "ingested" in run.tags:
            storage.delete_run_cascade(run.run_id)
            deleted += 1
    logger.info("Reset: deleted %d ingested runs", deleted)
    return deleted


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Ingest real reports into telemetry DB")
    parser.add_argument("--dry-run", action="store_true", help="List reports without ingesting")
    parser.add_argument("--reset", action="store_true", help="Delete all ingested data first")
    args = parser.parse_args()

    if args.reset and not args.dry_run:
        emitter = TelemetryEmitter(storage)
        reset_ingested(emitter)

    reports = discover_reports(ROOT)
    logger.info("Discovered %d report files", len(reports))

    # Classify and group
    family_groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    per_file_reports: list[Path] = []

    for path in reports:
        kind, run_type = classify_report(path)
        if kind in FAMILY_KINDS:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                family_groups[kind].append((path, data))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning("Skipping unreadable family file %s: %s", path, e)
        else:
            per_file_reports.append(path)

    emitter = TelemetryEmitter(storage)
    ingested = 0
    skipped = 0

    # Ingest family groups
    for fam_kind, members in family_groups.items():
        fam_run_type = dict(REPORT_PATTERNS).get(fam_kind, RunType.backtest)
        result = ingest_family(
            emitter, family_key=fam_kind.replace("_report", ""),
            kind=fam_kind, run_type=fam_run_type,
            reports=members, dry_run=args.dry_run,
        )
        if result:
            ingested += 1

    # Ingest per-file reports
    for path in per_file_reports:
        result = ingest_report(emitter, path, dry_run=args.dry_run)
        if result:
            ingested += 1
        else:
            skipped += 1

    logger.info("Done: %d ingested (%d families), %d skipped", ingested, len(family_groups), skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
