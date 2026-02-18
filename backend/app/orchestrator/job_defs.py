from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .calendar import Session

logger = logging.getLogger(__name__)

JobHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Job:
    name: str
    sessions: set[Session]
    deps: list[str]
    mode_allow: set[str]
    timeout_s: int
    retries: int
    handler: JobHandler
    min_interval_s: int = 0


def _require_ctx(ctx: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in ctx]
    if missing:
        raise KeyError(f"missing ctx keys: {missing}")


def _asof(ctx: dict[str, Any]) -> str:
    return str(ctx["asof_date"])


def _session(ctx: dict[str, Any]) -> str:
    raw = ctx["session"]
    return raw.value if hasattr(raw, "value") else str(raw)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(ctx: dict[str, Any]) -> dict[str, Path]:
    _require_ctx(ctx, "artifact_root")
    root = Path(str(ctx["artifact_root"]))
    paths = {
        "root": root,
        "signals": root / "signals",
        "targets": root / "targets",
        "orders": root / "orders",
        "fills": root / "fills",
        "reports": root / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _load_config(ctx: dict[str, Any]) -> dict[str, Any]:
    cfg = ctx.get("config")
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return {
        "enabled_jobs": getattr(cfg, "enabled_jobs", []),
        "disabled_jobs": getattr(cfg, "disabled_jobs", []),
        "mode": getattr(cfg, "mode", None),
        "paper_only": getattr(cfg, "paper_only", True),
        "risk_limits": getattr(cfg, "risk_limits", None),
        "account_equity": getattr(cfg, "account_equity", None),
        "order_notional_rounding": getattr(cfg, "order_notional_rounding", None),
        "max_order_notional": getattr(cfg, "max_order_notional", None),
        "max_total_order_notional": getattr(cfg, "max_total_order_notional", None),
        "max_orders": getattr(cfg, "max_orders", None),
    }


def _targets_for(symbols: list[str], gross: float) -> list[dict[str, Any]]:
    if not symbols:
        return []
    w = gross / len(symbols)
    out: list[dict[str, Any]] = []
    for i, s in enumerate(symbols):
        signed = w if i % 2 == 0 else -w
        out.append({"symbol": s, "target_weight": round(signed, 6)})
    return out


def _risk_limits(ctx: dict[str, Any]) -> dict[str, float | int]:
    cfg = _load_config(ctx)
    return {
        "max_gross": float(cfg.get("max_gross", 1.5)),
        "max_net_abs": float(cfg.get("max_net_abs", 0.5)),
        "max_symbol_abs_weight": float(cfg.get("max_symbol_abs_weight", 0.10)),
        "max_symbols": int(cfg.get("max_symbols", 200)),
    }


def _parse_legacy_risk_report(data: dict[str, Any]) -> dict[str, Any]:
    # Legacy shapes:
    # 1) {status, gross_exposure, nan_or_inf, limits}
    # 2) {status=failed, reason, missing_sleeves}
    status = str(data.get("status", "failed"))
    violations: list[dict[str, Any]] = []
    reason = data.get("reason")
    missing = data.get("missing_sleeves", []) if isinstance(data.get("missing_sleeves", []), list) else []
    if missing:
        violations.append({"code": "MISSING_TARGETS", "message": "missing targets", "details": {"missing_sleeves": missing}})
    if data.get("nan_or_inf"):
        violations.append({"code": "NAN_INF", "message": "NaN/Inf detected", "details": {}})
    if status != "ok" and not reason:
        reason = "legacy risk report failure"
    return {
        "status": status,
        "reason": reason,
        "missing_sleeves": missing,
        "violations": violations,
    }


def _canonical_or_legacy_risk(data: dict[str, Any]) -> dict[str, Any]:
    if "violations" in data and "metrics" in data and "limits" in data:
        return data
    return _parse_legacy_risk_report(data)


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def handle_data_refresh_intraday(ctx: dict[str, Any]) -> dict[str, Any]:
    _require_ctx(ctx, "asof_date", "session")
    p = _paths(ctx)
    out = p["signals"] / "data_refresh.json"
    payload = {
        "status": "ok",
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "source": "orchestrator_fallback",
    }
    _write_json(out, payload)
    return {"status": "ok", "artifacts": {"data_refresh": str(out)}, "summary": "data refresh recorded"}


def _generic_signal_handler(ctx: dict[str, Any], sleeve: str, symbols: list[str], run_type: str = "premarket") -> dict[str, Any]:
    p = _paths(ctx)
    sig_path = p["signals"] / f"{sleeve}_signals.json"
    tgt_path = p["targets"] / f"{sleeve}_targets.json"

    signal_payload = {
        "status": "ok",
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "sleeve": sleeve,
        "run_type": run_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": [{"symbol": s, "score": 1.0 if i % 2 == 0 else -1.0} for i, s in enumerate(symbols)],
        "stub": True,
        "note": "TODO: wire to production sleeve signal entrypoint",
    }
    target_payload = {
        "status": "ok",
        "asof_date": _asof(ctx),
        "sleeve": sleeve,
        "targets": _targets_for(symbols, gross=0.03),
        "stub": True,
    }
    _write_json(sig_path, signal_payload)
    _write_json(tgt_path, target_payload)
    return {
        "status": "ok",
        "artifacts": {f"{sleeve}_signals": str(sig_path), f"{sleeve}_targets": str(tgt_path)},
        "metrics": {"n_symbols": len(symbols), "run_type": run_type},
    }


def handle_signals_generate_core(ctx: dict[str, Any]) -> dict[str, Any]:
    """Core sleeve: CO->OC reversal futures signals.

    Delegates to ``run_paper_cycle_ibkr.phase_open`` when available.
    Falls back to the stub _generic_signal_handler if strategy imports fail.
    """
    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[3]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from backend.scripts.paper.run_paper_cycle_ibkr import phase_open, _load_config as _load_sleeve_cfg
        from datetime import date

        asof = date.fromisoformat(_asof(ctx))
        # Locate default config & inputs
        config_path = str(_root / "backend" / "configs" / "cooc_reversal_futures.yaml")
        inputs_path = str(_root / "data_cache" / "canonical_futures_daily.parquet")

        mode = ctx.get("mode", "noop")
        if ctx.get("dry_run", True):
            mode = "noop"  # never route orders from signal generation

        phase_open(
            cfg=_load_sleeve_cfg(config_path),
            inputs_path=inputs_path,
            asof=asof,
            mode=mode,
        )

        # Write simplified targets for downstream orchestrator jobs
        p = _paths(ctx)
        tgt_path = p["targets"] / "core_targets.json"
        sig_path = p["signals"] / "core_signals.json"
        _write_json(sig_path, {
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "core", "source": "phase_open",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        # Core sleeve manages its own orders; provide zero-weight targets
        # so downstream risk checks don't flag missing files
        _write_json(tgt_path, {
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "core",
            "targets": [],  # Core manages its own execution
        })

        return {
            "status": "ok",
            "artifacts": {"core_signals": str(sig_path), "core_targets": str(tgt_path)},
            "metrics": {"source": "phase_open"},
        }
    except Exception as exc:
        logger.warning("Core signal handler falling back to stub: %s", exc)
        return _generic_signal_handler(ctx, "core", ["SPY", "QQQ", "IWM"])


def handle_signals_generate_vrp(ctx: dict[str, Any]) -> dict[str, Any]:
    """VRP sleeve: options volatility premium signals.

    Delegates to ``run_three_sleeves_ibkr.run_vrp`` when available.
    Always runs in noop mode to avoid SPX sizing risk.
    Falls back to stub if imports fail.
    """
    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[3]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from scripts.run_three_sleeves_ibkr import run_vrp
        from algaie.trading.broker_ibkr import IBKRLiveBroker
        from datetime import date

        asof = date.fromisoformat(_asof(ctx))

        # VRP sleeve always noop — SPX sizing danger
        # Create a temporary readonly broker for signal generation (no trades)
        try:
            broker = IBKRLiveBroker.from_env()
        except Exception:
            broker = None

        if broker is not None:
            vrp_result = run_vrp(broker, capital=30_000.0, asof=asof, mode="noop")
            try:
                broker._disconnect()
            except Exception:
                pass
        else:
            vrp_result = {"status": "skipped", "reason": "no broker available"}

        p = _paths(ctx)
        sig_path = p["signals"] / "vrp_signals.json"
        tgt_path = p["targets"] / "vrp_targets.json"
        _write_json(sig_path, {
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "vrp", "source": "run_vrp",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "vrp_result": vrp_result,
        })
        # VRP is always noop, provide empty targets
        _write_json(tgt_path, {
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "vrp",
            "targets": [],  # VRP currently noop
        })

        return {
            "status": "ok",
            "artifacts": {"vrp_signals": str(sig_path), "vrp_targets": str(tgt_path)},
            "metrics": {"source": "run_vrp", "mode": "noop"},
        }
    except Exception as exc:
        logger.warning("VRP signal handler falling back to stub: %s", exc)
        return _generic_signal_handler(ctx, "vrp", ["SPY", "TLT"])


def handle_signals_generate_selector(ctx: dict[str, Any]) -> dict[str, Any]:
    """Selector sleeve: equities swing trading signals.

    Delegates to ``run_three_sleeves_ibkr._load_selector_signals`` when available.
    Falls back to stub if imports fail.
    """
    run_type = "intraday" if _session(ctx) == Session.INTRADAY.value else "premarket"
    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[3]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from scripts.run_three_sleeves_ibkr import _load_selector_signals
        from datetime import date

        asof = date.fromisoformat(_asof(ctx))
        signals_df = _load_selector_signals(asof, top_n=10)

        # Convert DataFrame signals to target weights
        targets = []
        if signals_df is not None and len(signals_df) > 0:
            for _, row in signals_df.iterrows():
                targets.append({
                    "symbol": str(row.get("symbol", "")),
                    "target_weight": round(float(row.get("weight", 0.0)), 6),
                    "score": round(float(row.get("score_final", 0.0)), 6),
                    "side": str(row.get("side", "")),
                })

        p = _paths(ctx)
        sig_path = p["signals"] / "selector_signals.json"
        tgt_path = p["targets"] / "selector_targets.json"
        _write_json(sig_path, {
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "selector", "run_type": run_type, "source": "_load_selector_signals",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": targets,
        })
        _write_json(tgt_path, {
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "selector",
            "targets": targets,
        })

        return {
            "status": "ok",
            "artifacts": {"selector_signals": str(sig_path), "selector_targets": str(tgt_path)},
            "metrics": {"n_symbols": len(targets), "run_type": run_type, "source": "_load_selector_signals"},
        }
    except Exception as exc:
        logger.warning("Selector signal handler falling back to stub: %s", exc)
        return _generic_signal_handler(ctx, "selector", ["AAPL", "MSFT", "NVDA"], run_type=run_type)


def _load_targets_required(ctx: dict[str, Any]) -> tuple[dict[str, Path], list[str]]:
    p = _paths(ctx)
    sleeves = ["core", "vrp", "selector"]
    missing: list[str] = []
    target_paths: dict[str, Path] = {}
    for sleeve in sleeves:
        path = p["targets"] / f"{sleeve}_targets.json"
        target_paths[sleeve] = path
        if not path.exists():
            missing.append(sleeve)
    return target_paths, missing


def handle_risk_checks_global(ctx: dict[str, Any]) -> dict[str, Any]:
    p = _paths(ctx)
    report_path = p["reports"] / "risk_checks.json"
    target_paths, missing = _load_targets_required(ctx)
    limits = _risk_limits(ctx)
    checked_at = datetime.now(timezone.utc).isoformat()

    violations: list[dict[str, Any]] = []
    combined: dict[str, float] = {}
    per_sleeve: dict[str, dict[str, Any]] = {}
    nan_or_inf = False

    if missing:
        violations.append(
            {
                "code": "MISSING_TARGETS",
                "message": "One or more sleeve target artifacts are missing",
                "details": {"missing_sleeves": missing},
            }
        )

    for sleeve, tpath in target_paths.items():
        targets = _read_json(tpath).get("targets", []) if tpath.exists() else []
        sleeve_gross = 0.0
        sleeve_net = 0.0
        sleeve_symbols = 0
        for row in targets:
            symbol = str(row.get("symbol", "")).strip()
            w = float(row.get("target_weight", 0.0))
            if w != w or w in (float("inf"), float("-inf")):
                nan_or_inf = True
            if symbol and abs(w) > 0:
                sleeve_symbols += 1
            sleeve_gross += abs(w)
            sleeve_net += w
            if symbol:
                combined[symbol] = combined.get(symbol, 0.0) + w
        per_sleeve[sleeve] = {
            "gross": round(sleeve_gross, 8),
            "net": round(sleeve_net, 8),
            "num_symbols": sleeve_symbols,
        }

    gross = sum(abs(w) for w in combined.values())
    net = sum(combined.values())
    num_symbols = sum(1 for w in combined.values() if abs(w) > 0)

    if nan_or_inf:
        violations.append({"code": "NAN_INF", "message": "NaN or Inf found in target weights", "details": {}})
    if gross > float(limits["max_gross"]):
        violations.append({"code": "GROSS_LIMIT", "message": "Gross exposure exceeds limit", "details": {"gross": gross}})
    if abs(net) > float(limits["max_net_abs"]):
        violations.append({"code": "NET_LIMIT", "message": "Net exposure exceeds limit", "details": {"net": net}})
    violating_symbols = {s: w for s, w in combined.items() if abs(w) > float(limits["max_symbol_abs_weight"])}
    if violating_symbols:
        violations.append(
            {
                "code": "SYMBOL_LIMIT",
                "message": "One or more symbols exceed absolute weight limit",
                "details": {"symbols": violating_symbols},
            }
        )
    if num_symbols > int(limits["max_symbols"]):
        violations.append(
            {
                "code": "TOO_MANY_SYMBOLS",
                "message": "Number of symbols exceeds limit",
                "details": {"num_symbols": num_symbols},
            }
        )

    status = "ok" if not violations else "failed"
    reason = None if status == "ok" else "; ".join(v["code"] for v in violations)
    report = {
        "status": status,
        "checked_at": checked_at,
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "reason": reason,
        "missing_sleeves": missing,
        "inputs": {"target_paths": {k: str(v) for k, v in target_paths.items()}},
        "metrics": {
            "nan_or_inf": nan_or_inf,
            "gross_exposure": round(gross, 8),
            "net_exposure": round(net, 8),
            "num_symbols": num_symbols,
            "per_sleeve": per_sleeve,
        },
        "limits": {
            "max_gross": float(limits["max_gross"]),
            "max_net_abs": float(limits["max_net_abs"]),
            "max_symbol_abs_weight": float(limits["max_symbol_abs_weight"]),
            "max_symbols": int(limits["max_symbols"]),
        },
        "violations": violations,
    }
    _write_json(report_path, report)
    logger.info("risk checks completed status=%s violations=%d", status, len(violations))
    if status != "ok":
        raise RuntimeError(f"risk checks failed: {reason}")
    return {"status": "ok", "artifacts": {"risk_report": str(report_path)}, "metrics": report["metrics"]}


def handle_order_build_and_route(ctx: dict[str, Any]) -> dict[str, Any]:
    _require_ctx(ctx, "mode", "dry_run", "broker")
    p = _paths(ctx)
    report_path = p["reports"] / "risk_checks.json"
    if not report_path.exists():
        raise RuntimeError("cannot route orders: risk report missing")

    risk_raw = _read_json(report_path)
    risk = _canonical_or_legacy_risk(risk_raw)
    if str(risk.get("status", "failed")) != "ok":
        raise RuntimeError(f"cannot route orders: risk report failed ({risk.get('reason')})")
    if isinstance(risk.get("violations"), list) and len(risk.get("violations", [])) > 0:
        raise RuntimeError("cannot route orders: risk violations present")

    target_paths, missing = _load_targets_required(ctx)
    if missing:
        raise RuntimeError(f"cannot build orders: missing targets for {missing}")

    cfg = _load_config(ctx)
    account_equity = _cfg_float(cfg, "account_equity", _cfg_float(cfg, "account_notional", 1_000_000))
    rounding = _cfg_float(cfg, "order_notional_rounding", 100)
    max_order_notional = _cfg_float(cfg, "max_order_notional", 25_000)
    max_total_order_notional = _cfg_float(cfg, "max_total_order_notional", 200_000)
    max_orders = _cfg_int(cfg, "max_orders", 50)
    fallback_price = _cfg_float(cfg, "price_fallback", 100.0)
    is_dry_run = bool(ctx["dry_run"])

    combined: dict[str, float] = {}
    for _, tpath in target_paths.items():
        targets = _read_json(tpath).get("targets", [])
        for row in targets:
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            combined[sym] = combined.get(sym, 0.0) + float(row.get("target_weight", 0.0))

    positions_resp = ctx["broker"].get_positions() if hasattr(ctx["broker"], "get_positions") else {"positions": []}
    positions = positions_resp.get("positions", []) if isinstance(positions_resp, dict) else []
    current_qty: dict[str, float] = {str(pos.get("symbol", "")).strip(): float(pos.get("qty", 0.0)) for pos in positions}

    # --- Structured price resolution ---
    # Priority: 1) broker.get_quote  2) ctx["prices"] (daily close cache)  3) synthetic (dry-run only)
    daily_close_cache: dict[str, float] = {}
    if isinstance(ctx.get("prices"), dict):
        daily_close_cache = {str(k): float(v) for k, v in ctx["prices"].items()}

    broker = ctx["broker"]
    has_get_quote = hasattr(broker, "get_quote")

    resolved_prices: dict[str, tuple[float, str]] = {}  # sym -> (price, source)
    missing_prices: list[str] = []
    used_fallback = False

    for sym in sorted(combined.keys()):
        # Step 1: broker quote
        price: float | None = None
        source = "unknown"
        if has_get_quote:
            try:
                price = broker.get_quote(sym)
            except Exception:
                price = None
            if price is not None:
                source = "broker"

        # Step 2: daily close cache
        if price is None and sym in daily_close_cache:
            price = daily_close_cache[sym]
            source = "daily_close"
            used_fallback = True
            logger.warning("price fallback to daily_close for %s: %.4f", sym, price)

        # Step 3: synthetic (dry-run only) OR hard fail
        if price is None:
            if is_dry_run:
                price = fallback_price
                source = "synthetic"
            else:
                missing_prices.append(sym)
                continue

        resolved_prices[sym] = (price, source)

    # Hard fail on missing prices in non-dry-run
    if missing_prices and not is_dry_run:
        rejected_path = p["orders"] / "rejected.json"
        _write_json(
            rejected_path,
            {
                "status": "failed",
                "reasons": ["missing_price"],
                "missing_symbols": missing_prices,
            },
        )
        raise RuntimeError(f"missing price for {missing_prices} in non-dry-run mode")

    orders: list[dict[str, Any]] = []
    for sym, weight in sorted(combined.items()):
        if sym not in resolved_prices:
            continue
        price, price_source = resolved_prices[sym]
        current_notional = current_qty.get(sym, 0.0) * price
        target_notional = weight * account_equity
        delta_notional = target_notional - current_notional
        if abs(delta_notional) < rounding:
            continue
        qty = int(round(abs(delta_notional) / max(price, 1e-6)))
        if qty < 1:
            qty = 1
        est_notional = float(qty) * price
        orders.append(
            {
                "symbol": sym,
                "qty": qty,
                "side": "BUY" if delta_notional > 0 else "SELL",
                "type": "MKT",
                "tif": "DAY",
                "est_price": round(price, 8),
                "price_source": price_source,
                "est_notional": round(est_notional, 8),
            }
        )

    total_abs_notional = sum(abs(float(o["est_notional"])) for o in orders)
    max_single = max([abs(float(o["est_notional"])) for o in orders], default=0.0)

    orders_path = p["orders"] / "orders.json"
    payload = {
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "mode": ctx["mode"],
        "dry_run": is_dry_run,
        "risk_report_path": str(report_path),
        "source_targets": {k: str(v) for k, v in target_paths.items()},
        "inputs": {
            "account_equity": account_equity,
            "price_fallback": fallback_price,
            "limits": {
                "max_order_notional": max_order_notional,
                "max_total_order_notional": max_total_order_notional,
                "max_orders": max_orders,
                "order_notional_rounding": rounding,
            },
        },
        "orders": orders,
        "summary": {
            "order_count": len(orders),
            "total_abs_notional": round(total_abs_notional, 8),
            "max_single_abs_notional": round(max_single, 8),
            "used_fallback_price": used_fallback,
        },
    }
    _write_json(orders_path, payload)

    reject_reasons: list[str] = []
    if not orders:
        reject_reasons.append("empty_order_list")
    if any(not isinstance(o.get("symbol"), str) or not o.get("symbol", "").strip() for o in orders):
        reject_reasons.append("invalid_symbol")
    if any(int(o.get("qty", 0)) <= 0 for o in orders):
        reject_reasons.append("non_positive_qty")
    if len(orders) > max_orders:
        reject_reasons.append("max_orders_exceeded")
    if any(abs(float(o.get("est_notional", 0.0))) > max_order_notional for o in orders):
        reject_reasons.append("max_order_notional_exceeded")
    if total_abs_notional > max_total_order_notional:
        reject_reasons.append("max_total_order_notional_exceeded")

    if reject_reasons:
        rejected_path = p["orders"] / "rejected.json"
        _write_json(
            rejected_path,
            {
                "status": "failed",
                "reasons": reject_reasons,
                "snapshot": {
                    "order_count": len(orders),
                    "total_notional": round(total_abs_notional, 8),
                    "max_single_notional": round(max_single, 8),
                },
            },
        )
        raise RuntimeError(f"order sanity gate failed: {reject_reasons}")

    routed = False
    route_artifact = None
    if not is_dry_run and ctx["mode"] not in {"noop"}:
        if ctx["mode"] == "paper":
            ctx["broker"].verify_paper()
        broker_resp = ctx["broker"].place_orders(payload)
        routed = True
        route_artifact = p["orders"] / "routed.json"
        _write_json(route_artifact, {"status": "ok", "broker_response": broker_resp})

    artifacts = {"orders": str(orders_path)}
    if route_artifact is not None:
        artifacts["routed"] = str(route_artifact)
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": {"routed": routed, "order_count": len(orders), "total_abs_notional": round(total_abs_notional, 8), "used_fallback_price": used_fallback},
    }


def handle_fills_reconcile(ctx: dict[str, Any]) -> dict[str, Any]:
    _require_ctx(ctx, "broker")
    p = _paths(ctx)
    fills = ctx["broker"].get_fills(None)
    fills_path = p["fills"] / "fills.json"
    pos_path = p["fills"] / "positions.json"
    _write_json(fills_path, fills if isinstance(fills, dict) else {"fills": fills})
    positions = ctx["broker"].get_positions()
    _write_json(pos_path, positions if isinstance(positions, dict) else {"positions": positions})
    return {"status": "ok", "artifacts": {"fills": str(fills_path), "positions": str(pos_path)}}


def handle_eod_reports(ctx: dict[str, Any]) -> dict[str, Any]:
    p = _paths(ctx)
    orders_path = p["orders"] / "orders.json"
    fills_path = p["fills"] / "fills.json"
    risk_path = p["reports"] / "risk_checks.json"

    orders = _read_json(orders_path) if orders_path.exists() else {"orders": []}
    fills = _read_json(fills_path) if fills_path.exists() else {"fills": []}
    risk = _read_json(risk_path) if risk_path.exists() else {"status": "missing"}

    summary = {
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "order_count": len(orders.get("orders", [])),
        "fill_count": len(fills.get("fills", [])) if isinstance(fills.get("fills", []), list) else 0,
        "risk_status": risk.get("status"),
    }
    out = p["reports"] / "eod_summary.json"
    _write_json(out, summary)
    return {"status": "ok", "artifacts": {"eod_summary": str(out)}, "summary": summary}


def default_jobs() -> list[Job]:
    return [
        Job("data_refresh_intraday", {Session.PREMARKET, Session.INTRADAY}, [], {"paper", "live", "noop"}, 120, 1, handle_data_refresh_intraday, min_interval_s=300),
        Job("signals_generate_core", {Session.PREMARKET}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_core),
        Job("signals_generate_vrp", {Session.PREMARKET}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_vrp),
        Job("signals_generate_selector", {Session.PREMARKET, Session.INTRADAY}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_selector, min_interval_s=300),
        Job("risk_checks_global", {Session.PREMARKET, Session.OPEN, Session.PRECLOSE}, ["signals_generate_core", "signals_generate_vrp", "signals_generate_selector"], {"paper", "live", "noop"}, 120, 0, handle_risk_checks_global),
        Job("order_build_and_route", {Session.OPEN}, ["risk_checks_global"], {"paper", "live"}, 120, 0, handle_order_build_and_route),
        Job("fills_reconcile", {Session.INTRADAY, Session.CLOSE}, [], {"paper", "live", "noop"}, 120, 0, handle_fills_reconcile, min_interval_s=300),
        Job("eod_reports", {Session.CLOSE, Session.OVERNIGHT}, ["fills_reconcile"], {"paper", "live", "noop"}, 120, 0, handle_eod_reports),
    ]


def topo_sort(jobs: list[Job]) -> list[Job]:
    by_name = {j.name: j for j in jobs}
    indeg = {j.name: 0 for j in jobs}
    graph: dict[str, list[str]] = {j.name: [] for j in jobs}
    for j in jobs:
        for dep in j.deps:
            if dep in by_name:
                graph[dep].append(j.name)
                indeg[j.name] += 1
    queue = sorted([n for n, d in indeg.items() if d == 0])
    out: list[Job] = []
    while queue:
        cur = queue.pop(0)
        out.append(by_name[cur])
        for nxt in sorted(graph[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(out) != len(jobs):
        raise ValueError("Cycle in job graph")
    return out


def filtered_jobs(all_jobs: list[Job], session: Session, mode: str, enabled: list[str], disabled: list[str]) -> list[Job]:
    result = [j for j in all_jobs if session in j.sessions and mode in j.mode_allow and j.name not in disabled]
    if enabled:
        enabled_set = set(enabled)
        result = [j for j in result if j.name in enabled_set]
    return topo_sort(result)
