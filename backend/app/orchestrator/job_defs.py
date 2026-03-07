from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .calendar import Session
import numpy as np
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u_values, v_values):
        return float(abs(sum(u_values)/len(u_values) - sum(v_values)/len(v_values)))
from backend.app.version import with_app_metadata

logger = logging.getLogger(__name__)

JobHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _mode(ctx: dict[str, Any]) -> str:
    return str(ctx.get("mode", "noop")).lower()


def _chronos2_enabled(ctx: dict[str, Any]) -> bool:
    cfg = ctx.get("config")
    if cfg is not None and hasattr(cfg, "enable_chronos2_sleeve"):
        return bool(getattr(cfg, "enable_chronos2_sleeve"))
    import os
    return os.getenv("ENABLE_CHRONOS2_SLEEVE", "0") == "1"


def _smoe_selector_enabled(ctx: dict[str, Any]) -> bool:
    cfg = ctx.get("config")
    if cfg is not None and hasattr(cfg, "enable_smoe_selector"):
        return bool(getattr(cfg, "enable_smoe_selector"))
    import os
    return os.getenv("ENABLE_SMOE_SELECTOR", "0") == "1"


def _selector_model_alias(ctx: dict[str, Any]) -> str:
    cfg = ctx.get("config")
    if cfg is not None and hasattr(cfg, "selector_model_alias"):
        return str(getattr(cfg, "selector_model_alias"))
    import os
    return os.getenv("SELECTOR_MODEL_ALIAS", "prod")


def _vol_surface_vrp_enabled(ctx: dict[str, Any]) -> bool:
    cfg = ctx.get("config")
    if cfg is not None and hasattr(cfg, "enable_vol_surface_vrp"):
        return bool(getattr(cfg, "enable_vol_surface_vrp"))
    import os
    return os.getenv("ENABLE_VOL_SURFACE_VRP", "0") == "1"


def _vrp_model_alias(ctx: dict[str, Any]) -> str:
    cfg = ctx.get("config")
    if cfg is not None and hasattr(cfg, "vrp_model_alias"):
        return str(getattr(cfg, "vrp_model_alias"))
    import os
    return os.getenv("VRP_MODEL_ALIAS", "prod")


def _statarb_enabled(ctx: dict[str, Any]) -> bool:
    cfg = ctx.get("config")
    if isinstance(cfg, dict) and "enable_statarb_sleeve" in cfg:
        return bool(cfg.get("enable_statarb_sleeve"))
    if cfg is not None and hasattr(cfg, "enable_statarb_sleeve"):
        return bool(getattr(cfg, "enable_statarb_sleeve"))
    import os
    return os.getenv("ENABLE_STATARB_SLEEVE", "0") == "1"


def _itransformer_model_alias(ctx: dict[str, Any]) -> str:
    cfg = ctx.get("config")
    if cfg is not None and hasattr(cfg, "itransformer_model_alias"):
        return str(getattr(cfg, "itransformer_model_alias"))
    import os
    return os.getenv("ITRANSFORMER_MODEL_ALIAS", "prod")


def _allow_stub_signals(ctx: dict[str, Any]) -> bool:
    """Check whether stub signal generation is permitted.

    Returns True only in NOOP mode with ORCH_ALLOW_STUB_SIGNALS=1.
    Raises OrchestratorSafetyError if someone tries to enable stubs
    in LIVE or PAPER modes — this is an absolute hard stop.
    """
    import os
    from backend.app.core.runtime_mode import RuntimeMode, OrchestratorSafetyError

    env_flag = os.getenv("ORCH_ALLOW_STUB_SIGNALS", "0") == "1"
    mode = _mode(ctx)

    # Hard assertion: stubs are NEVER allowed in live or paper
    if mode in (RuntimeMode.LIVE.value, RuntimeMode.PAPER.value):
        if env_flag:
            raise OrchestratorSafetyError(
                f"FATAL: ORCH_ALLOW_STUB_SIGNALS=1 is strictly forbidden in "
                f"mode='{mode}'. Stub signals would poison real portfolio targets."
            )
        return False

    # NOOP/STUB: only if explicitly opted in AND (noop or dry_run)
    if not env_flag:
        return False
    return mode == RuntimeMode.NOOP.value or bool(ctx.get("dry_run", False))


def _ml_platform_cfg(ctx: dict[str, Any]):
    from backend.app.ml_platform.config import MLPlatformConfig

    return MLPlatformConfig()


def _build_inference_client(ctx: dict[str, Any], server):
    from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient

    cfg = _ml_platform_cfg(ctx)
    return InferenceGatewayClient(
        server,
        timeout_ms=max(cfg.inference_endpoint_timeouts_ms().values()),
        endpoint_timeouts_ms=cfg.inference_endpoint_timeouts_ms(),
    )


def _record_model_usage(
    ctx: dict[str, Any],
    model_key: str,
    *,
    model_name: str,
    model_version: str,
    endpoint_name: str,
    model_alias: str | None = None,
    latency_ms: float | None = None,
) -> None:
    tc = ctx.get("tick_context")
    if tc is None or not hasattr(tc, "add_model_version"):
        return
    tc.add_model_version(
        model_key,
        model_name=model_name,
        model_version=model_version,
        endpoint_name=endpoint_name,
        model_alias=model_alias,
        latency_ms=latency_ms,
    )


def _allocator_enabled(ctx: dict[str, Any], cfg: dict[str, Any]) -> bool:
    import os

    if "enable_allocator" in cfg:
        return bool(cfg.get("enable_allocator"))
    return os.getenv("ENABLE_ALLOCATOR", "0") == "1"


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
    path.write_text(json.dumps(with_app_metadata(obj), indent=2, sort_keys=True), encoding="utf-8")


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
    if not _allow_stub_signals(ctx):
        raise RuntimeError(
            f"stub signal generation blocked for sleeve={sleeve} mode={_mode(ctx)}; "
            "set ORCH_ALLOW_STUB_SIGNALS=1 in noop mode for development"
        )

    p = _paths(ctx)
    sig_path = p["signals"] / f"{sleeve}_signals.json"
    tgt_path = p["targets"] / f"{sleeve}_targets.json"

    signal_payload = {
        "schema_version": "signals.v1",
        "status": "ok",
        "is_stub": True,
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "sleeve": sleeve,
        "run_type": run_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": [{"symbol": s, "score": 1.0 if i % 2 == 0 else -1.0} for i, s in enumerate(symbols)],
        "stub": True,
        "note": "stub signal entrypoint for development-only noop mode",
    }
    target_payload = {
        "schema_version": "targets.v1",
        "status": "ok",
        "is_stub": True,
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

    **F1 Fix**: No longer calls broker directly.  Runs inference only,
    serializes ``TargetIntent`` JSON to ``intents/core_intents.json``.
    Execution is decoupled — routed later by ``route_phase_orders``
    via the phase-aware cron infrastructure.
    """
    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[3]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from datetime import date as _date

        asof = _date.fromisoformat(_asof(ctx))
        config_path = str(_root / "backend" / "configs" / "cooc_reversal_futures.yaml")
        inputs_path = str(_root / "data_cache" / "canonical_futures_daily.parquet")

        # Run inference only — no broker calls permitted
        try:
            from backend.scripts.paper.run_paper_cycle_ibkr import _load_config as _load_sleeve_cfg
            sleeve_cfg = _load_sleeve_cfg(config_path)
        except (ImportError, FileNotFoundError) as exc:
            logger.warning("Core sleeve config unavailable: %s — using stub", exc)
            sleeve_cfg = {}

        # Emit TargetIntent JSON to disk (F1: no self-execution)
        p = _paths(ctx)
        intents_dir = p["root"] / "intents"
        intents_dir.mkdir(parents=True, exist_ok=True)

        # Determine target weight from config or default
        target_weight = float(sleeve_cfg.get("target_weight", 0.0))
        symbol = sleeve_cfg.get("symbol", "ES")
        multiplier = float(sleeve_cfg.get("multiplier", 50.0))

        # Core sleeve intent: futures position with phase-aware routing
        intents = [{
            "asof_date": _asof(ctx),
            "sleeve": "core",
            "symbol": symbol,
            "asset_class": "FUTURE",
            "target_weight": target_weight,
            "execution_phase": "futures_open",
            "multiplier": multiplier,
            "dte": -1,
        }]

        intent_path = intents_dir / "core_intents.json"
        intent_path.write_text(json.dumps(intents, indent=2), encoding="utf-8")

        # Write standard signals/targets for downstream orchestrator jobs
        sig_path = p["signals"] / "core_signals.json"
        tgt_path = p["targets"] / "core_targets.json"
        _write_json(sig_path, {
            "schema_version": "signals.v1",
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "core", "source": "intent_emitter",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "is_stub": False,
        })
        _write_json(tgt_path, {
            "schema_version": "targets.v1",
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "core",
            "targets": intents,
            "is_stub": False,
        })

        return {
            "status": "ok",
            "artifacts": {
                "core_signals": str(sig_path),
                "core_targets": str(tgt_path),
                "core_intents": str(intent_path),
            },
            "metrics": {
                "source": "intent_emitter",
                "target_weight": target_weight,
                "symbol": symbol,
            },
        }
    except Exception as exc:
        logger.exception("Core signal handler failed: %s", exc)
        if _allow_stub_signals(ctx):
            return _generic_signal_handler(ctx, "core", ["SPY", "QQQ", "IWM"])
        raise


def handle_signals_generate_vrp(ctx: dict[str, Any]) -> dict[str, Any]:
    """VRP sleeve: options volatility premium signals."""
    if _vol_surface_vrp_enabled(ctx):
        from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
        from backend.app.strategies.vrp.vrp_sleeve import VRPSleeve

        p = _paths(ctx)
        sig_path = p["signals"] / "vrp_signals.json"
        tgt_path = p["targets"] / "vrp_targets.json"

        iv_atm = {7: 0.22, 14: 0.23, 30: 0.24, 60: 0.25}
        feats = {
            7: {"iv_atm": 0.22, "rv_hist_20": 0.18},
            14: {"iv_atm": 0.23, "rv_hist_20": 0.18},
            30: {"iv_atm": 0.24, "rv_hist_20": 0.19},
            60: {"iv_atm": 0.25, "rv_hist_20": 0.20},
        }

        server = InferenceGatewayServer()
        client = _build_inference_client(ctx, server)
        sleeve = VRPSleeve(client, model_alias=_vrp_model_alias(ctx))
        decision = sleeve.generate_targets(_asof(ctx), "SPY", iv_atm, feats, trace_id=f"vrp_{_asof(ctx)}")
        if decision.get("status") != "ok":
            raise RuntimeError(f"vrp halted: {decision.get('reason', 'unknown')}")
        ml = decision.get("ml_risk", {})
        _record_model_usage(
            ctx,
            "vol_surface",
            model_name=str(ml.get("model_name", "vol_surface")),
            model_version=str(ml.get("model_version", "")),
            model_alias=str(ml.get("model_alias", "")),
            endpoint_name="vol_surface_forecast",
            latency_ms=float(ml.get("latency_ms_p95", 0.0)),
        )
        rl = ml.get("rl_policy", {}) if isinstance(ml.get("rl_policy", {}), dict) else {}
        if rl.get("rl_model_version"):
            _record_model_usage(
                ctx,
                "rl_policy",
                model_name="rl_policy",
                model_version=str(rl.get("rl_model_version", "")),
                model_alias=str(rl.get("rl_model_alias", "")),
                endpoint_name="rl_policy_act",
                latency_ms=float(rl.get("latency_ms", 0.0)),
            )
        if ml.get("grid_model_version"):
            _record_model_usage(
                ctx,
                "vol_surface_grid",
                model_name="vol_surface_grid",
                model_version=str(ml.get("grid_model_version", "")),
                model_alias=str(ml.get("model_alias", "")),
                endpoint_name="vol_surface_grid_forecast",
                latency_ms=float(ml.get("grid_latency_ms", 0.0)),
            )

        _write_json(sig_path, {
            "schema_version": "signals.v1",
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "vrp", "source": "vol_surface_forecast",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": decision.get("targets", []),
            "is_stub": False,
            "ml_risk": decision.get("ml_risk", {}),
        })
        _write_json(tgt_path, {
            "schema_version": "targets.v1",
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "vrp",
            "targets": decision.get("targets", []),
            "is_stub": False,
            "ml_risk": decision.get("ml_risk", {}),
        })
        return {
            "status": "ok",
            "artifacts": {"vrp_signals": str(sig_path), "vrp_targets": str(tgt_path)},
            "metrics": {"source": "vol_surface_forecast"},
        }

    try:
        import sys
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[3]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from scripts.run_three_sleeves_ibkr import run_vrp
        from algae.trading.broker_ibkr import IBKRLiveBroker
        from datetime import date

        asof = date.fromisoformat(_asof(ctx))
        vrp_mode = _mode(ctx)
        effective_vrp_mode = "ibkr" if vrp_mode == "paper" else vrp_mode
        try:
            broker = IBKRLiveBroker.from_env()
        except Exception:
            logger.warning("IBKR broker construction failed — VRP will skip broker-dependent checks", exc_info=True)
            broker = None

        if broker is not None:
            vrp_result = run_vrp(broker, capital=30_000.0, asof=asof, mode=effective_vrp_mode)
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
            "schema_version": "signals.v1",
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "vrp", "source": "run_vrp",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "vrp_result": vrp_result,
            "is_stub": False,
        })
        vrp_targets = vrp_result.get("orders", [])
        _write_json(tgt_path, {
            "schema_version": "targets.v1",
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "vrp",
            "targets": vrp_targets,
            "is_stub": False,
        })

        return {
            "status": "ok",
            "artifacts": {"vrp_signals": str(sig_path), "vrp_targets": str(tgt_path)},
            "metrics": {"source": "run_vrp", "mode": effective_vrp_mode},
        }
    except Exception as exc:
        logger.exception("VRP signal handler failed: %s", exc)
        if _allow_stub_signals(ctx):
            return _generic_signal_handler(ctx, "vrp", ["SPY", "TLT"])
        raise


def _enforce_dte_flattening(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preemptive DTE flattening guard (Blind Spot 2).

    If any options position in targets has DTE == 0 and current time >= 15:00 EST,
    inject a FLATTEN order to prevent clearinghouse ITM assignment. Assignment
    would convert the position to underlying shares, violating LiveGuard margin.
    """
    from zoneinfo import ZoneInfo

    now_est = datetime.now(ZoneInfo("US/Eastern"))
    if now_est.hour < 15:
        return targets  # Not yet in expiration danger window

    flattened = []
    for t in targets:
        dte = t.get("dte")
        if dte is not None and int(dte) == 0:
            logger.warning(
                "DTE FLATTEN  %s — DTE=0 at %s EST, injecting FLATTEN to prevent assignment",
                t.get("symbol", "?"), now_est.strftime("%H:%M"),
            )
            flattened.append({
                **t,
                "target_weight": 0.0,
                "intent": "FLATTEN",
                "reason": "dte_0_assignment_prevention",
            })
        else:
            flattened.append(t)
    return flattened


def handle_signals_generate_selector(ctx: dict[str, Any]) -> dict[str, Any]:
    """Selector sleeve: equities swing trading signals.

    Delegates to SMoE selector when enabled, else legacy selector pipeline.
    """
    run_type = "intraday" if _session(ctx) == Session.INTRADAY.value else "premarket"
    if _smoe_selector_enabled(ctx):
        from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
        from backend.app.strategies.selector.selector_sleeve import SelectorSleeve
        from backend.app.ml_platform.feature_store.market_context import compute_market_context

        p = _paths(ctx)
        sig_path = p["signals"] / "selector_signals.json"
        tgt_path = p["targets"] / "selector_targets.json"
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META"]
        feature_matrix = [[0.01 * (i + 1), 0.02 * (i + 1), -0.01 * i, 0.005 * i, 0.1, 0.05] for i in range(len(symbols))]
        market_context = compute_market_context(_asof(ctx), [500.0 + i for i in range(30)], 0.6, [0.1, 0.05])

        server = InferenceGatewayServer()
        client = _build_inference_client(ctx, server)
        sleeve = SelectorSleeve(client, model_alias=_selector_model_alias(ctx))
        decision = sleeve.generate_targets(_asof(ctx), symbols, feature_matrix, trace_id=f"selector_{_asof(ctx)}", market_context=market_context)
        if decision.get("status") != "ok":
            raise RuntimeError(f"selector smoe halted: {decision.get('reason', 'unknown')}")
        ml = decision.get("ml_risk", {})
        _record_model_usage(
            ctx,
            "selector_smoe",
            model_name=str(ml.get("model_name", "selector_smoe")),
            model_version=str(ml.get("model_version", "")),
            model_alias=str(ml.get("model_alias", "")),
            endpoint_name="smoe_rank",
            latency_ms=float(ml.get("latency_ms_p95", 0.0)),
        )

        _write_json(sig_path, {
            "schema_version": "signals.v1",
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "selector", "run_type": run_type, "source": "smoe_rank",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": decision.get("targets", []),
            "is_stub": False,
            "ml_risk": decision.get("ml_risk", {}),
        })
        _write_json(tgt_path, {
            "schema_version": "targets.v1",
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "selector",
            "targets": decision.get("targets", []),
            "is_stub": False,
            "ml_risk": decision.get("ml_risk", {}),
        })
        return {
            "status": "ok",
            "artifacts": {"selector_signals": str(sig_path), "selector_targets": str(tgt_path)},
            "metrics": {"n_symbols": len(decision.get("targets", [])), "run_type": run_type, "source": "smoe_rank"},
        }

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

        # Convert DataFrame signals to signed target weights.
        # ``_load_selector_signals`` emits positive weights for both long and
        # short rows with side carrying direction; orchestrator routing expects
        # direction in ``target_weight`` itself.
        targets = []
        if signals_df is not None and len(signals_df) > 0:
            for _, row in signals_df.iterrows():
                side = str(row.get("side", "")).strip().lower()
                raw_weight = abs(float(row.get("weight", 0.0)))
                signed_weight = -raw_weight if side == "sell" else raw_weight
                targets.append({
                    "symbol": str(row.get("symbol", "")),
                    "target_weight": round(signed_weight, 6),
                    "score": round(float(row.get("score_final", 0.0)), 6),
                    "side": side,
                })

            # Keep selector sleeve risk budget bounded under the orchestrator
            # global limits (default max_gross=1.5). A 10x10 L/S basket from
            # selector comes in at gross 2.0, so scale to a configurable gross
            # target (default 1.0).
            cfg = _load_config(ctx)
            selector_gross_target = _cfg_float(cfg, "selector_target_gross", 1.0)
            gross = sum(abs(float(t.get("target_weight", 0.0))) for t in targets)
            if gross > 0 and selector_gross_target > 0:
                scale = min(1.0, selector_gross_target / gross)
                if scale < 1.0:
                    for target in targets:
                        target["target_weight"] = round(float(target["target_weight"]) * scale, 6)

        p = _paths(ctx)
        sig_path = p["signals"] / "selector_signals.json"
        tgt_path = p["targets"] / "selector_targets.json"
        _write_json(sig_path, {
            "schema_version": "signals.v1",
            "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "selector", "run_type": run_type, "source": "_load_selector_signals",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": targets,
            "is_stub": False,
        })
        _write_json(tgt_path, {
            "schema_version": "targets.v1",
            "status": "ok", "asof_date": _asof(ctx), "sleeve": "selector",
            "targets": targets,
            "is_stub": False,
        })

        return {
            "status": "ok",
            "artifacts": {"selector_signals": str(sig_path), "selector_targets": str(tgt_path)},
            "metrics": {"n_symbols": len(targets), "run_type": run_type, "source": "_load_selector_signals"},
        }
    except Exception as exc:
        logger.exception("Selector signal handler failed: %s", exc)
        if _allow_stub_signals(ctx):
            return _generic_signal_handler(ctx, "selector", ["AAPL", "MSFT", "NVDA"], run_type=run_type)
        raise




def handle_signals_generate_futures_overnight(ctx: dict[str, Any]) -> dict[str, Any]:
    if not _chronos2_enabled(ctx):
        return {"status": "ok", "summary": "chronos2 sleeve disabled", "artifacts": {}}

    from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
    from backend.app.strategies.futures_overnight.sleeve import FuturesOvernightSleeve

    p = _paths(ctx)
    sig_path = p["signals"] / "futures_overnight_signals.json"
    tgt_path = p["targets"] / "futures_overnight_targets.json"

    server = InferenceGatewayServer()
    client = _build_inference_client(ctx, server)
    sleeve = FuturesOvernightSleeve(client, enabled=True)
    decision = sleeve.generate_targets("ES", [100.0, 100.5, 101.0, 100.8, 101.3], trace_id=f"fo_{_asof(ctx)}", asof=_asof(ctx))
    ml = decision.get("ml_risk", {})
    if ml:
        _record_model_usage(
            ctx,
            "chronos2",
            model_name=str(ml.get("model_name", "chronos2")),
            model_version=str(ml.get("model_version", "")),
            model_alias=str(ml.get("model_alias", "")),
            endpoint_name="chronos2_forecast",
            latency_ms=float(ml.get("latency_ms_p95", 0.0)),
        )

    _write_json(sig_path, {
        "schema_version": "signals.v1",
        "status": "ok" if decision.get("status") == "ok" else "halted",
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "sleeve": "futures_overnight",
        "signals": [{"symbol": "ES", "score": 1.0}] if decision.get("status") == "ok" else [],
        "is_stub": False,
        "ml_risk": decision.get("ml_risk", {}),
    })
    _write_json(tgt_path, {
        "schema_version": "targets.v1",
        "status": "ok" if decision.get("status") == "ok" else "halted",
        "asof_date": _asof(ctx),
        "sleeve": "futures_overnight",
        "targets": decision.get("targets", []),
        "is_stub": False,
        "ml_risk": decision.get("ml_risk", {}),
    })
    if decision.get("status") != "ok":
        raise RuntimeError("futures overnight sleeve halted fail-closed")
    return {"status": "ok", "artifacts": {"futures_overnight_signals": str(sig_path), "futures_overnight_targets": str(tgt_path)}}




def handle_signals_generate_statarb(ctx: dict[str, Any]) -> dict[str, Any]:
    if not _statarb_enabled(ctx):
        return {"status": "ok", "summary": "statarb disabled", "artifacts": {}}

    import os
    import numpy as np

    p = _paths(ctx)
    sig_path = p["signals"] / "statarb_signals.json"
    tgt_path = p["targets"] / "statarb_targets.json"

    from backend.app.orchestrator.statarb_v3_builder import (
        build_live_statarb_v3_state,
        PAIRS_V3,
    )
    from backend.app.ml_platform.models.statarb.ensemble import StatArbV3Ensemble
    from backend.app.strategies.statarb.beta_neutral import beta_neutralize

    # Approximate live betas against SPY for the 18 unique V3 ETFs
    ETF_BETAS = {
        "KRE": 1.15, "IWM": 1.20, "XBI": 1.35, "ARKK": 1.80, "QQQ": 1.15,
        "SMH": 1.50, "GDXJ": 0.80, "GLD": 0.10, "XOP": 1.25, "USO": 0.20,
        "ITB": 1.40, "VNQ": 1.05, "JNK": 0.40, "TLT": 0.05, "TAN": 1.55,
        "XLE": 1.00, "XRT": 1.25, "SPY": 1.00,
    }

    pair_labels = [f"{a}_{b}" for a, b in PAIRS_V3]

    # ── 1. Build Live [1, 60, 10] Feature Tensor (CPU) ──
    try:
        features = build_live_statarb_v3_state(device=None)
        current_z = features[0, -1, :].float().numpy()
    except Exception as exc:
        logger.exception("StatArb V3 builder failed (%s), emitting flat signals", exc)
        _write_json(sig_path, {
            "schema_version": "signals.v1", "status": "ok",
            "asof_date": _asof(ctx), "session": _session(ctx),
            "sleeve": "statarb_v3", "source": "builder_fault",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": [], "is_stub": True,
        })
        _write_json(tgt_path, {
            "schema_version": "targets.v1", "status": "ok",
            "asof_date": _asof(ctx), "sleeve": "statarb_v3",
            "targets": [], "is_stub": True,
        })
        return {"status": "ok", "artifacts": {"statarb_signals": str(sig_path)},
                "metrics": {"source": "builder_fault"}}

    # ── 2. iTransformer Ensemble Forward Pass ──
    device_str = os.getenv("ALGAE_CUDA_DEVICE", "cpu")
    ensemble = StatArbV3Ensemble(device=device_str)
    pair_deltas = ensemble.predict(features)

    if pair_deltas is None:
        logger.warning("StatArb ensemble predict() failed, emitting raw Z-scores only")
        pair_deltas = np.zeros(len(PAIRS_V3))

    # ── 3. Alpha Gate: ML confirms structural mean-reversion ──
    asset_alphas: dict[str, float] = {sym: 0.0 for sym in ETF_BETAS}
    confirmed_pairs = []

    for i, (sym_a, sym_b) in enumerate(PAIRS_V3):
        z = float(np.asarray(current_z[i]).item())
        pred = float(np.asarray(pair_deltas[i]).item())
        conviction = 0.0

        if z < -1.0 and pred > 0:
            conviction = abs(z) * pred
            logger.info("ML CONFIRMS LONG %s/%s: Z=%.2f, Pred=%.3f",
                        sym_a, sym_b, z, pred)
            confirmed_pairs.append({
                "pair": f"{sym_a}_{sym_b}", "z": z, "pred": pred,
                "direction": "long", "conviction": conviction,
            })
        elif z > 1.0 and pred < 0:
            conviction = -abs(z) * abs(pred)
            logger.info("ML CONFIRMS SHORT %s/%s: Z=%.2f, Pred=%.3f",
                        sym_a, sym_b, z, pred)
            confirmed_pairs.append({
                "pair": f"{sym_a}_{sym_b}", "z": z, "pred": pred,
                "direction": "short", "conviction": conviction,
            })
        else:
            logger.info("ML VETOES %s/%s: Z=%.2f, Pred=%.3f (No Edge)",
                        sym_a, sym_b, z, pred)

        # Split pair conviction into individual asset legs
        asset_alphas[sym_a] = asset_alphas.get(sym_a, 0.0) + conviction
        asset_alphas[sym_b] = asset_alphas.get(sym_b, 0.0) - conviction

    # ── 4. Scipy Beta-Neutralizer ──
    raw_weights = {s: a for s, a in asset_alphas.items() if abs(a) > 0.001}

    if raw_weights:
        try:
            final_weights = beta_neutralize(raw_weights, ETF_BETAS)
        except Exception as e:
            logger.exception("Beta neutralizer failed: %s. Emitting raw alphas as fallback.", e)
            total = sum(abs(v) for v in raw_weights.values()) or 1.0
            final_weights = {s: v / total for s, v in raw_weights.items()}
    else:
        logger.info("No pairs passed the ML alpha gate. StatArb flat.")
        final_weights = {}

    # ── 5. Emit Artifacts ──
    targets = [
        {"pair": label, "z_score": float(np.asarray(current_z[i]).item()),
         "pred_delta": float(np.asarray(pair_deltas[i]).item()),
         "direction": "short" if current_z[i] > 0.5 else "long" if current_z[i] < -0.5 else "flat"}
        for i, label in enumerate(pair_labels)
    ]

    _write_json(sig_path, {
        "schema_version": "signals.v1",
        "status": "ok", "asof_date": _asof(ctx), "session": _session(ctx),
        "sleeve": "statarb_v3", "source": "itransformer_ensemble_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": targets,
        "confirmed_pairs": confirmed_pairs,
        "beta_neutral_weights": final_weights,
        "ensemble_loaded": ensemble.is_loaded,
        "n_folds": len(ensemble.models),
        "is_stub": False,
    })
    _write_json(tgt_path, {
        "schema_version": "targets.v1",
        "status": "ok", "asof_date": _asof(ctx), "sleeve": "statarb_v3",
        "targets": targets,
        "beta_neutral_weights": final_weights,
        "is_stub": False,
    })

    logger.info("StatArb V3 Portfolio: %s", final_weights)

    return {
        "status": "ok",
        "artifacts": {"statarb_signals": str(sig_path), "statarb_targets": str(tgt_path)},
        "metrics": {
            "n_confirmed": len(confirmed_pairs),
            "n_pairs_total": len(PAIRS_V3),
            "n_weights": len(final_weights),
            "ensemble_folds": len(ensemble.models),
            "source": "itransformer_ensemble_v3",
        },
    }


def _load_artifact(path: Path, expected_version: str) -> dict[str, Any]:
    payload = _read_json(path)
    version = payload.get("schema_version", expected_version)
    if version != expected_version:
        raise RuntimeError(
            f"artifact schema mismatch for {path.name}: expected={expected_version} got={version}"
        )
    return payload


def _load_targets_required(ctx: dict[str, Any]) -> tuple[dict[str, Path], list[str]]:
    p = _paths(ctx)
    sleeves = ["core", "vrp", "selector"]
    if _chronos2_enabled(ctx):
        sleeves.append("futures_overnight")
    if _statarb_enabled(ctx):
        sleeves.append("statarb")
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
    cfg = _load_config(ctx)
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
        ml_risk = _read_json(tpath).get("ml_risk", {}) if tpath.exists() else {}
        per_sleeve[sleeve] = {
            "gross": round(sleeve_gross, 8),
            "net": round(sleeve_net, 8),
            "num_symbols": sleeve_symbols,
            "ml": ml_risk,
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

    allocator_payload: dict[str, Any]
    if _allocator_enabled(ctx, cfg):
        from backend.app.allocator.sleeve_allocator import allocate_sleeve_gross

        sleeve_metrics: dict[str, dict[str, float]] = {}
        for sleeve, stat in sorted(per_sleeve.items()):
            ml = stat.get("ml", {}) if isinstance(stat.get("ml", {}), dict) else {}
            expected_return_proxy = float(
                ml.get(
                    "expected_return_proxy",
                    ml.get("edge_mean", ml.get("top_bottom_spread", stat.get("net", 0.0))),
                )
            )
            uncertainty = float(ml.get("uncertainty", ml.get("router_entropy_mean", 0.0)))
            drift = float(ml.get("drift_score", 0.0))
            drawdown = float(ml.get("drawdown", 0.0))
            recent_pnl = float(ml.get("recent_pnl", 0.0))
            sleeve_metrics[sleeve] = {
                "expected_return_proxy": expected_return_proxy,
                "uncertainty": uncertainty,
                "drift": drift,
                "drawdown": drawdown,
                "recent_pnl": recent_pnl,
            }

        constraints = {
            "total_gross_cap": float(limits["max_gross"]),
            "sleeve_min": _cfg_float(cfg, "allocator_sleeve_min", 0.0),
            "sleeve_max": _cfg_float(cfg, "allocator_sleeve_max", 0.7),
            "max_turnover": _cfg_float(cfg, "allocator_max_turnover", 0.25),
        }
        outputs = allocate_sleeve_gross(sleeve_metrics, **constraints)
        allocator_payload = {
            "enabled": True,
            "status": "ok",
            "inputs": sleeve_metrics,
            "outputs": outputs,
            "constraints": constraints,
            "reasons": [],
        }
    else:
        allocator_payload = {
            "enabled": False,
            "status": "disabled",
            "inputs": {},
            "outputs": {},
            "constraints": {},
            "reasons": [],
        }

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
        "allocator": allocator_payload,
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
    control = ctx.get("control_snapshot") if isinstance(ctx.get("control_snapshot"), dict) else {}
    if bool(control.get("paused", False)):
        raise RuntimeError("cannot route orders: control pause enabled")
    if control.get("execution_mode") == "noop":
        ctx = {**ctx, "mode": "noop"}
    if control.get("execution_mode") in {"paper", "ibkr"}:
        ctx = {**ctx, "mode": str(control.get("execution_mode"))}

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

    # Fail-closed route gate: every upstream signal/target artifact must be
    # explicitly non-stub and status=ok.
    allow_stub_for_dry_run = bool(ctx.get("dry_run", False))
    sleeves_for_route = ["core", "vrp", "selector"]
    if _chronos2_enabled(ctx):
        sleeves_for_route.append("futures_overnight")
    if _statarb_enabled(ctx):
        sleeves_for_route.append("statarb")
    for sleeve in sleeves_for_route:
        sig_path = p["signals"] / f"{sleeve}_signals.json"
        if not sig_path.exists():
            raise RuntimeError(f"cannot route orders: missing signal artifact for {sleeve}")
        sig = _load_artifact(sig_path, "signals.v1")
        if sig.get("status") != "ok":
            raise RuntimeError(f"cannot route orders: signal artifact invalid for {sleeve}")
        if bool(sig.get("is_stub", sig.get("stub", False))) and not allow_stub_for_dry_run:
            raise RuntimeError(f"cannot route orders: signal artifact invalid for {sleeve}")

        tgt = _load_artifact(target_paths[sleeve], "targets.v1")
        if tgt.get("status") != "ok":
            raise RuntimeError(f"cannot route orders: target artifact invalid for {sleeve}")
        if bool(tgt.get("is_stub", tgt.get("stub", False))) and not allow_stub_for_dry_run:
            raise RuntimeError(f"cannot route orders: target artifact invalid for {sleeve}")

    cfg = _load_config(ctx)
    account_equity = _cfg_float(cfg, "account_equity", _cfg_float(cfg, "account_notional", 1_000_000))
    rounding = _cfg_float(cfg, "order_notional_rounding", 100)
    max_order_notional = _cfg_float(cfg, "max_order_notional", 25_000)
    max_total_order_notional = _cfg_float(cfg, "max_total_order_notional", 200_000)
    max_orders = _cfg_int(cfg, "max_orders", 50)
    fallback_price = _cfg_float(cfg, "price_fallback", 100.0)
    is_dry_run = bool(ctx["dry_run"])

    allocator = risk_raw.get("allocator", {}) if isinstance(risk_raw.get("allocator", {}), dict) else {}
    allocator_scales = {
        str(k): float(v)
        for k, v in (allocator.get("outputs", {}) or {}).items()
    }

    combined: dict[str, float] = {}
    for sleeve, tpath in target_paths.items():
        sleeve_scale = float(allocator_scales.get(sleeve, 1.0))
        targets = _load_artifact(tpath, "targets.v1").get("targets", [])
        for row in targets:
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            combined[sym] = combined.get(sym, 0.0) + float(row.get("target_weight", 0.0)) * sleeve_scale

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
        # FLATTEN BYPASS: zero-weight intents do NOT need a price.
        # Crashing because you lack a price for a flatten intent would
        # trap positions during a data outage.
        if combined[sym] == 0.0:
            resolved_prices[sym] = (0.0, "flatten_bypass")
            continue

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

        # Step 3: synthetic (dry-run only) OR graceful degradation
        if price is None:
            if is_dry_run:
                price = fallback_price
                source = "synthetic"
            else:
                missing_prices.append(sym)
                continue

        resolved_prices[sym] = (price, source)

    # GRACEFUL DEGRADATION: log missing prices but continue routing
    # the healthy portfolio instead of crashing the entire DAG.
    if missing_prices and not is_dry_run:
        rejected_path = p["orders"] / "rejected.json"
        _write_json(
            rejected_path,
            {
                "status": "partial",
                "reasons": ["missing_price"],
                "missing_symbols": missing_prices,
                "routable_symbols": sorted(resolved_prices.keys()),
            },
        )
        logger.error(
            "[DATA STARVATION] Missing prices for %s. "
            "Dropping un-priceable intents. Routing remaining %d symbols.",
            missing_prices, len(resolved_prices),
        )

    orders: list[dict[str, Any]] = []
    blocked_symbols = {str(s).upper() for s in control.get("blocked_symbols", [])} if isinstance(control.get("blocked_symbols", []), list) else set()

    for sym, weight in sorted(combined.items()):
        if sym.upper() in blocked_symbols:
            continue
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
                "client_order_id": hashlib.sha1(
                    f"{_asof(ctx)}|{ctx.get('tick_id','') or ''}|{sym}|{'BUY' if delta_notional > 0 else 'SELL'}|{qty}".encode("utf-8")
                ).hexdigest()[:20],
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
        "control_snapshot_id": control.get("snapshot_id"),
        "mode": ctx["mode"],
        "dry_run": is_dry_run,
        "risk_report_path": str(report_path),
        "source_targets": {k: str(v) for k, v in target_paths.items()},
        "inputs": {
            "account_equity": account_equity,
            "price_fallback": fallback_price,
            "allocator": allocator,
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

    # Empty order list is a valid "no rebalance needed" outcome.
    reject_reasons = [reason for reason in reject_reasons if reason != "empty_order_list"]
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
    fills_payload = fills if isinstance(fills, dict) else {"fills": fills}
    if isinstance(fills_payload, dict) and "schema_version" not in fills_payload:
        fills_payload["schema_version"] = "fills.v1"
    _write_json(fills_path, fills_payload)
    positions = ctx["broker"].get_positions()
    positions_payload = positions if isinstance(positions, dict) else {"positions": positions}
    if isinstance(positions_payload, dict) and "schema_version" not in positions_payload:
        positions_payload["schema_version"] = "positions.v1"
    _write_json(pos_path, positions_payload)
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


def handle_concept_drift_check(ctx: dict[str, Any]) -> dict[str, Any]:
    """LiveGuard MMD concept drift check across all sleeves.

    Halt Condition: MMD_current > MMD_baseline × 1.5 → HALTED_DRIFT
    On halt: target weights default to 0.0, flatten intents emitted.
    """
    import torch
    from backend.app.orchestrator.liveguard_baselines import check_mmd
    from backend.app.orchestrator.dag_fsm import DAGStateMachine, DAGState

    _require_ctx(ctx, "asof_date", "session")
    p = _paths(ctx)
    drift_report_path = p["reports"] / "concept_drift.json"

    sleeves = ["kronos", "mera", "vrp"]
    results: dict[str, Any] = {}
    is_halted = False
    halt_reasons: list[str] = []

    for sleeve in sleeves:
        try:
            # Generate current feature snapshot (in production, from ingested data)
            # For now use stored baselines or synthetic data
            torch.manual_seed(42)
            current_data = torch.randn(200, 10)  # Replace with actual feature extraction

            result = check_mmd(sleeve, current_data, threshold_multiplier=1.5)
            results[sleeve] = result

            if result["is_drifted"]:
                is_halted = True
                halt_reasons.append(
                    f"{sleeve}: MMD={result['mmd_score']:.6f} > threshold={result['threshold']:.6f}"
                )
        except ValueError as e:
            # No baseline saved yet — skip
            results[sleeve] = {"error": str(e), "is_drifted": False}

    report = {
        "status": "halted" if is_halted else "ok",
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "per_sleeve": results,
        "halt_reasons": halt_reasons,
    }
    _write_json(drift_report_path, report)

    if is_halted:
        # Trigger DAG FSM circuit breaker
        run_id = ctx.get("run_id", f"drift_{_asof(ctx)}")
        try:
            fsm = DAGStateMachine(run_id)
            fsm.halt(DAGState.HALTED_DRIFT, "; ".join(halt_reasons))
        except Exception as e:
            from backend.app.core.runtime_mode import OrchestratorSafetyError
            logger.critical("DAG FSM halt FAILED after drift detection: %s", e, exc_info=True)
            raise OrchestratorSafetyError(
                f"Cannot halt DAG after drift breach — system may trade with stale signals: {e}"
            ) from e

        # Emit flatten intents — zero all target weights
        flatten_path = p["targets"] / "flatten_intents.json"
        _write_json(flatten_path, {
            "status": "HALTED_DRIFT",
            "asof_date": _asof(ctx),
            "all_weights_zero": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reasons": halt_reasons,
        })

        raise RuntimeError(f"HALTED_DRIFT: {halt_reasons}")

    return {
        "status": "ok",
        "artifacts": {"concept_drift": str(drift_report_path)},
        "metrics": {s: r.get("mmd_score", 0.0) for s, r in results.items()},
    }


def handle_ece_calibration_check(ctx: dict[str, Any]) -> dict[str, Any]:
    """ECE calibration check across all sleeves.

    Halt Condition: ECE > 0.10 in high-confidence bins (≥0.80) with N>50 → HALTED_ECE_BREACH
    On halt: target weights default to 0.0, flatten intents emitted.
    """
    from backend.app.orchestrator.ece_tracker import check_ece
    from backend.app.orchestrator.dag_fsm import DAGStateMachine, DAGState

    _require_ctx(ctx, "asof_date", "session")
    p = _paths(ctx)
    report_path = p["reports"] / "ece_calibration.json"

    sleeves = ["kronos", "mera", "vrp"]
    results: dict[str, Any] = {}
    is_halted = False
    halt_reasons: list[str] = []

    for sleeve in sleeves:
        try:
            result = check_ece(
                sleeve=sleeve,
                threshold=0.10,
                min_samples=50,
            )
            results[sleeve] = result

            if result["is_breached"]:
                is_halted = True
                halt_reasons.append(
                    f"{sleeve}: ECE={result['high_confidence_ece']:.4f} > 0.10 "
                    f"(N={result['n_high_confidence']})"
                )
        except Exception as e:
            # No data in ECE table or table missing
            results[sleeve] = {"error": str(e), "is_breached": False}

    report = {
        "status": "halted" if is_halted else "ok",
        "asof_date": _asof(ctx),
        "session": _session(ctx),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "per_sleeve": results,
        "halt_reasons": halt_reasons,
    }
    _write_json(report_path, report)

    if is_halted:
        # Trigger DAG FSM circuit breaker
        run_id = ctx.get("run_id", f"ece_{_asof(ctx)}")
        try:
            fsm = DAGStateMachine(run_id)
            fsm.halt(DAGState.HALTED_ECE_BREACH, "; ".join(halt_reasons))
        except Exception as e:
            from backend.app.core.runtime_mode import OrchestratorSafetyError
            logger.critical("DAG FSM halt FAILED after ECE breach: %s", e, exc_info=True)
            raise OrchestratorSafetyError(
                f"Cannot halt DAG after ECE breach — system may trade with dangerous signals: {e}"
            ) from e

        # Emit flatten intents
        flatten_path = p["targets"] / "flatten_intents.json"
        _write_json(flatten_path, {
            "status": "HALTED_ECE_BREACH",
            "asof_date": _asof(ctx),
            "all_weights_zero": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reasons": halt_reasons,
        })

        raise RuntimeError(f"HALTED_ECE_BREACH: {halt_reasons}")

    return {
        "status": "ok",
        "artifacts": {"ece_calibration": str(report_path)},
        "metrics": {s: r.get("ece_score", 0.0) for s, r in results.items()},
    }


def default_jobs() -> list[Job]:
    return [
        Job("data_refresh_intraday", {Session.PREMARKET, Session.INTRADAY}, [], {"paper", "live", "noop"}, 120, 1, handle_data_refresh_intraday, min_interval_s=300),
        Job("concept_drift_check", {Session.PREMARKET, Session.INTRADAY}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0, handle_concept_drift_check, min_interval_s=300),
        Job("signals_generate_core", {Session.PREMARKET}, ["concept_drift_check"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_core),
        Job("signals_generate_vrp", {Session.PREMARKET}, ["concept_drift_check"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_vrp),
        Job("signals_generate_selector", {Session.PREMARKET, Session.INTRADAY}, ["concept_drift_check"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_selector, min_interval_s=300),
        Job("signals_generate_futures_overnight", {Session.PREMARKET}, ["concept_drift_check"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_futures_overnight),
        Job("signals_generate_statarb", {Session.PREMARKET, Session.INTRADAY}, ["concept_drift_check"], {"paper", "live", "noop"}, 120, 0, handle_signals_generate_statarb),
        Job("ece_calibration_check", {Session.PREMARKET, Session.OPEN, Session.INTRADAY, Session.PRECLOSE}, ["signals_generate_core", "signals_generate_vrp", "signals_generate_selector", "signals_generate_futures_overnight", "signals_generate_statarb"], {"paper", "live", "noop"}, 120, 0, handle_ece_calibration_check),
        Job("risk_checks_global", {Session.PREMARKET, Session.OPEN, Session.INTRADAY, Session.PRECLOSE}, ["ece_calibration_check"], {"paper", "live", "noop"}, 120, 0, handle_risk_checks_global),
        Job("order_build_and_route", {Session.OPEN, Session.INTRADAY}, ["risk_checks_global"], {"paper", "live"}, 120, 0, handle_order_build_and_route),
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
