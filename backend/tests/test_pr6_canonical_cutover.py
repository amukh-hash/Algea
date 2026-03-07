from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from backend.app.orchestrator.broker import PaperBrokerStub
from backend.app.orchestrator.calendar import Session
from backend.app.orchestrator.job_defs import Job, handle_order_build_and_route, handle_risk_checks_global
from backend.app.orchestrator.orchestrator import Orchestrator
from backend.tests.test_orchestrator import SpyBroker


def _write(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _base_ctx(root: Path, *, dry_run: bool = True, flags: dict[str, bool] | None = None, extra_cfg: dict | None = None):
    cfg = {
        "account_equity": 100_000,
        "order_notional_rounding": 1,
        "max_orders": 50,
        "max_order_notional": 500_000,
        "max_total_order_notional": 1_000_000,
        "FF_INTENT_CANONICAL_TRANSLATOR": False,
        "FF_INTENT_CANONICAL_PLANNER": False,
        "FF_RISK_REPORT_V1_ONLY": True,
    }
    if flags:
        cfg.update(flags)
    if extra_cfg:
        cfg.update(extra_cfg)
    return {
        "asof_date": "2026-02-17",
        "session": "open",
        "artifact_root": str(root),
        "mode": "paper",
        "dry_run": dry_run,
        "broker": SpyBroker(price_map={"SPY": 500.0}),
        "config": cfg,
        "control_snapshot": {"snapshot_id": "snap-1", "paused": False, "execution_mode": "paper", "blocked_symbols": []},
    }


def _seed_signals_and_targets(root: Path, weight: float = 0.05) -> None:
    for sleeve in ["core", "vrp", "selector"]:
        _write(root / "signals" / f"{sleeve}_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": False})
        _write(
            root / "targets" / f"{sleeve}_targets.json",
            {
                "schema_version": "targets.v1",
                "status": "ok",
                "is_stub": False,
                "targets": [{"symbol": "SPY", "target_weight": weight if sleeve == "core" else 0.0}],
            },
        )


def test_canonical_cutover_uses_intents_as_primary_input(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root, weight=0.05)
    _write(
        root / "intents" / "core_intents.json",
        [{
            "asof_date": "2026-02-17",
            "sleeve": "core",
            "symbol": "SPY",
            "asset_class": "EQUITY",
            "target_weight": -0.1,
            "execution_phase": "intraday",
        }],
    )

    ctx = _base_ctx(root, flags={"FF_INTENT_CANONICAL_TRANSLATOR": True, "FF_INTENT_CANONICAL_PLANNER": True})
    handle_risk_checks_global(ctx)
    out = handle_order_build_and_route(ctx)

    assert out["status"] == "ok"
    payload = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert payload["canonical_planner_active"] is True
    assert payload["planning_input_family"] == "mixed_intents"
    assert payload["orders"][0]["side"] == "SELL"


def test_rollback_to_legacy_target_path_when_cutover_off(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root, weight=0.05)
    _write(
        root / "intents" / "core_intents.json",
        [{
            "asof_date": "2026-02-17",
            "sleeve": "core",
            "symbol": "SPY",
            "asset_class": "EQUITY",
            "target_weight": -0.9,
            "execution_phase": "intraday",
        }],
    )
    ctx = _base_ctx(root, flags={"FF_INTENT_CANONICAL_TRANSLATOR": False, "FF_INTENT_CANONICAL_PLANNER": False})
    handle_risk_checks_global(ctx)
    out = handle_order_build_and_route(ctx)

    payload = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert out["status"] == "ok"
    assert payload["canonical_planner_active"] is False
    assert payload["planning_input_family"] == "targets_legacy"
    assert payload["orders"][0]["side"] == "BUY"


def test_allocator_applied_exactly_once_for_canonical_translated_intents(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root, weight=0.05)
    _write(
        root / "reports" / "risk_checks.json",
        {
            "schema_version": "risk_decision.v1",
            "decision_id": "d1",
            "status": "ok",
            "checked_at": "2026-02-17T10:00:00+00:00",
            "asof_date": "2026-02-17",
            "session": "open",
            "policy_version": "risk_decision_policy.v1",
            "input_contract_family": "targets_legacy",
            "source_sleeves": ["core", "vrp", "selector"],
            "input_artifact_refs": {},
            "generated_by": "test",
            "reason": None,
            "missing_sleeves": [],
            "inputs": {},
            "metrics": {"nan_or_inf": False, "gross_exposure": 0.2, "net_exposure": 0.2, "num_symbols": 1, "per_sleeve": {}},
            "limits": {"max_gross": 1.5, "max_net_abs": 0.5, "max_symbol_abs_weight": 0.5, "max_symbols": 200},
            "allocator": {"enabled": True, "status": "ok", "inputs": {}, "outputs": {"core": 0.5}, "constraints": {}, "reasons": []},
            "violations": [],
        },
    )

    ctx = _base_ctx(root, flags={"FF_INTENT_CANONICAL_TRANSLATOR": True, "FF_INTENT_CANONICAL_PLANNER": True})
    out = handle_order_build_and_route(ctx)
    payload = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert out["status"] == "ok"
    # 0.05 target with 0.5 allocator => 0.025 effective (single application)
    assert payload["orders"][0]["qty"] == 5


def test_native_intent_pass_through_scaling_is_not_applied(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root, weight=0.05)
    _write(
        root / "intents" / "core_intents.json",
        [{
            "asof_date": "2026-02-17",
            "sleeve": "core",
            "symbol": "SPY",
            "asset_class": "EQUITY",
            "target_weight": 0.3,
            "execution_phase": "intraday",
        }],
    )
    _write(
        root / "reports" / "risk_checks.json",
        {
            "schema_version": "risk_decision.v1",
            "decision_id": "d1",
            "status": "ok",
            "checked_at": "2026-02-17T10:00:00+00:00",
            "asof_date": "2026-02-17",
            "session": "open",
            "policy_version": "risk_decision_policy.v1",
            "input_contract_family": "mixed_intents",
            "source_sleeves": ["core", "vrp", "selector"],
            "input_artifact_refs": {},
            "generated_by": "test",
            "reason": None,
            "missing_sleeves": [],
            "inputs": {},
            "metrics": {"nan_or_inf": False, "gross_exposure": 0.3, "net_exposure": 0.3, "num_symbols": 1, "per_sleeve": {}},
            "limits": {"max_gross": 1.5, "max_net_abs": 0.5, "max_symbol_abs_weight": 0.5, "max_symbols": 200},
            "allocator": {"enabled": True, "status": "ok", "inputs": {}, "outputs": {"core": 0.25}, "constraints": {}, "reasons": []},
            "violations": [],
        },
    )
    ctx = _base_ctx(root, flags={"FF_INTENT_CANONICAL_TRANSLATOR": True, "FF_INTENT_CANONICAL_PLANNER": True})
    handle_order_build_and_route(ctx)
    payload = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    # pass-through native weight 0.3 => qty 60, not allocator-scaled
    assert payload["orders"][0]["qty"] == 60
    traces = payload["orders"][0]["intent_trace_refs"]
    assert traces[0]["allocator_scale_applied"] == 1.0


def test_canonical_order_traceability_fields_present(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root, weight=0.05)
    ctx = _base_ctx(root, flags={"FF_INTENT_CANONICAL_TRANSLATOR": True, "FF_INTENT_CANONICAL_PLANNER": True})
    handle_risk_checks_global(ctx)
    handle_order_build_and_route(ctx)
    payload = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    trace = payload["orders"][0]["intent_trace_refs"][0]
    for key in ["canonical_intent_ref", "source_sleeve", "source_artifact_path", "source_row_index", "derivation_policy_version", "allocator_scale_applied"]:
        assert key in trace


def test_run_once_surfaces_translation_failure(tmp_path: Path):
    from backend.app.orchestrator.config import OrchestratorConfig

    def _writer(sleeve: str, pair_only: bool = False):
        def _h(ctx: dict):
            root = Path(str(ctx["artifact_root"]))
            _write(root / "signals" / f"{sleeve}_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": False})
            if pair_only:
                targets = [{"pair": "X_Y", "z_score": 1.2}]
            else:
                targets = [{"symbol": "SPY", "target_weight": 0.01}]
            _write(root / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": targets})
            return {"status": "ok", "artifacts": {}}
        return _h

    jobs = [
        Job("signals_generate_core", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("core")),
        Job("signals_generate_vrp", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("vrp")),
        Job("signals_generate_selector", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("selector")),
        Job("signals_generate_statarb", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("statarb", pair_only=True)),
        Job("risk_checks_global", {Session.OPEN}, ["signals_generate_core", "signals_generate_vrp", "signals_generate_selector", "signals_generate_statarb"], {"paper"}, 30, 0, handle_risk_checks_global),
        Job("order_build_and_route", {Session.OPEN}, ["risk_checks_global"], {"paper"}, 30, 0, handle_order_build_and_route),
    ]
    cfg = OrchestratorConfig(
        artifact_root=tmp_path / "artifacts",
        db_path=tmp_path / "state.sqlite3",
        mode="paper",
        paper_only=True,
    )
    setattr(cfg, "max_gross", 1.5)
    setattr(cfg, "max_net_abs", 0.5)
    setattr(cfg, "max_symbol_abs_weight", 0.5)
    setattr(cfg, "max_symbols", 500)
    setattr(cfg, "FF_INTENT_CANONICAL_TRANSLATOR", True)
    setattr(cfg, "FF_INTENT_CANONICAL_PLANNER", True)
    setattr(cfg, "FF_RISK_REPORT_V1_ONLY", True)
    setattr(cfg, "enable_statarb_sleeve", True)
    orch = Orchestrator(config=cfg, jobs=jobs, broker=SpyBroker(price_map={"SPY": 500.0}))
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.OPEN, dry_run=True)
    assert "risk_checks_global" in res.failed_jobs


def test_statarb_fail_closed_under_canonical_planner(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root)
    _write(root / "signals" / "statarb_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": False})
    _write(root / "targets" / "statarb_targets.json", {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"pair": "X_Y", "z_score": 1.1}]})
    _write(root / "reports" / "risk_checks.json", {
        "schema_version": "risk_decision.v1",
        "decision_id": "d1",
        "status": "ok",
        "checked_at": "2026-02-17T10:00:00+00:00",
        "asof_date": "2026-02-17",
        "session": "open",
        "policy_version": "risk_decision_policy.v1",
        "input_contract_family": "targets_legacy",
        "source_sleeves": ["core", "vrp", "selector", "statarb"],
        "input_artifact_refs": {},
        "generated_by": "test",
        "reason": None,
        "missing_sleeves": [],
        "inputs": {},
        "metrics": {"nan_or_inf": False, "gross_exposure": 0.2, "net_exposure": 0.2, "num_symbols": 1, "per_sleeve": {}},
        "limits": {"max_gross": 1.5, "max_net_abs": 0.5, "max_symbol_abs_weight": 0.5, "max_symbols": 200},
        "allocator": {"enabled": False, "status": "disabled", "inputs": {}, "outputs": {}, "constraints": {}, "reasons": []},
        "violations": [],
    })

    ctx = _base_ctx(
        root,
        flags={"FF_INTENT_CANONICAL_TRANSLATOR": True, "FF_INTENT_CANONICAL_PLANNER": True},
        extra_cfg={"enable_statarb_sleeve": True},
    )
    with pytest.raises(RuntimeError, match="statarb"):
        handle_order_build_and_route(ctx)


def test_canonical_risk_report_input_family_truthful(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root)
    _write(
        root / "intents" / "core_intents.json",
        [{
            "asof_date": "2026-02-17",
            "sleeve": "core",
            "symbol": "SPY",
            "asset_class": "EQUITY",
            "target_weight": 0.05,
            "execution_phase": "intraday",
        }],
    )
    ctx = _base_ctx(root, flags={"FF_INTENT_CANONICAL_TRANSLATOR": True, "FF_INTENT_CANONICAL_PLANNER": True})
    handle_risk_checks_global(ctx)
    report = json.loads((root / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert report["input_contract_family"] == "mixed_intents"



def test_run_once_surfaces_missing_futures_multiplier_under_canonical_cutover(tmp_path: Path):
    from backend.app.orchestrator.config import OrchestratorConfig

    def _writer(sleeve: str, *, bad_futures: bool = False):
        def _h(ctx: dict):
            root = Path(str(ctx["artifact_root"]))
            _write(root / "signals" / f"{sleeve}_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": False})
            target = {"symbol": "UNKNOWN", "target_weight": 0.03} if bad_futures else {"symbol": "SPY", "target_weight": 0.01}
            _write(root / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [target]})
            return {"status": "ok", "artifacts": {}}

        return _h

    jobs = [
        Job("signals_generate_core", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("core")),
        Job("signals_generate_vrp", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("vrp")),
        Job("signals_generate_selector", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("selector")),
        Job("signals_generate_futures_overnight", {Session.OPEN}, [], {"paper"}, 30, 0, _writer("futures_overnight", bad_futures=True)),
        Job("risk_checks_global", {Session.OPEN}, ["signals_generate_core", "signals_generate_vrp", "signals_generate_selector", "signals_generate_futures_overnight"], {"paper"}, 30, 0, handle_risk_checks_global),
        Job("order_build_and_route", {Session.OPEN}, ["risk_checks_global"], {"paper"}, 30, 0, handle_order_build_and_route),
    ]

    cfg = OrchestratorConfig(
        artifact_root=tmp_path / "artifacts",
        db_path=tmp_path / "state.sqlite3",
        mode="paper",
        paper_only=True,
    )
    setattr(cfg, "FF_INTENT_CANONICAL_TRANSLATOR", True)
    setattr(cfg, "FF_INTENT_CANONICAL_PLANNER", True)
    setattr(cfg, "FF_RISK_REPORT_V1_ONLY", True)
    setattr(cfg, "enable_chronos2_sleeve", True)

    orch = Orchestrator(config=cfg, jobs=jobs, broker=SpyBroker(price_map={"SPY": 500.0, "UNKNOWN": 100.0}))
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.OPEN, dry_run=True)
    assert "risk_checks_global" in res.failed_jobs

    root = tmp_path / "artifacts" / "2026-02-17"
    risk_report = json.loads((root / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert risk_report["status"] == "failed"
    assert "TRANSLATION_FAILED" in str(risk_report.get("reason"))
    assert risk_report["metrics"]["translation_failure_count"] == 1



def test_legacy_risk_report_rejected_when_v1_only_enabled(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root)
    _write(root / "reports" / "risk_checks.json", {"status": "ok", "metrics": {}, "limits": {}, "violations": []})
    ctx = _base_ctx(root, flags={"FF_RISK_REPORT_V1_ONLY": True})
    with pytest.raises(Exception):
        handle_order_build_and_route(ctx)


def test_legacy_risk_report_allowed_when_v1_only_disabled(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _seed_signals_and_targets(root)
    _write(root / "reports" / "risk_checks.json", {"status": "ok", "metrics": {}, "limits": {}, "violations": []})
    ctx = _base_ctx(root, flags={"FF_RISK_REPORT_V1_ONLY": False})
    out = handle_order_build_and_route(ctx)
    assert out["status"] == "ok"


def test_non_default_paths_not_changed_default_job_graph_intact():
    from backend.app.orchestrator.job_defs import default_jobs

    names = [j.name for j in default_jobs()]
    assert "risk_checks_global" in names
    assert "order_build_and_route" in names
