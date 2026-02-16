from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.api.telemetry_routes import storage
from backend.app.telemetry.emitter import TelemetryEmitter
from backend.app.telemetry.schemas import EventLevel, EventType, RunStatus, RunType

SLEEVE_KEYS = [
    "pnl_net",
    "pnl_realized",
    "pnl_unrealized",
    "gross_exposure",
    "net_exposure",
    "leverage",
    "cash",
    "turnover",
    "cost_est",
    "slippage_est",
    "dd_current",
    "vol_realized",
    "vol_target",
    "alpha_raw",
    "alpha_blended",
    "gate_scale",
    "risk_scale",
    "latency_ms_decision_to_order",
    "latency_ms_order_to_fill",
]


def main(iterations: int, sleep_s: float) -> None:
    emitter = TelemetryEmitter(storage)
    sleeve_runs = [
        emitter.start_run(RunType.sleeve_live, f"Sleeve {name}", sleeve_name=name, meta={"venue": "paper"})
        for name in ("A", "B", "C")
    ]
    train_run = emitter.start_run(RunType.train, "Transformer retrain", meta={"epochs": 10})
    backtest_run = emitter.start_run(RunType.backtest, "Q1 backtest", meta={"window": "2025Q1"})

    for run_id in [*sleeve_runs, train_run, backtest_run]:
        emitter.set_status(run_id, RunStatus.running)

    for step in range(iterations):
        for i, run_id in enumerate(sleeve_runs):
            base = (i + 1) * 25
            emitter.emit_metric(run_id, "pnl_net", base + random.gauss(0, 5) + step * 0.4)
            emitter.emit_metric(run_id, "gross_exposure", 0.7 + random.random())
            emitter.emit_metric(run_id, "net_exposure", random.uniform(-0.5, 0.5))
            emitter.emit_metric(run_id, "dd_current", abs(random.gauss(0.05, 0.02)))
            emitter.emit_metric(run_id, "gate_scale", random.uniform(0.3, 1.0))
            for key in [k for k in SLEEVE_KEYS if k not in {"pnl_net", "gross_exposure", "net_exposure", "dd_current", "gate_scale"}]:
                emitter.emit_metric(run_id, key, random.uniform(0, 1))
            emitter.emit_event(
                run_id,
                EventLevel.info,
                EventType.DECISION_MADE,
                "Weights adjusted",
                payload={"pre": {"ES": 0.4}, "post": {"ES": 0.3}, "constraints": {"max_leverage": 1.2}},
            )

        emitter.emit_metric(train_run, "train_loss", max(0.01, 1.0 / (step + 1) + random.random() * 0.02))
        emitter.emit_metric(train_run, "val_loss", max(0.01, 1.2 / (step + 1) + random.random() * 0.03))
        emitter.emit_metric(train_run, "lr", 0.001)
        emitter.emit_metric(train_run, "grad_norm", random.uniform(0.1, 2.0))

        emitter.emit_metric(backtest_run, "cum_net", step * random.uniform(0.5, 1.2))
        emitter.emit_metric(backtest_run, "dd", random.uniform(0, 0.15))
        emitter.emit_metric(backtest_run, "turnover", random.uniform(0.1, 0.8))
        emitter.emit_metric(backtest_run, "costs", random.uniform(0.01, 0.05))

        if step % 5 == 0:
            emitter.emit_event(backtest_run, EventLevel.info, EventType.BACKTEST_COMPLETE, "Interim backtest checkpoint", payload={"step": step})
            emitter.emit_event(train_run, EventLevel.info, EventType.CHECKPOINT_SAVED, "Checkpoint saved", payload={"step": step})
        time.sleep(sleep_s)

    out = Path("backend/artifacts/dev")
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps({"generated_at": datetime.now(tz=timezone.utc).isoformat(), "iterations": iterations}, indent=2))
    emitter.register_artifact(train_run, str(summary), kind="report", mime="application/json")
    emitter.register_artifact(backtest_run, str(summary), kind="report", mime="application/json")

    for run_id in [*sleeve_runs, train_run, backtest_run]:
        emitter.set_status(run_id, RunStatus.completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()
    main(iterations=args.iterations, sleep_s=args.sleep)
