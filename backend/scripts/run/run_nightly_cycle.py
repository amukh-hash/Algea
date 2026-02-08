from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from algaie.data.eligibility.build import build_eligibility
from algaie.data.features.build import build_features
from algaie.data.priors.build import build_priors
from algaie.data.signals.build import build_signals
from algaie.execution.equity_strategy import EquityStrategy
from algaie.execution.interfaces import ExecutionContext, SignalFrame
from algaie.execution.options.executor import NoopExecutor
from algaie.execution.options.strategy import StrikeSelector
from algaie.execution.options_strategy import OptionsStrategy
from backend.scripts._cli_utils import load_pipeline_config, prepare_run, write_artifact_log


def run(config_path: str, input_paths: list[Path], asof: date) -> None:
    config = load_pipeline_config(config_path)
    paths, registry, run_dir = prepare_run(config)
    canonical = pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)

    eligibility = build_eligibility(canonical, asof)
    eligibility_path = paths.eligibility / f"asof={asof}" / "eligibility.parquet"
    eligibility_path.parent.mkdir(parents=True, exist_ok=True)
    eligibility.to_parquet(eligibility_path, index=False)
    registry.register("eligibility", eligibility_path, "v1")

    features = build_features(canonical)
    features_path = paths.features / "features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_path, index=False)
    registry.register("features", features_path, "v1")

    priors = build_priors(canonical, asof)
    priors_path = paths.priors / f"date={asof}" / "priors.parquet"
    priors_path.parent.mkdir(parents=True, exist_ok=True)
    priors.to_parquet(priors_path, index=False)
    registry.register("priors", priors_path, "v1")

    signals = build_signals(features, priors)
    signals_path = paths.signals / f"date={asof}" / "signals.parquet"
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(signals_path, index=False)
    registry.register("signals", signals_path, "v1")

    equity_strategy = EquityStrategy()
    options_strategy = OptionsStrategy(selector=StrikeSelector(), executor=NoopExecutor())
    signal_frame = SignalFrame(signals)
    context = ExecutionContext(asof=asof, prices=canonical)
    equity_decisions = equity_strategy.run(signal_frame)
    options_decisions = options_strategy.run(signal_frame, context)

    summary = {
        "asof": str(asof),
        "signals": len(signals),
        "equity_decisions": len(equity_decisions),
        "options_decisions": len(options_decisions),
    }
    report_path = paths.reports / "nightly" / str(asof) / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    registry.register("nightly_summary", report_path, "v1")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--asof", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path.strip()) for path in args.inputs.split(",")]
    run(args.config, input_paths, date.fromisoformat(args.asof))


if __name__ == "__main__":
    main()
