from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from algae.core.config import PipelineConfig, load_config
from algae.core.paths import ArtifactPaths, ensure_artifact_dirs
from algae.data.eligibility.build import build_eligibility
from algae.data.features.build import build_features
from algae.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2
from algae.data.signals.build import build_signals
from algae.research.backtest_engine import BacktestConfig, BacktestEngine
from algae.trading.costs import CommissionModel, SlippageModel
from algae.research.walk_forward import build_walk_forward_splits
from algae.trading.risk import PortfolioTargetConfig
from backend.scripts._cli_utils import prepare_run, write_artifact_log


def _load_inputs(input_paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)


def _resolve_date(value: Optional[str], fallback: Optional[str]) -> date:
    if value:
        return date.fromisoformat(value)
    if fallback:
        return date.fromisoformat(fallback)
    raise ValueError("start/end date required")


def run(config: PipelineConfig, input_paths: list[Path], args: argparse.Namespace) -> None:
    paths, registry, run_dir = prepare_run(config)
    ensure_artifact_dirs(paths)
    canonical = _load_inputs(input_paths)

    signals_path = Path(args.signals) if args.signals else None
    eligibility_path = Path(args.eligibility) if args.eligibility else None
    if signals_path and signals_path.exists():
        signals = pd.read_parquet(signals_path)
    else:
        features = build_features(canonical)
        model = SimpleChronos2(FoundationModelConfig(enable_quantiles=config.enable_quantiles))
        priors = model.infer(canonical)
        signals = build_signals(features, priors)
    if eligibility_path and eligibility_path.exists():
        eligibility = pd.read_parquet(eligibility_path)
    else:
        eligibility = build_eligibility(canonical, date.fromisoformat(args.asof or args.end))

    start = _resolve_date(args.start, config.backtest.start)
    end = _resolve_date(args.end, config.backtest.end)

    target_config = PortfolioTargetConfig(
        top_k=args.top_k or config.portfolio.top_k,
        weight_method=config.portfolio.weight_method,
        softmax_temp=config.portfolio.softmax_temp,
        max_weight_per_name=config.portfolio.max_weight_per_name,
        max_names=config.portfolio.max_names,
        min_dollar_position=config.portfolio.min_dollar_position,
        cash_buffer_pct=config.portfolio.cash_buffer_pct,
    )
    backtest_config = BacktestConfig(
        fill_price_mode=args.fill_price_mode or config.backtest.fill_price_mode,
        rounding_policy=args.rounding_policy or config.backtest.rounding_policy,
        hold_days=config.portfolio.hold_days or 10,
        exit_policy=config.portfolio.exit_policy,
        slippage_model=SlippageModel(
            model=config.backtest.slippage_model,
            bps=config.backtest.slippage_bps,
            volume_impact=config.backtest.slippage_volume_impact,
        ),
        commission_model=CommissionModel(
            per_trade=config.backtest.commission_per_trade,
            per_share=config.backtest.commission_per_share,
            bps=config.backtest.commission_bps,
            min_commission=config.backtest.commission_min,
        ),
    )
    engine = BacktestEngine(target_config, backtest_config)

    output_root = paths.backtests / run_dir.name
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if args.walk_forward or config.backtest.walk_forward:
        signals["date"] = pd.to_datetime(signals["date"])
        splits = build_walk_forward_splits(
            signals["date"],
            train_window_days=args.train_window_days or config.backtest.train_window_days,
            test_window_days=args.test_window_days or config.backtest.test_window_days,
            step_days=args.step_days or config.backtest.step_days,
            expanding=config.backtest.expanding_window,
            holdout_pct=args.holdout_pct or config.backtest.holdout_pct,
        )
        fold_metrics = []
        for idx, split in enumerate(splits, start=1):
            result = engine.run(
                signals=signals,
                eligibility=eligibility,
                prices=canonical,
                start=split.test_start.date(),
                end=split.test_end.date(),
            )
            fold_dir = output_root / f"fold_{idx}"
            engine.write_artifacts(result, fold_dir)
            fold_metrics.append({"fold": idx, "metrics": result.metrics})
        (output_root / "fold_metrics.json").write_text(
            json.dumps(fold_metrics, indent=2), encoding="utf-8"
        )
    else:
        result = engine.run(signals, eligibility, canonical, start, end)
        engine.write_artifacts(result, output_root)

    registry.register("backtest", output_root, "v1")
    registry.dump(output_root / "artifacts.json")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--signals")
    parser.add_argument("--eligibility")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--asof")
    parser.add_argument("--walk_forward", action="store_true")
    parser.add_argument("--train_window_days", type=int)
    parser.add_argument("--test_window_days", type=int)
    parser.add_argument("--step_days", type=int)
    parser.add_argument("--holdout_pct", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--fill_price_mode")
    parser.add_argument("--rounding_policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_paths = [Path(path.strip()) for path in args.inputs.split(",")]
    run(config, input_paths, args)


if __name__ == "__main__":
    main()
