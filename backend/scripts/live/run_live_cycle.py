from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from algea.core.config import load_config
from algea.core.paths import ArtifactPaths, ensure_artifact_dirs
from algea.data.eligibility.build import build_eligibility
from algea.data.features.build import build_features
from algea.data.signals.build import build_signals
from algea.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2
from algea.trading.broker_ibkr import IBKRLiveBroker
from algea.trading.portfolio import Portfolio, Position
from algea.trading.risk import PortfolioTargetBuilder, PortfolioTargetConfig
from backend.scripts._cli_utils import prepare_run, write_artifact_log


def _load_inputs(input_paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path) for path in input_paths], ignore_index=True)


def _build_signals(canonical: pd.DataFrame, config) -> pd.DataFrame:
    features = build_features(canonical)
    model = SimpleChronos2(FoundationModelConfig(enable_quantiles=config.enable_quantiles))
    priors = model.infer(canonical)
    return build_signals(features, priors)


def _portfolio_from_broker(positions) -> Portfolio:
    portfolio = Portfolio(cash=0.0)
    for position in positions:
        portfolio.positions[position.ticker] = Position(
            ticker=position.ticker,
            quantity=position.quantity,
            avg_cost=position.avg_cost,
            entry_date=date.today(),
        )
    return portfolio


def run(config_path: str, input_paths: list[Path], asof: date, mode: str) -> None:
    config = load_config(config_path)
    paths, registry, run_dir = prepare_run(config)
    ensure_artifact_dirs(paths)
    canonical = _load_inputs(input_paths)

    signals = _build_signals(canonical, config)
    eligibility = build_eligibility(canonical, asof)
    target_config = PortfolioTargetConfig(
        top_k=config.portfolio.top_k,
        weight_method=config.portfolio.weight_method,
        softmax_temp=config.portfolio.softmax_temp,
        max_weight_per_name=config.portfolio.max_weight_per_name,
        max_names=config.portfolio.max_names,
        min_dollar_position=config.portfolio.min_dollar_position,
        cash_buffer_pct=config.portfolio.cash_buffer_pct,
    )
    target_builder = PortfolioTargetBuilder(target_config)

    live_root = paths.live / run_dir.name
    live_root.mkdir(parents=True, exist_ok=True)
    (live_root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if mode == "noop":
        targets = target_builder.build_targets(
            signals=signals,
            eligibility=eligibility,
            prices=canonical,
            asof=pd.Timestamp(asof),
            total_equity=1.0,
        )
        intents = targets.to_dict(orient="records")
        (live_root / "intents.json").write_text(json.dumps(intents, indent=2), encoding="utf-8")
    else:
        broker = IBKRLiveBroker.from_env()
        account = broker.get_account()
        broker_positions = broker.get_positions()
        portfolio = _portfolio_from_broker(broker_positions)
        targets = target_builder.build_targets(
            signals=signals,
            eligibility=eligibility,
            prices=canonical,
            asof=pd.Timestamp(asof),
            total_equity=account.equity,
        )
        intents = portfolio.build_order_intents(
            asof=asof,
            target_weights=targets,
            prices=canonical,
            equity=account.equity,
            rounding_policy=config.broker.rounding_policy,
        )
        orders = broker.submit_orders(intents)
        orders_df = pd.DataFrame(
            [
                {
                    "date": order.asof,
                    "ticker": order.ticker,
                    "qty": order.quantity,
                    "side": order.side,
                    "status": order.status,
                    "broker_order_id": order.broker_order_id,
                    "client_order_id": order.client_order_id,
                }
                for order in orders
            ]
        )
        orders_df.to_parquet(live_root / "orders.parquet", index=False)
        registry.register("live_orders", live_root / "orders.parquet", "v1")

    registry.dump(live_root / "artifacts.json")
    write_artifact_log(registry, run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--mode", choices=["noop", "ibkr"], default="noop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path.strip()) for path in args.inputs.split(",")]
    run(args.config, input_paths, date.fromisoformat(args.asof), args.mode)


if __name__ == "__main__":
    main()
