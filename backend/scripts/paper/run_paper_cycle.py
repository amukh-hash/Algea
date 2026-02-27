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
from algea.trading.broker_alpaca import AlpacaPaperBroker
from algea.trading.portfolio import Portfolio, Position
from algea.trading.reconciliation import reconcile_positions
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


def run(config_path: str, input_paths: list[Path], asof: date) -> None:
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

    broker = AlpacaPaperBroker.from_env()
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
    if len(intents) > config.broker.max_orders_per_day:
        raise RuntimeError("max_orders_per_day exceeded")
    for intent in intents:
        price_row = canonical[(canonical["date"] == pd.Timestamp(asof)) & (canonical["ticker"] == intent.ticker)]
        if price_row.empty:
            raise KeyError(f"Missing price for {intent.ticker} on {asof}")
        notional = float(price_row.iloc[0]["close"]) * intent.quantity
        if notional > config.broker.max_notional_per_order:
            raise RuntimeError("max_notional_per_order exceeded")
    if config.broker.dry_run:
        orders = []
    else:
        orders = broker.submit_orders(intents)

    paper_root = paths.paper / run_dir.name
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
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
    orders_df.to_parquet(paper_root / "orders.parquet", index=False)
    intents_df = pd.DataFrame(
        [
            {
                "date": intent.asof,
                "ticker": intent.ticker,
                "qty": intent.quantity,
                "side": intent.side,
                "reason": intent.reason,
            }
            for intent in intents
        ]
    )
    intents_df.to_parquet(paper_root / "order_intents.parquet", index=False)

    recon = reconcile_positions(asof, broker_positions, account)
    (paper_root / f"reconciliation_{asof}.json").write_text(
        json.dumps(
            {"asof": str(recon.asof), "positions": recon.positions, "account": recon.account}, indent=2
        ),
        encoding="utf-8",
    )

    registry.register("paper_orders", paper_root / "orders.parquet", "v1")
    registry.dump(paper_root / "artifacts.json")
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
