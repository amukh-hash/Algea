from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from algaie.research.equity_curve import build_equity_curve
from algaie.research.metrics import compute_metrics
from algaie.research.reports import build_summary
from algaie.research.trade_log import build_trade_log
from algaie.trading.calendar import trading_days
from algaie.trading.costs import CommissionModel, SlippageModel
from algaie.trading.fills import ExecutionSimulator, FillConfig
from algaie.trading.portfolio import Portfolio
from algaie.trading.risk import PortfolioTargetBuilder, PortfolioTargetConfig


@dataclass(frozen=True)
class BacktestConfig:
    fill_price_mode: str = "next_open"
    rounding_policy: str = "fractional"
    initial_cash: float = 1_000_000.0
    hold_days: int = 10
    exit_policy: str = "hybrid"
    slippage_model: SlippageModel = SlippageModel()
    commission_model: CommissionModel = CommissionModel()


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict


class BacktestEngine:
    def __init__(
        self,
        target_config: PortfolioTargetConfig,
        backtest_config: BacktestConfig,
    ) -> None:
        self.target_builder = PortfolioTargetBuilder(target_config)
        self.backtest_config = backtest_config
        self.simulator = ExecutionSimulator(
            FillConfig(
                price_mode=backtest_config.fill_price_mode,
                slippage_model=backtest_config.slippage_model,
            )
        )

    def run(
        self,
        signals: pd.DataFrame,
        eligibility: pd.DataFrame,
        prices: pd.DataFrame,
        start: date,
        end: date,
    ) -> BacktestResult:
        from algaie.data.common import ensure_datetime

        prices = ensure_datetime(prices.copy())
        signals = ensure_datetime(signals.copy())
        eligibility = ensure_datetime(eligibility.copy())

        calendar = trading_days(start, end)
        portfolio = Portfolio(cash=self.backtest_config.initial_cash)
        snapshots: List[dict] = []
        orders_log: List[dict] = []

        for idx, current in enumerate(calendar):
            current_ts = pd.Timestamp(current)
            next_date = calendar[idx + 1] if idx + 1 < len(calendar) else None
            equity = portfolio.total_equity(prices, current)
            targets = self.target_builder.build_targets(
                signals=signals,
                eligibility=eligibility,
                prices=prices,
                asof=current_ts,
                total_equity=equity,
            )
            targets = self._apply_exit_policy(targets, portfolio, current)
            intents = portfolio.build_order_intents(
                asof=current,
                target_weights=targets,
                prices=prices,
                equity=equity,
                rounding_policy=self.backtest_config.rounding_policy,
            )
            fills, orders = self.simulator.simulate(intents, prices, current, next_date)
            for fill in fills:
                portfolio.update_from_fill(fill, fill.asof)
                commission = self.backtest_config.commission_model.commission(
                    shares=fill.quantity,
                    notional=fill.price * fill.quantity,
                )
                portfolio.cash -= commission
            for order in orders:
                orders_log.append(
                    {
                        "date": order.asof,
                        "ticker": order.ticker,
                        "qty": order.quantity,
                        "side": order.side,
                        "fill_px": order.fill_price,
                        "status": order.status,
                        "client_order_id": order.client_order_id,
                        "commission": float(
                            self.backtest_config.commission_model.commission(
                                shares=order.quantity,
                                notional=(order.fill_price or 0.0) * order.quantity,
                            )
                        ),
                    }
                )
            snapshot = portfolio.snapshot(prices, current)
            snapshots.append({
                "date": snapshot.asof,
                "equity": snapshot.equity,
                "cash": snapshot.cash,
                "gross_exposure": snapshot.gross_exposure,
                "net_exposure": snapshot.net_exposure,
            })

        equity_curve = build_equity_curve(snapshots)
        trades = build_trade_log(portfolio.trade_log)
        orders_df = pd.DataFrame(orders_log)
        metrics_result = compute_metrics(equity_curve, trades)
        metrics = {"equity": metrics_result.equity, "trades": metrics_result.trades}
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            orders=orders_df,
            metrics=metrics,
        )

    def write_artifacts(self, result: BacktestResult, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_parquet(destination / "equity_curve.parquet", index=False)
        result.trades.to_parquet(destination / "trades.parquet", index=False)
        result.orders.to_parquet(destination / "orders.parquet", index=False)
        (destination / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        summary = build_summary(result.metrics)
        (destination / "summary.md").write_text(summary, encoding="utf-8")

    def _apply_exit_policy(self, targets: pd.DataFrame, portfolio: Portfolio, asof: date) -> pd.DataFrame:
        exit_policy = self.backtest_config.exit_policy
        hold_days = self.backtest_config.hold_days
        exit_tickers = set()
        for ticker, position in portfolio.positions.items():
            held_days = (asof - position.entry_date).days
            if exit_policy in {"time", "hybrid"} and held_days >= hold_days:
                exit_tickers.add(ticker)
        if exit_policy in {"signal", "hybrid"}:
            target_tickers = set(targets["ticker"].tolist()) if not targets.empty else set()
            for ticker in portfolio.positions:
                if ticker not in target_tickers:
                    exit_tickers.add(ticker)
        if not exit_tickers or targets.empty:
            return targets
        return targets[~targets["ticker"].isin(exit_tickers)].copy()
