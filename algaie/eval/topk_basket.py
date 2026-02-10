"""
Top-K basket evaluation — equal-weight portfolio simulation.

Ported from deprecated/backend_app_snapshot/eval/topk_basket_eval.py.
Keeps the simple ``topk_basket`` helper for backward compatibility.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Original stub (backward-compat)
# ---------------------------------------------------------------------------

def topk_basket(signals: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    return signals.sort_values(["date", "rank"]).groupby("date").head(k).copy()


# ---------------------------------------------------------------------------
# Full evaluator
# ---------------------------------------------------------------------------

class TopKBasketEvaluator:
    """
    Simulates an equal-weight Top-K basket strategy.

    Call ``step()`` once per trading day, then ``get_metrics()`` for summary.

    Parameters
    ----------
    k : number of top-ranked assets per day
    cost_bps : one-way transaction cost in basis points
    """

    def __init__(self, k: int = 20, cost_bps: int = 10) -> None:
        self.k = k
        self.cost_bps = cost_bps
        self.history: List[Dict] = []

    def step(
        self,
        date: str,
        scores: Dict[str, float],
        targets: Dict[str, float],
    ) -> None:
        """
        Record one day's selection.

        Parameters
        ----------
        scores : ``{symbol: model_score}``
        targets : ``{symbol: forward_return}``
        """
        top_items = heapq.nlargest(self.k, scores.items(), key=lambda x: x[1])
        tickers = [x[0] for x in top_items]

        rets = [targets.get(t, 0.0) for t in tickers]
        raw_ret = float(np.mean(rets)) if rets else 0.0

        # Turnover
        turnover = 0.0
        if self.history:
            prev = set(self.history[-1]["tickers"])
            curr = set(tickers)
            overlap = len(prev & curr)
            turnover = (self.k - overlap) / self.k

        cost = turnover * 2 * (self.cost_bps / 10_000.0)
        net_ret = raw_ret - cost

        self.history.append(
            {
                "date": date,
                "raw_ret": raw_ret,
                "net_ret": net_ret,
                "turnover": turnover,
                "cost": cost,
                "tickers": tickers,
            }
        )

    def get_metrics(self) -> Dict[str, float]:
        if not self.history:
            return {}

        df = pd.DataFrame(self.history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["cum_ret"] = (1 + df["net_ret"]).cumprod()

        total_ret = float(df["cum_ret"].iloc[-1] - 1.0)
        avg_ret = df["net_ret"].mean()
        std_ret = df["net_ret"].std()
        sharpe = float((avg_ret / (std_ret + 1e-9)) * np.sqrt(252))

        peaks = df["cum_ret"].cummax()
        dds = (peaks - df["cum_ret"]) / peaks.clip(1e-9)
        max_dd = float(dds.max())

        return {
            "total_ret": total_ret,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "avg_turnover": float(df["turnover"].mean()),
            "final_cum": float(df["cum_ret"].iloc[-1]),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

