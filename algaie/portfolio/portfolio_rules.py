"""
Portfolio construction rules with turnover controls.

Provides buffer-zone hysteresis, slot-cap limits, and hold-bonus
persistence to keep turnover low for a 10-day swing-trading rebalance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""

    top_k: int = 50
    long_only: bool = True
    weight_scheme: str = "equal"          # "equal" only for now
    rebalance_horizon_days: int = 10

    # ── Turnover controls ─────────────────────────────────────────────
    buffer_entry_rank: int = 40           # new entries from ranks ≤ this
    buffer_exit_rank: int = 70            # held names exit if rank > this
    max_replacements: Optional[int] = 10  # max sells (and buys) per rebalance
    hold_bonus: float = 0.0              # added to score for currently held names
    turnover_limit: float = 0.50          # 1-way turnover target (for tuning)

    # ── Minimum hold ──────────────────────────────────────────────────
    min_hold_periods: int = 1             # min rebalance periods before eligible for exit

    def __post_init__(self):
        if self.buffer_entry_rank >= self.buffer_exit_rank:
            raise ValueError(
                f"buffer_entry_rank ({self.buffer_entry_rank}) must be < "
                f"buffer_exit_rank ({self.buffer_exit_rank})"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


def construct_portfolio(
    date_df,
    prev_holdings: Optional[Dict[str, float]],
    cfg: PortfolioConfig,
    hold_ages: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
    """Construct a portfolio for a single rebalance date.

    Parameters
    ----------
    date_df : DataFrame
        Rows for one date with at least columns ``symbol``, ``score_final``.
    prev_holdings : dict or None
        symbol → weight from previous rebalance.  None on first period.
    cfg : PortfolioConfig
        Construction rules.
    hold_ages : dict or None
        symbol → number of consecutive rebalance periods held.  None on first.

    Returns
    -------
    weights : dict
        symbol → weight (positive, summing to 1.0).
    new_hold_ages : dict
        symbol → updated hold-age counter.
    diagnostics : dict
        n_held, n_new, n_exits, turnover_1way, n_available.
    """
    if prev_holdings is None:
        prev_holdings = {}
    if hold_ages is None:
        hold_ages = {}

    df = date_df.copy()
    n_available = len(df)

    # ── 1.  Apply hold bonus to currently held names ──────────────────
    held_set = set(prev_holdings.keys())
    if cfg.hold_bonus > 0 and held_set:
        bonus = df["symbol"].isin(held_set).astype(float) * cfg.hold_bonus
        df = df.copy()
        df["score_adj"] = df["score_final"] + bonus
    else:
        df["score_adj"] = df["score_final"]

    # ── 2.  Rank by adjusted score (1 = best) ────────────────────────
    df = df.sort_values("score_adj", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    sym_rank = dict(zip(df["symbol"], df["rank"]))

    # ── 3.  Buffer-zone logic ─────────────────────────────────────────
    #  - Held names survive unless rank > exit threshold
    #    AND they've been held for at least min_hold_periods
    #  - New entries only from rank ≤ entry threshold
    survivors = []
    exits = []
    for sym in prev_holdings:
        if sym not in sym_rank:
            # Symbol disappeared from universe
            exits.append(sym)
            continue
        rank = sym_rank[sym]
        age = hold_ages.get(sym, 1)
        if rank > cfg.buffer_exit_rank and age >= cfg.min_hold_periods:
            exits.append(sym)
        else:
            survivors.append(sym)

    # Candidates: not currently held, rank ≤ entry threshold
    candidates = [
        sym for sym, rank in sym_rank.items()
        if sym not in held_set and rank <= cfg.buffer_entry_rank
    ]
    # Sort candidates by rank (best first)
    candidates.sort(key=lambda s: sym_rank[s])

    # ── 4.  Slot cap: limit replacements ──────────────────────────────
    n_slots_open = cfg.top_k - len(survivors)
    n_slots_open = max(n_slots_open, 0)

    if cfg.max_replacements is not None:
        max_exits = min(len(exits), cfg.max_replacements)
        # If we have more exits than max_replacements, keep worst exits
        if len(exits) > max_exits:
            # Sort exits by rank (worst rank = most deserving of exit)
            exits.sort(key=lambda s: sym_rank.get(s, 9999), reverse=True)
            forced_keep = exits[max_exits:]   # these survive despite bad rank
            exits = exits[:max_exits]
            survivors.extend(forced_keep)

        # Recalculate open slots
        n_slots_open = cfg.top_k - len(survivors)
        n_slots_open = max(n_slots_open, 0)

        max_buys = min(cfg.max_replacements, n_slots_open)
        candidates = candidates[:max_buys]
    else:
        candidates = candidates[:n_slots_open]

    # ── 5.  Fill portfolio ────────────────────────────────────────────
    portfolio_syms = survivors + candidates

    # If still have open slots and not enough candidates from entry zone,
    # fill with next-best-ranked available names
    if len(portfolio_syms) < cfg.top_k:
        already = set(portfolio_syms)
        fill = [
            sym for sym, rank in sym_rank.items()
            if sym not in already
        ]
        fill.sort(key=lambda s: sym_rank[s])
        needed = cfg.top_k - len(portfolio_syms)
        portfolio_syms.extend(fill[:needed])

    # Trim if over (shouldn't happen, but defensive)
    portfolio_syms = portfolio_syms[:cfg.top_k]

    # ── 6.  Assign weights ────────────────────────────────────────────
    k = len(portfolio_syms)
    if k == 0:
        return {}, {}, {"n_held": 0, "n_new": 0, "n_exits": 0,
                        "turnover_1way": 0.0, "n_available": n_available}

    if cfg.weight_scheme == "equal":
        w = 1.0 / k
        weights = {sym: w for sym in portfolio_syms}
    else:
        raise ValueError(f"Unknown weight_scheme: {cfg.weight_scheme}")

    # ── 7.  Compute diagnostics ───────────────────────────────────────
    new_set = set(portfolio_syms)
    old_set = set(prev_holdings.keys())
    n_new = len(new_set - old_set)
    n_exits_final = len(old_set - new_set)
    n_held = len(new_set & old_set)

    # 1-way turnover = 0.5 * sum(|new_w - old_w|)
    all_syms = new_set | old_set
    turnover_1way = 0.5 * sum(
        abs(weights.get(s, 0.0) - prev_holdings.get(s, 0.0))
        for s in all_syms
    )

    # ── 8.  Update hold ages ──────────────────────────────────────────
    new_hold_ages = {}
    for sym in portfolio_syms:
        if sym in hold_ages:
            new_hold_ages[sym] = hold_ages[sym] + 1
        else:
            new_hold_ages[sym] = 1

    diagnostics = {
        "n_held": n_held,
        "n_new": n_new,
        "n_exits": n_exits_final,
        "turnover_1way": round(turnover_1way, 6),
        "n_available": n_available,
        "n_portfolio": k,
    }

    return weights, new_hold_ages, diagnostics
