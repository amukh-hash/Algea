from __future__ import annotations


def estimate_cost_return(
    half_spread_entry: float,
    half_spread_exit: float,
    commission_cash: float,
    contract_notional: float,
    impact_k: float = 0.0,
    contracts: int = 0,
    depth: float = 1.0,
    vol: float = 0.0,
) -> float:
    base_cash = half_spread_entry + half_spread_exit + commission_cash
    impact = impact_k * (abs(contracts) / max(depth, 1.0)) * vol * contract_notional
    return (base_cash + impact) / max(contract_notional, 1.0)
