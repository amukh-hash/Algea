from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List

from algaie.trading.broker_base import BrokerAccount, BrokerPosition


@dataclass(frozen=True)
class ReconciliationReport:
    asof: date
    positions: List[dict]
    account: dict


def reconcile_positions(
    asof: date,
    broker_positions: List[BrokerPosition],
    account: BrokerAccount,
) -> ReconciliationReport:
    positions = [
        {
            "ticker": position.ticker,
            "quantity": position.quantity,
            "avg_cost": position.avg_cost,
        }
        for position in broker_positions
    ]
    account_payload = {
        "cash": account.cash,
        "equity": account.equity,
        "buying_power": account.buying_power,
    }
    return ReconciliationReport(asof=asof, positions=positions, account=account_payload)
