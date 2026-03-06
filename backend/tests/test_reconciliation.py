from datetime import date

from algae.trading.broker_base import BrokerAccount, BrokerPosition
from algae.trading.reconciliation import reconcile_positions


def test_reconciliation_normalizes_positions():
    positions = [BrokerPosition(ticker="AAA", quantity=10.0, avg_cost=100.0)]
    account = BrokerAccount(asof=date(2024, 1, 2), cash=1000.0, equity=2000.0, buying_power=3000.0)
    report = reconcile_positions(date(2024, 1, 2), positions, account)
    assert report.positions[0]["ticker"] == "AAA"
    assert report.account["equity"] == 2000.0
