from datetime import date

from algaie.trading.orders import Fill
from algaie.trading.portfolio import Portfolio, Position


def test_partial_reduce_long_realized_pnl():
    p = Portfolio(cash=0.0, positions={"A": Position("A", 10, 100.0, date(2026, 1, 1))})
    p.update_from_fill(Fill(asof=date(2026, 1, 2), ticker="A", quantity=4, price=110.0, side="sell"), date(2026, 1, 2))
    assert p.positions["A"].quantity == 6
    assert p.positions["A"].avg_cost == 100.0
    assert round(p.positions["A"].realized_pnl, 6) == 40.0


def test_partial_reduce_short_return_sign_positive_when_profitable():
    p = Portfolio(cash=0.0, positions={"A": Position("A", -10, 100.0, date(2026, 1, 1))})
    p.update_from_fill(Fill(asof=date(2026, 1, 2), ticker="A", quantity=4, price=90.0, side="buy"), date(2026, 1, 2))
    last = p.trade_log[-1]
    assert last["pnl"] > 0
    assert last["ret"] > 0


def test_flip_position_conservation_shape():
    p = Portfolio(cash=0.0, positions={"A": Position("A", 5, 100.0, date(2026, 1, 1))})
    p.update_from_fill(Fill(asof=date(2026, 1, 2), ticker="A", quantity=7, price=95.0, side="sell"), date(2026, 1, 2))
    assert p.positions["A"].quantity == -2
    assert p.positions["A"].avg_cost == 95.0
