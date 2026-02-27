from backend.app.strategies.selector.portfolio_constructor import SelectorPortfolioConstructor


def test_selector_portfolio_constraints():
    c = SelectorPortfolioConstructor(per_symbol_cap=0.05)
    targets = c.construct({"A": 10, "B": 9, "C": -8, "D": -9})
    assert all(abs(t["target_weight"]) <= 0.05 for t in targets)
