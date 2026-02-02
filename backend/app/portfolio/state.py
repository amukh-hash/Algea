from dataclasses import dataclass, field
from typing import Dict
from backend.app.risk.posture import RiskPosture

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity

@dataclass
class PortfolioState:
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    posture: RiskPosture = RiskPosture.NORMAL

    @property
    def total_equity(self) -> float:
        mv = sum(p.market_value for p in self.positions.values())
        return self.cash + mv

    def update_price(self, ticker: str, price: float):
        if ticker in self.positions:
            self.positions[ticker].current_price = price

    def add_fill(self, ticker: str, quantity: float, price: float):
        if quantity == 0: return

        if ticker not in self.positions:
            if quantity > 0:
                self.positions[ticker] = Position(ticker, quantity, price, price)
                self.cash -= quantity * price
            else:
                # Short selling not supported in this simple logic yet?
                # Or closing non-existent position?
                pass
        else:
            pos = self.positions[ticker]
            cost = quantity * price

            if quantity > 0: # Buy
                # Avg Price update
                total_cost = pos.quantity * pos.avg_price + cost
                new_qty = pos.quantity + quantity
                pos.avg_price = total_cost / new_qty
                pos.quantity = new_qty
                self.cash -= cost
            else: # Sell
                # Realize PnL logic needed?
                # FIFO or Avg Cost. Avg Cost:
                # Sell reduces quantity, doesn't change avg price.
                self.cash -= cost # quantity is negative, so cash increases
                pos.quantity += quantity

            if pos.quantity == 0:
                del self.positions[ticker]
