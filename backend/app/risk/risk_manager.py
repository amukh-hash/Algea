from typing import Dict, Any, Optional
from backend.app.risk.types import RiskDecision, ActionType
from backend.app.risk.posture import RiskPosture
from backend.app.risk.crash_override import CrashOverride
from backend.app.portfolio.state import PortfolioState
from backend.app.models.signal_types import ModelSignal

class RiskManager:
    def __init__(self):
        self.crash_override = CrashOverride()

    def evaluate(self, ticker: str, signal: ModelSignal, portfolio: PortfolioState, breadth_data: Dict[str, float]) -> RiskDecision:
        # 1. Determine Global Posture
        bpi = breadth_data.get("bpi", 50.0)
        ad_slope = breadth_data.get("ad_slope", 0.0)

        posture = self.crash_override.check(bpi, ad_slope)

        # Override portfolio posture?
        # Usually we update portfolio posture.
        # But here we return a decision for a ticker.

        # 2. Check Signal
        # Use 3D horizon for swing?
        # If 3D is Up (prob > 0.6) -> Buy
        # If 3D is Down (prob < 0.4) -> Sell

        prob_up = signal.direction_probs.get("3D", 0.5)

        current_pos = portfolio.positions.get(ticker)
        qty_held = current_pos.quantity if current_pos else 0

        # 3. Decision Logic
        action = ActionType.NO_NEW_RISK
        quantity = 0.0
        reason = f"Posture: {posture}, Prob: {prob_up:.2f}"

        # Defensive: Liquidation or Hold only
        if posture == RiskPosture.DEFENSIVE:
            if qty_held > 0:
                # Reduce or Liquidate?
                # If signal is bad, liquidate.
                if prob_up < 0.4:
                    action = ActionType.LIQUIDATE
                    quantity = -qty_held
                    reason += " | Defensive + Bearish Signal"
                else:
                    action = ActionType.HOLD # Or reduce
            else:
                action = ActionType.NO_NEW_RISK

        # Cautious: No new buys? Or reduced size?
        elif posture == RiskPosture.CAUTIOUS:
            if qty_held > 0:
                if prob_up < 0.4:
                    action = ActionType.SELL # Exit
                    quantity = -qty_held
            else:
                # Entry allowed?
                if prob_up > 0.7: # High conviction only
                    action = ActionType.BUY
                    quantity = 10.0 # Placeholder size
                    reason += " | Cautious Entry"

        # Normal
        else:
            if qty_held > 0:
                if prob_up < 0.4:
                    action = ActionType.SELL
                    quantity = -qty_held
            else:
                if prob_up > 0.6:
                    action = ActionType.BUY
                    quantity = 20.0 # Full size placeholder

        return RiskDecision(ticker, action, quantity, reason)
