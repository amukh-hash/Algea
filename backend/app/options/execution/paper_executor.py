from typing import Optional, List
from datetime import datetime
from backend.app.options.execution.executor import OptionsExecutor
from backend.app.options.types import OptionsDecision, OptionsPosition, SpreadCandidate

class PaperExecutor(OptionsExecutor):
    def __init__(self):
        self.positions: List[OptionsPosition] = []

    def execute(self, decision: OptionsDecision) -> Optional[OptionsPosition]:
        if decision.action == "OPEN" and decision.candidate:
            # Simulate Fill
            pos = OptionsPosition(
                ticker=decision.candidate.underlying_ticker,
                expiry=decision.candidate.expiry_date,
                short_strike=decision.candidate.short_strike,
                long_strike=decision.candidate.long_strike,
                quantity=decision.quantity,
                entry_price=decision.candidate.net_credit,
                current_price=decision.candidate.net_credit,
                open_timestamp=datetime.now() # Mock time
            )
            self.positions.append(pos)
            print(f"[PAPER] Opened position: {pos}")
            return pos

        elif decision.action == "CLOSE":
            # Find and remove
            # Simplified logic
            if self.positions:
                closed = self.positions.pop(0)
                print(f"[PAPER] Closed position: {closed}")
                return closed

        return None

    def get_positions(self) -> List[OptionsPosition]:
        return self.positions
