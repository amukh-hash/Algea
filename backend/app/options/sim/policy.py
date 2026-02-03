from dataclasses import dataclass

@dataclass
class SimPolicy:
    fill_penalty_slippage: float = 0.02 # $ per share
    take_profit_pct: float = 0.50 # 50% of max profit
    stop_loss_pct: float = 2.0 # 200% of credit received (loss = 2x credit)

    # Or defined as multiple of max loss?
    # Common: Stop @ 2x credit or 3x credit.

    hold_to_expiry: bool = False
    close_at_dte: int = 0 # Close on expiry day or before?

    # Execution
    use_mid_price: bool = True
