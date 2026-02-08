from typing import Dict
from backend.app.options.gate.context import OptionsContext

def compute_label(ctx: OptionsContext, future_price: float, expiry_price: float) -> int:
    """
    1 if profitable spread could have been opened, 0 otherwise.
    Requires looking into the future.
    """
    # Synthetic label logic:
    # If price stayed above (Price - 5%) for next 5 days?
    
    # Simple rule: Did price go up or stay flat?
    if future_price >= ctx.underlying_price * 0.98:
        return 1
    return 0
