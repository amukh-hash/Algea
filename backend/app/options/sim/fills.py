from backend.app.options.sim.policy import SimPolicy

def estimate_fill_price(
    bid: float,
    ask: float,
    action: str, # "buy" or "sell"
    policy: SimPolicy
) -> float:
    mid = (bid + ask) / 2.0
    half_spread = (ask - bid) / 2.0

    # Conservative penalty
    # If buying, we pay more. If selling, we get less.
    penalty = policy.fill_penalty_slippage

    if action == "buy":
        # Buy at mid + penalty, capped at Ask (or can exceed in bad liquidity?)
        # Let's assume mid + penalty.
        price = mid + penalty
        # In sim, we shouldn't get better than bid/ask unless we assume mid fills.
        # Conservative: Buy at Ask? Or Mid + Skew?
        # User said: "mid +/- penalty"
        return price

    elif action == "sell":
        price = mid - penalty
        return price

    return mid
