def calculate_spread_payoff(
    expiry_price: float,
    short_strike: float,
    long_strike: float,
    strategy: str = "put_credit_spread"
) -> float:
    """
    Returns value of the spread at expiry (per share).
    Positive = Liability (what you pay to close or settle).
    Wait, usually we track PnL.

    Let's return the intrinsic value of the spread (what it costs to buy back).
    For Credit Spread:
    Profit = Credit Received - Closing Cost.
    Closing Cost = Max(0, Short Strike - Price) - Max(0, Long Strike - Price) [For Put Spread]
    """
    if strategy == "put_credit_spread":
        # Short Put Value (Liability)
        short_val = max(0.0, short_strike - expiry_price)
        # Long Put Value (Asset)
        long_val = max(0.0, long_strike - expiry_price)

        # Net value (Cost to close)
        return max(0.0, short_val - long_val)

    return 0.0
