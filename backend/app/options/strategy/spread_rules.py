from backend.app.options.types import SpreadCandidate
from backend.app.models.types import DistributionForecast

def validate_spread(candidate: SpreadCandidate, min_width: float = 0.5, max_loss_cap: float = 1000.0) -> bool:
    if candidate.width < min_width:
        return False
        
    if candidate.max_loss > max_loss_cap:
        # In practice we adjust size, but if single lot risk is too high?
        # Usually max_loss per contract is width * 100 - credit.
        pass
        
    if candidate.short_strike >= candidate.long_strike:
        # Invalid vertical spread structure for Put Credit Spread (Short < Long? No. Short Put < Long Put is Debit.)
        # Put Credit Spread: Short Strike > Long Strike (Sell high strike put, Buy low strike put)
        # Wait, Puts: Higher strike = higher price. 
        # Bull Put Spread: Sell Higher Strike Put (ITM/ATM/OTM), Buy Lower Strike Put.
        # Net Credit.
        # Risk: Price drops below Short.
        
        # Example: Stock 100.
        # Sell 95 Put (Credit $1.00)
        # Buy 90 Put (Debit $0.20)
        # Net Credit $0.80.
        # Short Strike (95) > Long Strike (90).
        
        if candidate.short_strike <= candidate.long_strike:
             return False
             
    return True

def calculate_prob_profit(candidate: SpreadCandidate, forecast: DistributionForecast) -> float:
    # Use forecast quantiles to estimate Prob(Price > Short Strike @ Expiry)
    # This is rough mock logic using the median or quantiles.
    
    # We need forecast at Expiry (or close to DTE).
    # Forecast usually has 1D, 3D.
    # If DTE is 5, we might interpolate or use 3D.
    
    # Mock: return 0.7
    return 0.7
