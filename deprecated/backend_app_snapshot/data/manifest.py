import os
import json
from typing import List, Dict

# Paths
MANIFEST_PATH = os.path.join("backend", "data", "microcosm_manifest.json")

# Default fallback universe (for testing/bootstrap)
DEFAULT_UNIVERSE = {
    "leaders": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA"],
    "vol_beasts": ["AMD", "NFLX", "COIN"],
    "liquidity_proxies": ["SPY", "QQQ", "IWM"]
}

def load_manifest() -> Dict[str, List[str]]:
    """Loads the universe manifest from JSON or returns default."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading manifest: {e}")
            return DEFAULT_UNIVERSE
    return DEFAULT_UNIVERSE

def get_trade_universe() -> List[str]:
    """Returns the primary trading universe (e.g. 30 tickers)."""
    # For Phase 1, we might just treat 'leaders' as the trade set
    data = load_manifest()
    return sorted(list(set(data.get('leaders', []))))

def get_context_universe() -> List[str]:
    """Returns the full context universe (e.g. 120 tickers)."""
    data = load_manifest()
    all_tickers = []
    for group in data.values():
        if isinstance(group, list):
            all_tickers.extend(group)
    return sorted(list(set(all_tickers)))

def get_microcosm_hash() -> str:
    """Returns a hash of the current universe for versioning."""
    import hashlib
    universe = get_context_universe()
    return hashlib.md5("".join(universe).encode()).hexdigest()
