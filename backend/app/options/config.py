from backend.app.core.config import get_str, get_float, get_int

# DTE Buckets
OPTIONS_DTE_BUCKET = get_str("OPTIONS_DTE_BUCKET", "3-7") # "0-1", "3-7", "14-21"

# Risk & Exposure
OPTIONS_MAX_RISK = get_float("OPTIONS_MAX_RISK", 0.02) # % of equity per trade
OPTIONS_MAX_CLUSTER_EXPOSURE = get_float("OPTIONS_MAX_CLUSTER_EXPOSURE", 0.10) # Max exposure to correlated assets

# Liquidity
OPTIONS_MIN_VOLUME = get_int("OPTIONS_MIN_VOLUME", 50)
OPTIONS_MIN_OI = get_int("OPTIONS_MIN_OI", 100)

# Gate Tuning Defaults
OPTIONS_GATE_PRECISION_TARGET = get_float("OPTIONS_GATE_PRECISION_TARGET", 0.70)
