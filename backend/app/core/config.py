import os

def get_str(key: str, default: str, allowed: list = None) -> str:
    val = os.getenv(key, default)
    if allowed and val not in allowed:
        raise ValueError(f"Config {key} must be one of {allowed}, got {val}")
    return val

def get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")

def get_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))

def get_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))

# Core Options Configuration
ENABLE_OPTIONS = get_bool("ENABLE_OPTIONS", False)
OPTIONS_MODE = get_str("OPTIONS_MODE", "off", allowed=["off", "monitor", "paper", "live"])
OPTIONS_SEED = get_int("OPTIONS_SEED", 42)
OPTIONS_DATA_VERSION = get_str("OPTIONS_DATA_VERSION", "v1")

# Equities Simulation
EQUITIES_MODE = get_str("EQUITIES_MODE", "real", allowed=["real", "mock"])
