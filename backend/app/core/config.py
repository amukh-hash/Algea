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

# Execution Mode
EXECUTION_MODE = get_str("EXECUTION_MODE", "LEGACY", allowed=["LEGACY", "SHADOW", "RANKING"])

# Rebalancing Parameters
REBALANCE_CALENDAR = os.getenv("REBALANCE_CALENDAR", "MON,WED,FRI").split(",")
MIN_HOLD_DAYS = get_int("MIN_HOLD_DAYS", 5)
MAX_HOLD_DAYS = get_int("MAX_HOLD_DAYS", 10)

# Data Paths
# Data Paths
DATA_DIR = os.getenv("DATA_DIR", "backend/data")
PRIORS_DIR = os.path.join(DATA_DIR, "priors")
SIGNALS_DIR = os.path.join(DATA_DIR, "signals")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")

# Inference Mode Flags
NIGHTLY_MOCK_INFERENCE = get_bool("NIGHTLY_MOCK_INFERENCE", False)

# Training Configuration
TRAIN_START_DATE = "2016-01-01"
TRAIN_SPLIT_DATE = "2023-01-01" # Start of Validation
TEST_SPLIT_DATE = "2024-01-01"  # Start of Test
TRAIN_END_DATE = "2025-12-31"   # End of data window

# Model Hyperparameters
LOOKBACK_WINDOW_CHRONOS = 512
LOOKBACK_WINDOW_RANKER = 60
TARGET_HORIZON = 10
