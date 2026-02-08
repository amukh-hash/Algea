
# Migration and Safety Toggles
ALLOW_LEGACY_READ = True  # read old artifacts if new missing
WRITE_BOTH_PATHS = True   # temporarily write new + old for parity
STRICT_SCHEMA = True      # fail fast on schema mismatch
FAIL_ON_MISSING_DIRS = True # do not silently continue

# Data Quality Gates
PRIORS_MIN_COVERAGE = 0.95
MAX_MISSING_BARS_RATE = 0.05

# Universe & Trading Parameters
UNIVERSE_REBALANCE = "monthly"
UNIVERSE_BUFFER_MAX = 525
UNIVERSE_TRADE_MAX = 450

# Model Parameters
CONTEXT_LEN = 512
PRIOR_HORIZON = 20
LABEL_HORIZON_TD = 10
SEQUENCE_LEN = 60

# Swing Strategy Configuration (Mirrored from Core)
SWING_LABEL_HORIZONS = [5, 10]
TEACHER_FORECAST_HORIZON_DAYS = 30
TEACHER_CONTEXT_DAYS = 252
TOPK_K = 20
COST_BPS = 10
RISK_LAMBDA_VOL = 0.25
RISK_KAPPA_DD = 0.50
TURNOVER_GAMMA = 0.05
PRIORS_ENABLED = True
