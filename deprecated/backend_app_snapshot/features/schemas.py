
from typing import Dict, Any

# B1: Security Master
SCHEMA_SECURITY_MASTER = {
    "columns": {
        "symbol": "string",
        "primary_id": "string", # FIGI/PermID
        "exchange": "string",
        "asset_type": "string", # COMMON, ADR, ETF
        "ipo_date": "datetime64[ns]",
        "delist_date": "datetime64[ns]", # Nullable
        "sector": "string", # Nullable
        "industry": "string" # Nullable
    },
    "nullable": ["delist_date", "sector", "industry"],
    "index": ["symbol"] # Logic often uses symbol as index or lookup
}

# B2: Daily Bars (Partitioned)
SCHEMA_OHLCV_ADJ = {
    "columns": {
        "date": "datetime64[ns]",
        "open_adj": "float64",
        "high_adj": "float64",
        "low_adj": "float64",
        "close_adj": "float64",
        "volume": "int64",
        "dollar_volume": "float64",
        "split_factor": "float64", # Nullable
        "dividend": "float64",     # Nullable
        "data_version": "string"
    },
    "nullable": ["split_factor", "dividend"],
    "index": ["date"]
}

# B5: Universe Manifest
SCHEMA_UNIVERSE_MANIFEST = {
    "columns": {
        "asof_date": "datetime64[ns]",
        "symbol": "string",
        "eligible": "bool",
        "reason_code": "string", # Ok, PRICE<5, etc
        "close_adj": "float64",
        "adv20_median": "float64",
        "vol20_median": "float64",
        "ipo_age_td": "int32",
        "sector": "string",
        "universe_version": "string"
    },
    "nullable": ["sector"],
    "index": ["symbol"] # per date file
}

# B6: FeatureFrame
SCHEMA_FEATUREFRAME = {
    "columns": {
        "date": "datetime64[ns]",
        "symbol": "string",
        "ret_1d": "float64",
        "ret_3d": "float64",
        "ret_5d": "float64",
        "ret_10d": "float64",
        "ret_20d": "float64",
        "vol_20d": "float64",
        "ewma_vol_10d": "float64",
        "ewma_vol_20d": "float64",
        "parkinson_vol_10d": "float64",
        "parkinson_vol_20d": "float64",
        "range_pct_1d": "float64",
        "vol_chg_1d": "float64",
        "dollar_vol_20d": "float64",
        "adv_dollars_20d": "float64",
        "volume_z_20d": "float64", # Nullable
        "volume_stability_20d": "float64",
        "gap_freq_20d": "float64",
        "rel_ret_1d": "float64",
        "beta_spy_20d": "float64",
        "xsec_disp_ret_1d": "float64",
        # Covariates - allow dynamic? Or fixed?
        # Fixed for now as per spec
        "spy_ret_1d": "float64",
        "qqq_ret_1d": "float64",
        "iwm_ret_1d": "float64",
        "vix_level": "float64",
        "rate_proxy": "float64",
        "market_breadth_ad": "float64",
        "feature_version": "string",
        "data_version": "string"
    },
    "nullable": [
        "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "vol_20d", "vol_chg_1d",
        "ewma_vol_10d", "ewma_vol_20d",
        "parkinson_vol_10d", "parkinson_vol_20d",
        "range_pct_1d",
        "dollar_vol_20d", "adv_dollars_20d", "volume_z_20d", "volume_stability_20d", "gap_freq_20d",
        "rel_ret_1d", "beta_spy_20d", "xsec_disp_ret_1d",
        "spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d",
        "vix_level", "rate_proxy", "market_breadth_ad"
    ],
}

# B7: Priors
SCHEMA_PRIORS = {
    "columns": {
        "date": "datetime64[ns]",
        "symbol": "string",
        "drift": "float64",
        "vol_forecast": "float64",
        "tail_risk": "float64",
        "trend_conf": "float64",
        "prior_version": "string",
        "chronos_model_id": "string",
        "context_len": "int32",
        "horizon": "int32"
    },
    "nullable": []
}

# B8: Labels
SCHEMA_LABELS = {
    "columns": {
        "date": "datetime64[ns]",
        "symbol": "string",
        "fwd_ret": "float64",
        "fwd_up": "int8",
        "fwd_vol": "float64",
        "horizon": "int32"
    }
}

# B9: Leaderboard
SCHEMA_LEADERBOARD = {
    "columns": {
        "date": "datetime64[ns]",
        "symbol": "string",
        "score": "float64",
        "score_calibrated_ev": "float64",
        "rank": "int32",
        "sector": "string",
        "liquidity_adv20": "float64",
        "prior_version": "string",
        "model_version": "string",
        "cal_version": "string",
        "feature_version": "string"
    },
    "nullable": ["sector"]
}
