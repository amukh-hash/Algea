from typing import List

# Ordered list of features used by the Selector model
# Must match exactly between training and inference.
# Order matters.

SELECTOR_FEATURES_V1 = [
    # Price / Trend
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "volatility_20d",

    # Volume / Liquidity
    "volume_log_change_5d",

    # Market Breadth (Context)
    "ad_line_trend_5d",
    "bpi_level",

    # Teacher Priors (Attached)
    "teacher_drift_20d",
    "teacher_vol_20d",
    "teacher_downside_q10_20d",
    "teacher_trend_conf_20d"
]

def get_feature_list(version: str = "v1") -> List[str]:
    if version == "v1":
        return SELECTOR_FEATURES_V1
    else:
        raise ValueError(f"Unknown feature contract version: {version}")
