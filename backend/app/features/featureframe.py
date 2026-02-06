import os
import pandas as pd
import logging
from typing import Dict, List
from backend.app.ops import pathmap, artifact_registry
from backend.app.features import schemas, validators
from backend.app.data import calendar, ingest_daily, security_master

logger = logging.getLogger(__name__)


def _compute_data_version(source_versions: Dict[str, List[str]]) -> str:
    normalized = {k: sorted(set(v)) for k, v in source_versions.items() if v}
    if not normalized:
        return "unknown"
    return artifact_registry.stable_hash(normalized)


def build_featureframe(
    start_date, end_date, feature_spec: Dict, code_version: str = "v1"
) -> pd.DataFrame:
    """
    Computes features for the given window.
    Returns Schema B6.
    """
    trading_days = calendar.get_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError("No trading days found in requested range.")

    symbols = feature_spec.get("symbols") if feature_spec else None
    if not symbols:
        try:
            master = security_master.load_security_master()
            symbols = master["symbol"].dropna().astype(str).unique().tolist()
        except Exception:
            paths = pathmap.get_paths()
            ohlcv_root = os.path.join(paths.data_canonical, "ohlcv_adj")
            daily_root = os.path.join(paths.data_canonical, "daily_parquet")
            if os.path.exists(ohlcv_root):
                symbols = [p.split("ticker=")[-1] for p in os.listdir(ohlcv_root) if p.startswith("ticker=")]
            elif os.path.exists(daily_root):
                symbols = []
                for fname in os.listdir(daily_root):
                    if fname.endswith(".parquet"):
                        symbols.append(fname.replace("_daily.parquet", "").replace(".parquet", ""))
                    elif fname.startswith("ticker="):
                        symbols.append(fname.split("ticker=")[-1])
            else:
                raise FileNotFoundError("No canonical OHLCV partitions found.")

    cov_path = pathmap.resolve("covariates")
    breadth_path = pathmap.resolve("breadth")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(f"Missing covariates file: {cov_path}")
    if not os.path.exists(breadth_path):
        raise FileNotFoundError(f"Missing breadth file: {breadth_path}")

    cov_df = pd.read_parquet(cov_path)
    breadth_df = pd.read_parquet(breadth_path)
    cov_df["date"] = pd.to_datetime(cov_df["date"])
    breadth_df["date"] = pd.to_datetime(breadth_df["date"])

    required_cov_cols = [
        "date",
        "spy_ret_1d",
        "qqq_ret_1d",
        "iwm_ret_1d",
        "vix_level",
        "rate_proxy"
    ]
    missing_cov = [c for c in required_cov_cols if c not in cov_df.columns]
    if missing_cov:
        raise ValueError(f"Covariates missing required columns: {missing_cov}")

    frames: List[pd.DataFrame] = []
    source_versions: Dict[str, List[str]] = {"ohlcv": [], "covariates": [], "breadth": []}
    if "data_version" in cov_df.columns:
        source_versions["covariates"].extend(cov_df["data_version"].dropna().astype(str).unique().tolist())
    if "data_version" in breadth_df.columns:
        source_versions["breadth"].extend(breadth_df["data_version"].dropna().astype(str).unique().tolist())

    for symbol in symbols:
        ohlcv = ingest_daily.load_ohlcv(symbol, start_date=start_date, end_date=end_date)
        if ohlcv.empty:
            continue

        ohlcv = ohlcv.copy()
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv = ohlcv.sort_values("date")
        ohlcv = ohlcv.set_index("date").reindex(pd.DatetimeIndex(trading_days))

        if "data_version" in ohlcv.columns:
            source_versions["ohlcv"].extend(
                ohlcv["data_version"].dropna().astype(str).unique().tolist()
            )

        if "close_adj" not in ohlcv.columns or ohlcv["close_adj"].isnull().all():
            if "ret_1d" not in ohlcv.columns:
                continue
        else:
            if "ret_1d" not in ohlcv.columns:
                ohlcv["ret_1d"] = ohlcv["close_adj"].pct_change()
            if "ret_3d" not in ohlcv.columns:
                ohlcv["ret_3d"] = ohlcv["close_adj"].pct_change(3)
            if "ret_5d" not in ohlcv.columns:
                ohlcv["ret_5d"] = ohlcv["close_adj"].pct_change(5)
            if "ret_10d" not in ohlcv.columns:
                ohlcv["ret_10d"] = ohlcv["close_adj"].pct_change(10)

        if "ret_3d" not in ohlcv.columns or "ret_5d" not in ohlcv.columns or "ret_10d" not in ohlcv.columns:
            continue
        ohlcv["vol_20d"] = ohlcv["ret_1d"].rolling(20).std()
        ohlcv["vol_chg_1d"] = ohlcv["vol_20d"].pct_change()
        if "dollar_volume" not in ohlcv.columns:
            if "close_adj" in ohlcv.columns and "volume" in ohlcv.columns:
                ohlcv["dollar_volume"] = ohlcv["close_adj"] * ohlcv["volume"]
            elif "volume" in ohlcv.columns:
                ohlcv["dollar_volume"] = ohlcv["volume"]
        ohlcv["dollar_vol_20d"] = ohlcv["dollar_volume"].rolling(20).mean()
        vol_mean = ohlcv["volume"].rolling(20).mean()
        vol_std = ohlcv["volume"].rolling(20).std()
        ohlcv["volume_z_20d"] = (ohlcv["volume"] - vol_mean) / vol_std

        feat = ohlcv.reset_index().rename(columns={"index": "date"})
        feat["symbol"] = symbol

        feat = feat[[
            "date", "symbol",
            "ret_1d", "ret_3d", "ret_5d", "ret_10d",
            "vol_20d", "vol_chg_1d",
            "dollar_vol_20d", "volume_z_20d"
        ]]

        feat = feat.merge(cov_df[required_cov_cols], on="date", how="left")
        if "market_breadth_ad" not in breadth_df.columns:
            raise ValueError("Breadth missing required column: market_breadth_ad")
        feat = feat.merge(breadth_df[["date", "market_breadth_ad"]], on="date", how="left")

        frames.append(feat)

    if not frames:
        raise ValueError("FeatureFrame build produced no data.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[
        "ret_1d", "ret_3d", "ret_5d", "ret_10d",
        "vol_20d", "dollar_vol_20d",
        "volume_z_20d", "vol_chg_1d",
        "spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d",
        "vix_level", "rate_proxy", "market_breadth_ad"
    ])

    data_version = feature_spec.get("data_version") if feature_spec else None
    if not data_version:
        data_version = _compute_data_version(source_versions)
    feature_version = artifact_registry.compute_feature_version(feature_spec or {}, code_version)
    df["feature_version"] = feature_version
    df["data_version"] = data_version

    validators.validate_df(df, schemas.SCHEMA_FEATUREFRAME, context="FeatureFrame Build", strict=True)
    return df


def write_featureframe(df: pd.DataFrame, feature_spec: Dict, version_tag: str = "v1") -> str:
    code_version = "v1"
    fv = artifact_registry.compute_feature_version(feature_spec, code_version)
    if "feature_version" not in df.columns:
        df["feature_version"] = fv
    if "data_version" not in df.columns:
        df["data_version"] = feature_spec.get("data_version", "unknown")

    path = pathmap.resolve("featureframe", version=version_tag)

    validators.validate_df(df, schemas.SCHEMA_FEATUREFRAME, context="FeatureFrame Write")

    df.to_parquet(path)

    meta = {
        "version": fv,
        "spec": feature_spec,
        "rows": len(df),
        "columns": list(df.columns)
    }
    artifact_registry.write_metadata(path + ".meta", meta)

    return path
