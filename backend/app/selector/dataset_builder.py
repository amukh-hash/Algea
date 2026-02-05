
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
import torch
from backend.app.ops import pathmap, artifact_registry, config
from backend.app.features import schemas, validators
from backend.app.data import calendar, ingest_daily

logger = logging.getLogger(__name__)

def build_labels_fwd(start_date, end_date, horizon_td: int = None) -> pd.DataFrame:
    """
    Builds forward returns, direction, and realized vol labels.
    Shift = horizon trading days.
    """
    if horizon_td is None:
        horizon_td = config.LABEL_HORIZON_TD

    trading_days = calendar.get_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError("No trading days found for label build.")

    labels = []
    paths = pathmap.get_paths()
    ohlcv_root = os.path.join(paths.data_canonical, "ohlcv_adj")
    if not os.path.exists(ohlcv_root):
        raise FileNotFoundError(f"Missing OHLCV root: {ohlcv_root}")
    symbols = [p.split("ticker=")[-1] for p in os.listdir(ohlcv_root) if p.startswith("ticker=")]

    for symbol in symbols:
        df = ingest_daily.load_ohlcv(symbol, start_date=start_date, end_date=end_date)
        if df.empty:
            continue
        df = df.sort_values("date")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").reindex(pd.DatetimeIndex(trading_days))
        if df["close_adj"].isnull().all():
            continue

        ret_series = df["close_adj"].pct_change()
        for current in trading_days:
            try:
                fwd_date = calendar.shift_trading_days(pd.Timestamp(current), horizon_td)
            except Exception:
                continue

            if current not in df.index or fwd_date not in df.index:
                continue

            close_now = df.loc[current, "close_adj"]
            close_fwd = df.loc[fwd_date, "close_adj"]
            if pd.isna(close_now) or pd.isna(close_fwd):
                continue

            fwd_ret = float(close_fwd / close_now - 1.0)
            window_rets = ret_series.loc[current:fwd_date].dropna()
            if not window_rets.empty and window_rets.index[0] == pd.Timestamp(current):
                window_rets = window_rets.iloc[1:]
            fwd_vol = float(window_rets.std(ddof=0)) if not window_rets.empty else 0.0
            labels.append({
                "date": pd.to_datetime(current),
                "symbol": symbol,
                "fwd_ret": fwd_ret,
                "fwd_up": int(fwd_ret > 0),
                "fwd_vol": fwd_vol,
                "horizon": int(horizon_td)
            })

    labels_df = pd.DataFrame(labels)
    validators.validate_df(labels_df, schemas.SCHEMA_LABELS, context="Labels Build", strict=True)
    return labels_df

def build_rank_dataset(
    start_date,
    end_date,
    sequence_len: int = 60,
    min_group_size: int = 200,
    horizon_td: int = None
) -> Dict:
    """
    Builds the tensors for the selector model.
    X: [Batch, Seq, Features]
    y: [Batch] (Listwise?)
    """
    feature_path = pathmap.resolve("featureframe", version="v1")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"FeatureFrame missing: {feature_path}")

    features = pd.read_parquet(feature_path)
    features["date"] = pd.to_datetime(features["date"])

    labels = build_labels_fwd(start_date, end_date, horizon_td=horizon_td)
    labels["date"] = pd.to_datetime(labels["date"])

    trading_days = calendar.get_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError("No trading days found for dataset build.")

    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    dates_list: List[str] = []
    symbols_list: List[List[str]] = []
    groups: List[Dict] = []

    paths = pathmap.get_paths()

    def resolve_latest_priors_path(date: pd.Timestamp) -> str:
        dstr = str(date).split(" ")[0]
        priors_dir = os.path.join(paths.priors, f"date={dstr}")
        if not os.path.exists(priors_dir):
            raise FileNotFoundError(f"Priors directory missing: {priors_dir}")
        candidates = [p for p in os.listdir(priors_dir) if p.startswith("priors_v") and p.endswith(".parquet")]
        if not candidates:
            raise FileNotFoundError(f"No priors files found in {priors_dir}")
        candidates.sort()
        return os.path.join(priors_dir, candidates[-1])

    feature_cols = [
        "ret_1d", "ret_3d", "ret_5d", "ret_10d",
        "vol_20d", "vol_chg_1d",
        "dollar_vol_20d", "volume_z_20d",
        "spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d",
        "vix_level", "rate_proxy", "market_breadth_ad"
    ]
    prior_cols = ["drift", "vol_forecast", "tail_risk", "trend_conf"]

    for current in trading_days:
        current = pd.Timestamp(current)
        day_features = features[features["date"] <= current].copy()
        if day_features.empty:
            continue

        grouped = day_features.groupby("symbol")
        seq_frames = []
        seq_symbols = []
        for symbol, g in grouped:
            g = g.sort_values("date").tail(sequence_len)
            if len(g) < sequence_len:
                continue
            seq_frames.append(g)
            seq_symbols.append(symbol)

        if len(seq_frames) < min_group_size:
            continue

        X = np.stack([f[feature_cols].values.astype(np.float32) for f in seq_frames])

        priors_path = resolve_latest_priors_path(current)
        priors = pd.read_parquet(priors_path)
        priors_map = priors.set_index("symbol")
        priors_vec = []
        for symbol in seq_symbols:
            if symbol in priors_map.index:
                priors_vec.append(priors_map.loc[symbol, prior_cols].values.astype(np.float32))
            else:
                priors_vec.append(np.zeros(len(prior_cols), dtype=np.float32))
        priors_vec = np.stack(priors_vec)
        priors_seq = np.repeat(priors_vec[:, None, :], sequence_len, axis=1)
        X = np.concatenate([X, priors_seq], axis=-1)

        day_labels = labels[labels["date"] == current]
        label_map = day_labels.set_index("symbol")
        y = []
        keep_symbols = []
        keep_X = []
        for idx, symbol in enumerate(seq_symbols):
            if symbol not in label_map.index:
                continue
            y.append(float(label_map.loc[symbol, "fwd_ret"]))
            keep_symbols.append(symbol)
            keep_X.append(X[idx])

        if len(keep_symbols) < min_group_size:
            continue

        X_list.append(torch.tensor(np.stack(keep_X), dtype=torch.float32))
        y_list.append(torch.tensor(np.array(y), dtype=torch.float32))
        date_str = current.strftime("%Y-%m-%d")
        dates_list.append(date_str)
        symbols_list.append(keep_symbols)
        groups.append({
            "date": date_str,
            "size": len(keep_symbols),
            "symbols": keep_symbols
        })

    dataset = {
        "dates": dates_list,
        "X": X_list,
        "y": y_list,
        "symbols": symbols_list,
        "sequence_len": sequence_len,
        "label_horizon": int(horizon_td) if horizon_td is not None else int(config.LABEL_HORIZON_TD),
        "groups": groups,
        "feature_cols": feature_cols,
        "prior_cols": prior_cols
    }
    return dataset
