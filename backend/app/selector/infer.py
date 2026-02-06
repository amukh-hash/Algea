
import os
import json
import pandas as pd
import logging
from typing import Dict, List, Optional
import torch
import numpy as np
from backend.app.ops import pathmap, artifact_registry, config
from backend.app.features import schemas, validators
from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.selector_scaler import SelectorFeatureScaler
from backend.app.models.calibration import ScoreCalibrator

logger = logging.getLogger(__name__)

class SelectorInference:
    def __init__(self, model_version: str = "latest"):
        self.model_version = model_version
        self.model_path = pathmap.resolve("model_selector", version=model_version)
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.feature_cols = None
        self.prior_cols = None
        self.sequence_len = None

    def _load_model_bundle(self) -> None:
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model: {self.model_path}")
        meta_path = self.model_path + ".meta"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing model metadata: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        input_dim = meta.get("input_dim")
        if input_dim is None:
            raise ValueError("Model metadata missing input_dim.")

        self.model = RankTransformer(d_input=input_dim, d_model=64, n_head=2, n_layers=2)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()
        self.feature_cols = meta.get("feature_cols")
        self.prior_cols = meta.get("prior_cols")
        self.sequence_len = meta.get("sequence_len", config.SEQUENCE_LEN)

        paths = pathmap.get_paths()
        scaler_path = os.path.join(paths.calibration, "selector_scaler_v1.joblib")
        calib_path = os.path.join(paths.calibration, "selector_calibration_v1.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Missing calibrator: {calib_path}")
        self.scaler = SelectorFeatureScaler.load(scaler_path)
        self.calibrator = ScoreCalibrator.load(calib_path)

    def _build_sequences(
        self,
        asof_date: pd.Timestamp,
        symbols: List[str],
        features_df: pd.DataFrame,
        priors_df: pd.DataFrame,
        sequence_len: int,
    ) -> Dict[str, torch.Tensor]:
        feature_cols = self.feature_cols or [
            "ret_1d", "ret_3d", "ret_5d", "ret_10d",
            "vol_20d", "vol_chg_1d",
            "dollar_vol_20d", "volume_z_20d",
            "spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d",
            "vix_level", "rate_proxy", "market_breadth_ad"
        ]
        prior_cols = self.prior_cols or ["drift", "vol_forecast", "tail_risk", "trend_conf"]
        priors_map = priors_df.set_index("symbol")

        tokens = []
        keep_symbols = []
        for symbol in symbols:
            sym_df = features_df[(features_df["symbol"] == symbol) & (features_df["date"] == asof_date)]
            if sym_df.empty:
                continue
            feat = sym_df[feature_cols].values.astype("float32")[0]

            if symbol in priors_map.index:
                prior_vec = priors_map.loc[symbol, prior_cols].values.astype("float32")
            else:
                prior_vec = np.zeros(len(prior_cols), dtype="float32")

            token = np.concatenate([feat, prior_vec], axis=0)
            tokens.append(token)
            keep_symbols.append(symbol)

        if not tokens:
            raise ValueError("No valid tokens for inference.")

        X = torch.tensor(np.stack(tokens), dtype=torch.float32)
        return {"X": X, "symbols": keep_symbols}

    def predict(self, date: str, symbols: List[str], features_df: pd.DataFrame, priors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference for the given universe.
        Returns Leaderboard DataFrame (Schema B9).
        """
        self._load_model_bundle()

        asof_date = pd.to_datetime(date)
        features_df = features_df.copy()
        priors_df = priors_df.copy()
        features_df["date"] = pd.to_datetime(features_df["date"])
        priors_df["date"] = pd.to_datetime(priors_df["date"])

        seq = self._build_sequences(
            asof_date=asof_date,
            symbols=symbols,
            features_df=features_df,
            priors_df=priors_df,
            sequence_len=self.sequence_len,
        )

        X = seq["X"]
        X_scaled = self.scaler.transform(X).unsqueeze(0)
        with torch.no_grad():
            scores_tensor = self.model(X_scaled)["score"].squeeze(0).squeeze(-1)
        scores = scores_tensor.numpy().tolist()
        calibrated_scores = self.calibrator.predict(scores)
        ranks = pd.Series(scores).rank(ascending=False, method="first").astype(int).tolist()

        meta = {
            "prior_version": priors_df["prior_version"].iloc[0] if "prior_version" in priors_df.columns else "unknown",
            "model_version": self.model_version,
            "cal_version": self.calibrator.version,
            "feature_version": features_df["feature_version"].iloc[0] if "feature_version" in features_df.columns else "unknown"
        }

        symbols_used = seq["symbols"]
        adv20 = []
        for symbol in symbols_used:
            df = features_df[features_df["symbol"] == symbol].sort_values("date").tail(20)
            adv20.append(float(df["dollar_vol_20d"].iloc[-1]) if not df.empty else 0.0)

        sector = ["UNKNOWN"] * len(symbols_used)

        # 4. Construct Leaderboard
        df = pd.DataFrame({
            "date": asof_date,
            "symbol": symbols_used,
            "score": scores,
            "score_calibrated_ev": list(calibrated_scores),
            "rank": ranks,
            "sector": sector,
            "liquidity_adv20": adv20,
            "prior_version": meta["prior_version"],
            "model_version": meta["model_version"],
            "cal_version": meta["cal_version"],
            "feature_version": meta["feature_version"]
        })
        
        # Validate Schema B9
        validators.validate_df(df, schemas.SCHEMA_LEADERBOARD, context="Inference Leaderboard")
        
        return df

def write_leaderboard(asof_date, df: pd.DataFrame) -> str:
    path = pathmap.resolve("leaderboard", date=asof_date)
    
    # Ensure dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    
    return path
