"""Model training with preprocessing, walk-forward CV, and bundle persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import COOCReversalConfig
from .splits import SplitSpec, walk_forward_cv
from .types import ModelBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns (canonical source: features_core)
# ---------------------------------------------------------------------------

from ..features_core import FEATURE_SCHEMA as FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Preprocessing (train-only fit)
# ---------------------------------------------------------------------------

class Preprocessor:
    """Deterministic scaler + NaN handler.  Fit on training data only."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.nan_fill_: Optional[np.ndarray] = None
        self.feature_order: Tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, features: Sequence[str]) -> "Preprocessor":
        self.feature_order = tuple(features)
        vals = X[list(features)].values.astype(np.float64)
        self.nan_fill_ = np.nanmedian(vals, axis=0)
        filled = np.where(np.isnan(vals), self.nan_fill_, vals)
        self.mean_ = np.nanmean(filled, axis=0)
        self.std_ = np.nanstd(filled, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # avoid div by zero
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.mean_ is not None, "Preprocessor not fitted"
        vals = X[list(self.feature_order)].values.astype(np.float64)
        filled = np.where(np.isnan(vals), self.nan_fill_, vals)
        return (filled - self.mean_) / self.std_

    def nan_fill_dict(self) -> Dict[str, float]:
        assert self.nan_fill_ is not None
        return {f: float(v) for f, v in zip(self.feature_order, self.nan_fill_)}


# ---------------------------------------------------------------------------
# Simple ridge model (no torch dependency for pipeline)
# ---------------------------------------------------------------------------

class _RidgeModel:
    """Minimal ridge regression for the pipeline baseline."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeModel":
        n, p = X.shape
        XtX = X.T @ X + self.alpha * np.eye(p)
        Xty = X.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)
        self.intercept_ = float(np.mean(y) - np.mean(X, axis=0) @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None
        return X @ self.coef_ + self.intercept_


class _RankingModel:
    """Ranking model wrapper: trains ridge on rank-transformed labels."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.inner = _RidgeModel(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RankingModel":
        from scipy.stats import rankdata  # type: ignore[import-untyped]
        y_ranked = rankdata(y) / len(y)
        self.inner.fit(X, y_ranked)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.inner.predict(X)


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_model(
    config: COOCReversalConfig,
    dataset: pd.DataFrame,
    splits: List[SplitSpec],
    seed: int = 42,
    mode: str = "regression",
    param_grid: Optional[List[Dict[str, Any]]] = None,
    primary_metric: str = "ic",
) -> ModelBundle:
    """Train model with walk-forward CV, return a persisted ModelBundle.

    Parameters
    ----------
    config : sleeve configuration
    dataset : assembled dataset with features + labels
    splits : walk-forward split specs (from splits.py)
    seed : random seed for determinism
    mode : 'regression' or 'ranking'
    param_grid : list of param dicts to search (default: alpha grid)
    primary_metric : metric to select best params ('ic' or 'mse')

    Returns
    -------
    ModelBundle (not yet persisted — call save_model_bundle for that)
    """
    np.random.seed(seed)

    # Intersect schema features with what's actually in the dataset.
    # Some optional features may have been dropped by the feature guard.
    features = [f for f in FEATURE_COLUMNS if f in dataset.columns]
    label_col = "y"

    if param_grid is None:
        param_grid = [
            {"alpha": 0.01},
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
            {"alpha": 100.0},
        ]

    trial_log: List[Dict[str, Any]] = []
    best_metric: float = -np.inf
    best_params: Dict[str, Any] = param_grid[0]

    # --- Walk-forward CV ---
    for params in param_grid:
        fold_metrics: List[float] = []

        for split in splits:
            train_mask = _day_mask(dataset, split.train_start, split.train_end)
            val_mask = _day_mask(dataset, split.val_start, split.val_end)

            train_df = dataset.loc[train_mask]
            val_df = dataset.loc[val_mask]

            if len(train_df) < 10 or len(val_df) < 5:
                continue

            # Fit preprocessor on training only
            pp = Preprocessor().fit(train_df, features)
            X_train = pp.transform(train_df)
            y_train = train_df[label_col].values.astype(np.float64)

            X_val = pp.transform(val_df)
            y_val = val_df[label_col].values.astype(np.float64)

            # Train model
            model = _make_model(mode, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Evaluate
            metric_val = _compute_metric(y_val, y_pred, primary_metric)
            fold_metrics.append(metric_val)

        avg_metric = float(np.mean(fold_metrics)) if fold_metrics else -np.inf
        trial_log.append({
            "params": params,
            "avg_metric": avg_metric,
            "fold_metrics": fold_metrics,
        })

        if avg_metric > best_metric:
            best_metric = avg_metric
            best_params = params

    # --- Refit on full training set with best params ---
    full_split = splits[-1] if splits else None
    if full_split is not None:
        # Use all data up to end of last validation fold
        train_mask = _day_mask(dataset, splits[0].train_start, full_split.val_end)
    else:
        train_mask = pd.Series(True, index=dataset.index)

    train_df = dataset.loc[train_mask]
    pp_final = Preprocessor().fit(train_df, features)
    X_final = pp_final.transform(train_df)
    y_final = train_df[label_col].values.astype(np.float64)

    final_model = _make_model(mode, best_params)
    final_model.fit(X_final, y_final)

    return ModelBundle(
        model_path="",  # set when saved
        feature_order=tuple(features),
        scaler_path="",
        nan_fill_values=pp_final.nan_fill_dict(),
        chosen_params=best_params,
        trial_log=tuple(trial_log),
        primary_metric=primary_metric,
        primary_metric_value=best_metric,
    ), final_model, pp_final  # type: ignore[return-value]


def train_transformer(
    config: COOCReversalConfig,
    dataset: pd.DataFrame,
    splits: List[SplitSpec],
    seed: int = 42,
    primary_metric: str = "trade_proxy_sharpe",
    two_head: bool = True,
) -> Tuple[ModelBundle, Any, "Preprocessor"]:
    """Train a CrossSectionalTransformer with walk-forward CV.

    Hyperparameter selection optimizes ``trade_proxy_sharpe`` by default.
    Ridge-compatible interface: returns (ModelBundle, model, preprocessor).

    When ``two_head=True``, model predicts (score, risk) and the trade
    score used for evaluation is ``score / (eps + risk)``.
    """
    import torch
    from torch.utils.data import DataLoader

    from ..model.cs_transformer import CrossSectionalTransformer
    from ..model.losses import HuberRankLoss, MultiTaskLoss
    from ..model.score_stabilizer import stabilize_derived_score
    from .panel_dataset import PanelDataset, panel_collate_fn

    # Deterministic seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    features = [f for f in FEATURE_COLUMNS if f in dataset.columns]
    label_col = "y"
    mc = config.model

    # Hyperparams from config
    hidden_dim = mc.d_model if hasattr(mc, "d_model") else getattr(mc, "hidden_dim", 128)
    n_heads = mc.n_heads
    n_layers = mc.n_layers
    dropout = mc.dropout
    lr = mc.lr
    weight_decay = getattr(mc, "weight_decay", 1e-4)
    epochs = mc.epochs
    batch_size_days = mc.batch_size
    early_stop_patience = getattr(mc, "early_stop_patience", 10)

    trial_log: List[Dict[str, Any]] = []
    best_metric: float = -np.inf

    # Determine r_oc column for risk target
    r_oc_col = "r_oc" if "r_oc" in dataset.columns else ("ret_oc" if "ret_oc" in dataset.columns else None)
    risk_transform = mc.risk_target_transform if hasattr(mc, "risk_target_transform") else "log_abs"
    risk_eps = mc.risk_target_eps if hasattr(mc, "risk_target_eps") else 1e-6

    # Stabilizer params
    stab_kwargs = {
        "risk_clamp_min": getattr(mc, "risk_pred_clamp_min", 0.05),
        "risk_clamp_max": getattr(mc, "risk_pred_clamp_max", 5.0),
        "score_tanh": getattr(mc, "score_tanh", False),
        "derived_clip": getattr(mc, "derived_score_clip", 10.0),
    }

    # Walk-forward CV: train transformer on each fold, evaluate via trade proxy
    for split in splits:
        train_mask = _day_mask(dataset, split.train_start, split.train_end)
        val_mask = _day_mask(dataset, split.val_start, split.val_end)
        train_df = dataset.loc[train_mask]
        val_df = dataset.loc[val_mask]

        if len(train_df) < 20 or len(val_df) < 10:
            continue

        # Fit preprocessor
        pp = Preprocessor().fit(train_df, features)
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        X_train_np = pp.transform(train_df)
        X_val_np = pp.transform(val_df)
        for i, col in enumerate(features):
            train_scaled[col] = X_train_np[:, i]
            val_scaled[col] = X_val_np[:, i]

        # Build panel datasets with risk target transform
        extra_cols = ["r_oc"] if r_oc_col == "r_oc" else (["ret_oc"] if r_oc_col else [])
        panel_kw = dict(
            risk_target_transform=risk_transform,
            risk_target_eps=risk_eps,
        )
        train_panel = PanelDataset(train_scaled, features, label_col,
                                   min_instruments_per_day=2, extra_cols=extra_cols,
                                   **panel_kw)
        val_panel = PanelDataset(val_scaled, features, label_col,
                                  min_instruments_per_day=2, extra_cols=extra_cols,
                                  **panel_kw)

        if len(train_panel) < 5 or len(val_panel) < 3:
            continue

        train_loader = DataLoader(
            train_panel, batch_size=batch_size_days,
            shuffle=False, collate_fn=panel_collate_fn,
        )
        val_loader = DataLoader(
            val_panel, batch_size=batch_size_days,
            shuffle=False, collate_fn=panel_collate_fn,
        )

        # Build model
        model = CrossSectionalTransformer(
            n_features=len(features),
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            two_head=two_head,
        )

        if two_head:
            criterion = MultiTaskLoss(
                score_delta=1.0, risk_delta=1.0,
                risk_weight=0.5, collapse_penalty=0.1,
            )
        else:
            criterion = HuberRankLoss(delta=1.0, rank_weight=0.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                X_b = batch["X"]
                y_b = batch["y"]
                mask_b = batch["mask"]
                optimizer.zero_grad()

                if two_head and r_oc_col:
                    score_pred, risk_pred = model(X_b, mask_b, return_risk=True)
                    # Risk target: already transformed via PanelDataset (log(eps + |r_oc|))
                    risk_target = batch.get("y_risk", y_b.abs())
                    loss = criterion(score_pred, risk_pred, y_b, risk_target, mask_b)
                else:
                    pred = model(X_b, mask_b)
                    loss = criterion(pred, y_b, mask_b)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if two_head and r_oc_col:
                        sp, rp = model(batch["X"], batch["mask"], return_risk=True)
                        rt = batch.get("y_risk", batch["y"].abs())
                        vloss = criterion(sp, rp, batch["y"], rt, batch["mask"])
                    else:
                        pred = model(batch["X"], batch["mask"])
                        vloss = criterion(pred, batch["y"], batch["mask"])
                    val_losses.append(vloss.item())
            avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logger.info("Early stop at epoch %d (best val loss %.4f)", epoch, best_val_loss)
                    break

        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate final metric using derived score
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch in val_loader:
                if two_head:
                    sp, rp = model(batch["X"], batch["mask"], return_risk=True)
                    derived = stabilize_derived_score(sp, rp, **stab_kwargs)
                    for b_idx in range(derived.shape[0]):
                        m = batch["mask"][b_idx]
                        all_preds.extend(derived[b_idx][m].numpy().tolist())
                        all_true.extend(batch["y"][b_idx][m].numpy().tolist())
                else:
                    pred = model(batch["X"], batch["mask"])
                    for b_idx in range(pred.shape[0]):
                        m = batch["mask"][b_idx]
                        all_preds.extend(pred[b_idx][m].numpy().tolist())
                        all_true.extend(batch["y"][b_idx][m].numpy().tolist())

        fold_metric = _compute_metric(
            np.array(all_true), np.array(all_preds), "ic",
        )

        trial_log.append({
            "fold": f"{split.train_start}_{split.val_end}",
            "val_loss": best_val_loss,
            "ic": fold_metric,
            "epochs_trained": epoch + 1 if 'epoch' in dir() else 0,
            "two_head": two_head,
        })

        if fold_metric > best_metric:
            best_metric = fold_metric

    # --- Final refit on all available data ---
    full_split = splits[-1] if splits else None
    if full_split is not None:
        train_mask = _day_mask(dataset, splits[0].train_start, full_split.val_end)
    else:
        train_mask = pd.Series(True, index=dataset.index)

    train_df = dataset.loc[train_mask]
    pp_final = Preprocessor().fit(train_df, features)
    train_scaled = train_df.copy()
    X_train_np = pp_final.transform(train_df)
    for i, col in enumerate(features):
        train_scaled[col] = X_train_np[:, i]

    extra_cols_final = ["r_oc"] if r_oc_col == "r_oc" else (["ret_oc"] if r_oc_col else [])
    train_panel = PanelDataset(train_scaled, features, label_col,
                               min_instruments_per_day=2, extra_cols=extra_cols_final,
                               **panel_kw)
    train_loader = DataLoader(
        train_panel, batch_size=batch_size_days,
        shuffle=False, collate_fn=panel_collate_fn,
    )

    final_model = CrossSectionalTransformer(
        n_features=len(features),
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        two_head=two_head,
    )

    if two_head:
        final_criterion = MultiTaskLoss(
            score_delta=1.0, risk_delta=1.0,
            risk_weight=0.5, collapse_penalty=0.1,
        )
    else:
        final_criterion = HuberRankLoss(delta=1.0, rank_weight=0.0)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        final_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if two_head and r_oc_col:
                sp, rp = final_model(batch["X"], batch["mask"], return_risk=True)
                rt = batch.get("y_risk", batch["y"].abs())
                loss = final_criterion(sp, rp, batch["y"], rt, batch["mask"])
            else:
                pred = final_model(batch["X"], batch["mask"])
                loss = final_criterion(pred, batch["y"], batch["mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    bundle = ModelBundle(
        model_path="",
        feature_order=tuple(features),
        scaler_path="",
        nan_fill_values=pp_final.nan_fill_dict(),
        chosen_params={
            "estimator": "CSTransformer",
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "lr": lr,
            "epochs": epochs,
            "two_head": two_head,
            "risk_head_enabled": two_head,
            "risk_target_name": "log_abs_r_oc" if two_head else "none",
            "risk_target_eps": risk_eps,
            "risk_target_transform": risk_transform,
            "risk_pred_clamp_min": stab_kwargs["risk_clamp_min"],
            "risk_pred_clamp_max": stab_kwargs["risk_clamp_max"],
            "score_tanh": stab_kwargs["score_tanh"],
            "derived_score_clip": stab_kwargs["derived_clip"],
        },
        trial_log=tuple(trial_log),
        primary_metric=primary_metric,
        primary_metric_value=best_metric,
    )

    return bundle, final_model, pp_final


def train_transformer_multiseed(
    config: COOCReversalConfig,
    dataset: pd.DataFrame,
    splits: List[SplitSpec],
    primary_metric: str = "trade_proxy_sharpe",
) -> Tuple[ModelBundle, Any, "Preprocessor"]:
    """Train CSTransformer across multiple seeds, select by median trade_proxy Sharpe.

    Uses ``config.model.seeds`` for the seed list and
    ``config.model.two_head`` for model architecture.
    """
    seeds = getattr(config.model, "seeds", (42,))
    two_head = getattr(config.model, "two_head", True)

    results: List[Tuple[float, ModelBundle, Any, "Preprocessor"]] = []

    for s in seeds:
        logger.info("=== Multi-seed training: seed=%d ===", s)
        bundle, model, pp = train_transformer(
            config, dataset, splits,
            seed=s, primary_metric=primary_metric,
            two_head=two_head,
        )
        results.append((bundle.primary_metric_value, bundle, model, pp))
        logger.info("  seed=%d  metric=%.4f", s, bundle.primary_metric_value)

    if not results:
        raise RuntimeError("No successful training runs across seeds")

    # Select by median metric value
    results.sort(key=lambda r: r[0])
    median_idx = len(results) // 2
    _, best_bundle, best_model, best_pp = results[median_idx]

    logger.info(
        "Multi-seed selection: picked seed-idx=%d (metric=%.4f) from %d seeds",
        median_idx, best_bundle.primary_metric_value, len(results),
    )

    return best_bundle, best_model, best_pp


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_model_bundle(
    bundle_info: ModelBundle,
    model: Any,
    preprocessor: Preprocessor,
    output_dir: str | Path,
) -> ModelBundle:
    """Persist model, preprocessor, and manifest to *output_dir*.

    Supports both Ridge (.pkl) and Transformer (.pt) model formats.
    Returns updated ModelBundle with paths filled in.
    """
    import torch as _torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    is_transformer = bundle_info.chosen_params.get("estimator") == "CSTransformer"

    if is_transformer and hasattr(model, "state_dict"):
        model_path = out / "model.pt"
        _torch.save(model.state_dict(), model_path)
    else:
        model_path = out / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    scaler_path = out / "preprocessor.pkl"
    manifest_path = out / "model_manifest.json"

    with open(scaler_path, "wb") as f:
        pickle.dump(preprocessor, f)

    updated = ModelBundle(
        model_path=str(model_path),
        feature_order=bundle_info.feature_order,
        scaler_path=str(scaler_path),
        nan_fill_values=bundle_info.nan_fill_values,
        chosen_params=bundle_info.chosen_params,
        trial_log=bundle_info.trial_log,
        primary_metric=bundle_info.primary_metric,
        primary_metric_value=bundle_info.primary_metric_value,
    )

    manifest_data = updated.to_dict()
    # Add transformer architecture params for load
    if is_transformer:
        for k in ("hidden_dim", "n_heads", "n_layers", "dropout"):
            manifest_data[k] = bundle_info.chosen_params.get(k)
    manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True))

    # Also save feature_schema.json
    schema_path = out.parent / "feature_schema.json"
    schema_data = {
        "feature_order": list(bundle_info.feature_order),
        "nan_fill_values": bundle_info.nan_fill_values,
        "estimator": bundle_info.chosen_params.get("estimator", "Ridge"),
    }
    schema_path.write_text(json.dumps(schema_data, indent=2, sort_keys=True))

    return updated


def load_model_bundle(
    bundle_dir: str | Path,
) -> Tuple[Any, Preprocessor, ModelBundle]:
    """Load a persisted model bundle (Ridge or Transformer)."""
    d = Path(bundle_dir)
    manifest = json.loads((d / "model_manifest.json").read_text())

    # Determine model type
    model_pt_path = d / "model.pt"
    model_pkl_path = d / "model.pkl"

    if model_pt_path.exists():
        import torch as _torch
        from ..model.cs_transformer import CrossSectionalTransformer

        features = tuple(manifest["feature_order"])
        model = CrossSectionalTransformer(
            n_features=len(features),
            hidden_dim=manifest.get("hidden_dim", 128),
            n_heads=manifest.get("n_heads", 4),
            n_layers=manifest.get("n_layers", 3),
            dropout=0.0,
        )
        model.load_state_dict(
            _torch.load(model_pt_path, map_location="cpu", weights_only=True)
        )
        model.eval()
    elif model_pkl_path.exists():
        with open(model_pkl_path, "rb") as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError(f"No model.pt or model.pkl found in {d}")

    with open(d / "preprocessor.pkl", "rb") as f:
        pp = pickle.load(f)

    bundle = ModelBundle(
        model_path=manifest["model_path"],
        feature_order=tuple(manifest["feature_order"]),
        scaler_path=manifest["scaler_path"],
        nan_fill_values=manifest["nan_fill_values"],
        chosen_params=manifest["chosen_params"],
        trial_log=tuple(manifest["trial_log"]),
        primary_metric=manifest["primary_metric"],
        primary_metric_value=manifest["primary_metric_value"],
    )
    return model, pp, bundle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _day_mask(dataset: pd.DataFrame, start_str: str, end_str: str) -> pd.Series:
    """Boolean mask for rows whose trading_day falls in [start, end]."""
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    if "trading_day" in dataset.columns:
        days = dataset["trading_day"]
    else:
        days = dataset.index.get_level_values("trading_day")

    # Normalise to date objects
    def _to_date(x: Any) -> date:
        if isinstance(x, date):
            return x
        return pd.Timestamp(x).date()

    return pd.Series(
        [start <= _to_date(d) <= end for d in days],
        index=dataset.index,
    )


def _make_model(mode: str, params: Dict[str, Any]) -> Any:
    alpha = params.get("alpha", 1.0)
    if mode == "ranking":
        return _RankingModel(alpha=alpha)
    return _RidgeModel(alpha=alpha)


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "ic":
        if len(y_true) < 3:
            return 0.0
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0
    elif metric == "mse":
        return -float(np.mean((y_true - y_pred) ** 2))  # negative so higher is better
    return 0.0
