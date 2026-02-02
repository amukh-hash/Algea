"""
data_utils.py - Data preprocessing utilities for leak-proof ML

Contains:
1. purged_chronological_split: Time-series splits with purge gaps
2. robust_scaler_fit: Fit scaler on train only
3. clip_outliers: Clip instead of remove extreme values
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler


def purged_chronological_split(df, n_splits=3, purge_window=60):
    """
    Creates time-series splits with 'purge' gap to prevent look-ahead leakage.
    
    Args:
        df: DataFrame or array-like
        n_splits: Number of CV folds
        purge_window: Gap between train and val (should be >= max feature window)
    
    Yields:
        (train_indices, val_indices) tuples
    
    Example:
        for train_idx, val_idx in purged_chronological_split(df, purge_window=60):
            X_train, X_val = X[train_idx], X[val_idx]
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_index, val_index in tscv.split(df):
        if len(val_index) > purge_window:
            val_index_purged = val_index[purge_window:]
        else:
            raise ValueError(
                f"Validation set ({len(val_index)}) smaller than purge window ({purge_window})!"
            )
        yield train_index, val_index_purged


def single_purged_split(n_samples, train_ratio=0.8, val_ratio=0.1, purge_window=60):
    """
    Single train/val/test split with purge gaps (not CV).
    
    Returns:
        (train_idx, val_idx, test_idx) arrays
    """
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end + purge_window, val_end)
    test_idx = np.arange(val_end + purge_window, n_samples)
    
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Not enough data after purge gaps. Reduce purge_window or add more data.")
    
    return train_idx, val_idx, test_idx


def clip_outliers(series, n_sigma=4):
    """
    Clip outliers to ±n_sigma instead of removing them.
    Preserves signal magnitude while preventing gradient explosion.
    
    Args:
        series: pd.Series or np.array
        n_sigma: Clip threshold (default 4)
    
    Returns:
        Clipped series
    """
    if isinstance(series, pd.Series):
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        lower = median - n_sigma * iqr
        upper = median + n_sigma * iqr
        return series.clip(lower=lower, upper=upper)
    else:
        # numpy
        median = np.median(series)
        q75, q25 = np.percentile(series, [75, 25])
        iqr = q75 - q25
        lower = median - n_sigma * iqr
        upper = median + n_sigma * iqr
        return np.clip(series, lower, upper)


def fit_robust_scaler(X_train):
    """
    Fit RobustScaler on training data only.
    Returns fitted scaler for later use.
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def buy_hold_baseline(prices):
    """
    Calculate Buy & Hold return as baseline.
    Model must beat this to have positive alpha.
    """
    return (prices[-1] - prices[0]) / prices[0]


def cv_stacking_generate(X, y, model_class, model_kwargs, n_folds=5):
    """
    Cross-Validation Stacking to generate out-of-sample predictions.
    
    Used for Judge training: instead of training Judge on small Val set,
    we generate predictions for the entire Training set via K-Fold.
    
    Args:
        X: Feature array (numpy or tensor)
        y: Labels array
        model_class: Model class to instantiate (e.g., StudentModel)
        model_kwargs: Dict of kwargs for model instantiation
        n_folds: Number of CV folds
    
    Returns:
        oos_predictions: Array of out-of-sample predictions (same length as X)
        oos_confidences: Array of confidence scores
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    n_samples = len(X)
    oos_predictions = np.zeros(n_samples)
    oos_confidences = np.zeros(n_samples)
    
    kf = KFold(n_splits=n_folds, shuffle=False)  # No shuffle for time-series
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"CV-Stacking Fold {fold_idx + 1}/{n_folds}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        
        # Train temporary model on this fold
        model = model_class(**model_kwargs)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on held-out fold
        preds = model.predict(X_val_fold)
        
        # Get confidence (probability of predicted class)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_val_fold)
            confidences = probs.max(axis=1)
        else:
            confidences = np.ones(len(val_idx))
        
        oos_predictions[val_idx] = preds
        oos_confidences[val_idx] = confidences
    
    return oos_predictions, oos_confidences
