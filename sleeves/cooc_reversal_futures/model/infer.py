"""Inference helper for CS-Transformer at runtime.

Used by ``signal_mode.model_predictions()`` and the sleeve.
Supports both single-head and two-head (score + risk) models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def predict_panel(
    model_dir: str | Path,
    feature_frame: pd.DataFrame,
    features: Sequence[str],
    nan_fill: Dict[str, float],
    instruments: Sequence[str],
    inst_col: str = "instrument",
) -> Dict[str, float]:
    """Run CS-Transformer inference on a single-day panel.

    Parameters
    ----------
    model_dir : directory containing ``model.pt`` + ``model_manifest.json``
    feature_frame : DataFrame with feature columns
    features : ordered feature names
    nan_fill : NaN fill values per feature
    instruments : instrument symbols to predict
    inst_col : column name for instrument filtering

    Returns
    -------
    ``{instrument: score}`` dict.
    For two-head models, score = raw_score / (eps + risk).
    """
    import json
    import torch

    from .cs_transformer import CrossSectionalTransformer

    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))

    two_head = manifest.get("two_head", False)

    model = CrossSectionalTransformer(
        n_features=len(features),
        hidden_dim=manifest.get("hidden_dim", 128),
        n_heads=manifest.get("n_heads", 4),
        n_layers=manifest.get("n_layers", 3),
        dropout=0.0,
        two_head=two_head,
    )
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    # Build panel [N, F]
    vals = []
    valid_instruments = []
    for inst in sorted(instruments):
        subset = feature_frame[feature_frame[inst_col] == inst]
        if subset.empty:
            continue
        row = subset.iloc[-1][list(features)].copy()
        for col in features:
            if col in nan_fill and pd.isna(row[col]):
                row[col] = nan_fill[col]
        vals.append(row.values.astype(np.float64))
        valid_instruments.append(inst)

    if not vals:
        return {inst: 0.0 for inst in instruments}

    X = torch.tensor(np.array(vals), dtype=torch.float32).unsqueeze(0)  # [1, N, F]
    with torch.no_grad():
        if two_head:
            raw_scores, risk = model(X, return_risk=True)
            # Stabilized derived score using manifest params
            from .score_stabilizer import stabilize_derived_score, stabilizer_params_from_manifest
            stab_kw = stabilizer_params_from_manifest(manifest)
            derived = stabilize_derived_score(raw_scores, risk, **stab_kw)
            scores = derived.squeeze(0).numpy()  # [N]
        else:
            scores = model(X).squeeze(0).numpy()  # [N]

    predictions = {inst: float(scores[i]) for i, inst in enumerate(valid_instruments)}
    for inst in instruments:
        if inst not in predictions:
            predictions[inst] = 0.0

    return predictions
