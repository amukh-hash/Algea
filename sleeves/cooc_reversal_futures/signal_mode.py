"""Signal mode gating for CO→OC reversal futures sleeve.

Determines whether to use heuristic (baseline ``r_co``) or model-pack
predictions for paper/live execution.

Score direction convention
--------------------------
  - Higher score → short (expected lower r_oc → reversal down)
  - Lower score  → long  (expected higher r_oc → reversal up)
  - HEURISTIC baseline: score = ``r_co``
  - MODEL: score = transformer output (trained on ``y = -r_oc``)
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalMode(Enum):
    HEURISTIC = "heuristic"
    MODEL = "model"


def resolve_signal_mode(
    pack_dir: Optional[str | Path] = None,
    require_model_sanity: bool = True,
    require_phase15: bool = True,
    require_trade_proxy: bool = True,
) -> SignalMode:
    """Decide whether to use model or heuristic predictions.

    Parameters
    ----------
    pack_dir
        Path to a production pack.  If None or missing, returns HEURISTIC.
    require_model_sanity
        If True (default), the model_sanity gate must have passed in the
        pack's validation report for MODEL mode to activate.
    require_phase15
        If True (default), the Phase 1.5 alignment report must show
        ``phase15_status: PASS`` for MODEL mode.
    require_trade_proxy
        If True (default), the trade_proxy gate must have passed in the
        pack's validation report for MODEL mode to activate.

    Returns
    -------
    SignalMode
        HEURISTIC or MODEL.

    Notes
    -----
    If **anything** fails (missing pack, missing model, gate fail) the
    function returns HEURISTIC.  This guarantees the Phase 2 runner
    never breaks.
    """
    if pack_dir is None:
        logger.info("No pack_dir → HEURISTIC mode")
        return SignalMode.HEURISTIC

    pack_path = Path(pack_dir)
    if not pack_path.exists():
        logger.warning("Pack dir %s does not exist → HEURISTIC mode", pack_path)
        return SignalMode.HEURISTIC

    # Check model files exist (support both .pt and .pkl)
    model_dir = pack_path / "model"
    has_pt = (model_dir / "model.pt").exists()
    has_pkl = (model_dir / "model.pkl").exists()
    if not (has_pt or has_pkl):
        logger.warning("No model.pt or model.pkl in %s → HEURISTIC mode", model_dir)
        return SignalMode.HEURISTIC

    # Check promotion status (RESEARCH_ONLY packs cannot be promoted)
    schema_path = pack_path / "feature_schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        promo_status = schema.get("promotion_status", "")
        if promo_status == "RESEARCH_ONLY":
            logger.info(
                "Pack is RESEARCH_ONLY (data_provider=%s) → HEURISTIC mode",
                schema.get("data_provider", "unknown"),
            )
            return SignalMode.HEURISTIC

    # Check multi-window promotion report (if present)
    promo_report_path = pack_path / "promotion_windows_report.json"
    if promo_report_path.exists():
        promo_report = json.loads(promo_report_path.read_text(encoding="utf-8"))
        if not promo_report.get("overall_passed", False):
            logger.info(
                "Multi-window promotion not passed → HEURISTIC mode (primary=%s, stress=%d/%d)",
                promo_report.get("primary_passed", False),
                promo_report.get("stress_passed_count", 0),
                promo_report.get("stress_required", 1),
            )
            return SignalMode.HEURISTIC

    # Check validation gates
    if require_model_sanity or require_trade_proxy:
        vr_path = pack_path / "validation_report.json"
        if not vr_path.exists():
            logger.warning("No validation_report.json → HEURISTIC mode")
            return SignalMode.HEURISTIC

        report = json.loads(vr_path.read_text(encoding="utf-8"))
        gates = report.get("gates", [])

        if require_model_sanity:
            sanity_gate = next((g for g in gates if g["name"] == "model_sanity"), None)
            if sanity_gate is None or not sanity_gate.get("passed"):
                logger.info(
                    "model_sanity gate not passed → HEURISTIC mode (detail: %s)",
                    sanity_gate.get("detail") if sanity_gate else "gate missing",
                )
                return SignalMode.HEURISTIC

        if require_trade_proxy:
            tp_gate = next((g for g in gates if g["name"] == "trade_proxy"), None)
            if tp_gate is not None and not tp_gate.get("passed"):
                logger.info(
                    "trade_proxy gate not passed → HEURISTIC mode (detail: %s)",
                    tp_gate.get("detail", ""),
                )
                return SignalMode.HEURISTIC
            # If gate is missing, that's OK (backward compat with old packs)

    # Check Phase 1.5 gate
    if require_phase15:
        manifest_path = pack_path / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            phase15_status = manifest.get("phase15_status", "")
            if phase15_status != "PASS":
                logger.info(
                    "phase15_status=%s (not PASS) → HEURISTIC mode",
                    phase15_status or "missing",
                )
                return SignalMode.HEURISTIC
        else:
            logger.info("No run_manifest.json → cannot verify phase15 → HEURISTIC mode")
            return SignalMode.HEURISTIC

    logger.info("Model pack validated (model_sanity + phase15 + trade_proxy) → MODEL mode from %s", pack_path)
    return SignalMode.MODEL


def heuristic_predictions(
    gold_frame: pd.DataFrame,
    roots: list[str],
) -> Dict[str, float]:
    """Compute baseline reversal scores: ``r_co``.

    Score semantics: higher ``r_co`` → short (reversal from overnight up).
    This aligns with the trade proxy which longs lowest-scored and
    shorts highest-scored instruments.

    Parameters
    ----------
    gold_frame
        Must have columns ``instrument`` (or ``root``) and ``r_co``
        (or ``ret_co``).  We take the most recent row per instrument.
    roots
        Instrument symbols to produce predictions for.

    Returns
    -------
    dict
        ``{instrument: reversal_score}`` based on ``r_co``.
    """
    # Identify column names
    inst_col = "instrument" if "instrument" in gold_frame.columns else "root"
    r_co_col = "r_co" if "r_co" in gold_frame.columns else "ret_co"

    predictions: Dict[str, float] = {}
    for root in roots:
        subset = gold_frame[gold_frame[inst_col] == root]
        if subset.empty:
            logger.warning("No gold data for %s — predicting 0.0", root)
            predictions[root] = 0.0
            continue
        # Most recent row
        latest = subset.iloc[-1]
        r_co = float(latest[r_co_col]) if r_co_col in latest.index else 0.0
        predictions[root] = r_co  # reversal score: higher → short
    return predictions


def model_predictions(
    pack_dir: str | Path,
    feature_frame: pd.DataFrame,
    roots: list[str],
) -> Dict[str, float]:
    """Load model from pack and predict reversal scores.

    Supports both Ridge (.pkl) and Transformer (.pt) model formats.

    Parameters
    ----------
    pack_dir
        Path to production pack.
    feature_frame
        DataFrame with feature columns matching ``feature_schema.json``.
        Must have ``instrument`` (or ``root``) column.
    roots
        Instrument symbols to predict for.
    """
    import pickle

    pack_path = Path(pack_dir)
    schema = json.loads((pack_path / "feature_schema.json").read_text(encoding="utf-8"))
    features = schema["feature_order"]
    nan_fill = schema["nan_fill_values"]

    # Identify column for instrument filtering
    inst_col = "instrument" if "instrument" in feature_frame.columns else "root"

    model_dir = pack_path / "model"

    # Try transformer (.pt) first, then ridge (.pkl)
    if (model_dir / "model.pt").exists():
        return _predict_transformer(model_dir, feature_frame, features, nan_fill, roots, inst_col)

    # Ridge / legacy path
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    predictions: Dict[str, float] = {}
    for root in roots:
        subset = feature_frame[feature_frame[inst_col] == root]
        if subset.empty:
            logger.warning("No feature data for %s — predicting 0.0", root)
            predictions[root] = 0.0
            continue
        latest = subset.iloc[[-1]][features].copy()
        for col in features:
            if col in nan_fill:
                latest[col] = latest[col].fillna(nan_fill[col])
        X = preprocessor.transform(latest)
        pred = float(model.predict(X)[0])
        predictions[root] = pred

    return predictions


def _predict_transformer(
    model_dir: Path,
    feature_frame: pd.DataFrame,
    features: list[str],
    nan_fill: dict,
    roots: list[str],
    inst_col: str,
) -> Dict[str, float]:
    """Transformer inference for a single day panel.

    Loads ``model.pt`` + ``model_manifest.json``, builds a panel
    ``[1, N, F]``, runs forward pass, returns per-instrument scores.
    """
    import torch

    manifest = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))

    # Lazy import to avoid hard torch dependency at module level
    from .model.cs_transformer import CrossSectionalTransformer  # noqa: E402

    model = CrossSectionalTransformer(
        n_features=len(features),
        hidden_dim=manifest.get("hidden_dim", 128),
        n_heads=manifest.get("n_heads", 4),
        n_layers=manifest.get("n_layers", 3),
        dropout=0.0,  # inference mode
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    # Build panel [N, F] for the latest day
    vals = []
    valid_roots = []
    for root in sorted(roots):
        subset = feature_frame[feature_frame[inst_col] == root]
        if subset.empty:
            continue
        row = subset.iloc[-1][features].copy()
        for col in features:
            if col in nan_fill and pd.isna(row[col]):
                row[col] = nan_fill[col]
        vals.append(row.values.astype(np.float64))
        valid_roots.append(root)

    if not vals:
        return {r: 0.0 for r in roots}

    X = torch.tensor(np.array(vals), dtype=torch.float32).unsqueeze(0)  # [1, N, F]
    with torch.no_grad():
        scores = model(X).squeeze(0).numpy()  # [N]

    predictions: Dict[str, float] = {}
    for i, root in enumerate(valid_roots):
        predictions[root] = float(scores[i])
    # Fill missing
    for root in roots:
        if root not in predictions:
            predictions[root] = 0.0

    return predictions
