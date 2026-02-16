"""Export production pack: model bundle, feature schema, splits, validation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import ModelBundle, RunManifest, SplitSpec, ValidationReport


def export_production_pack(
    run_manifest: RunManifest,
    output_dir: str | Path,
) -> Path:
    """Produce a self-contained production pack directory.

    Contents
    --------
    production_pack/
        run_manifest.json
        model/           (model.pkl, preprocessor.pkl, model_manifest.json)
        feature_schema.json
        splits.json
        validation_report.json
        session_semantics_report.json   (if Phase 1.5 ran)
        feature_parity_report.json      (if Phase 1.5 ran)
        coverage_report.json            (if Phase 1.5 ran)
        trade_proxy_report.json         (if Phase 1.5 ran)

    Parameters
    ----------
    run_manifest : complete RunManifest from the pipeline run
    output_dir : destination for the production pack

    Returns
    -------
    Path to the production pack directory
    """
    pack_dir = Path(output_dir) / "production_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    # --- Run manifest ---
    (pack_dir / "run_manifest.json").write_text(run_manifest.to_json())

    # --- Model bundle ---
    if run_manifest.model is not None:
        model_dir = pack_dir / "model"
        model_dir.mkdir(exist_ok=True)
        model_src = Path(run_manifest.model.model_path)
        if model_src.exists():
            shutil.copy2(model_src, model_dir / model_src.name)
        scaler_src = Path(run_manifest.model.scaler_path)
        if scaler_src.exists():
            shutil.copy2(scaler_src, model_dir / scaler_src.name)
        manifest_src = model_src.parent / "model_manifest.json"
        if manifest_src.exists():
            shutil.copy2(manifest_src, model_dir / "model_manifest.json")

    # --- Feature schema ---
    if run_manifest.model is not None:
        # Determine data provider and promotion status
        chosen = run_manifest.model.chosen_params
        data_provider = chosen.get("data_provider", "yfinance")
        promotion_status = "PROMOTABLE" if data_provider == "ibkr_hist" else "RESEARCH_ONLY"

        schema: Dict[str, Any] = {
            "feature_order": list(run_manifest.model.feature_order),
            "nan_fill_values": run_manifest.model.nan_fill_values,
            "chosen_params": chosen,
            "estimator": chosen.get("estimator", "Ridge"),
            "data_provider": data_provider,
            "promotion_status": promotion_status,
            "risk_head_enabled": chosen.get("risk_head_enabled", False),
            "risk_target_name": chosen.get("risk_target_name", "none"),
            "risk_target_eps": chosen.get("risk_target_eps", 1e-6),
            "risk_pred_clamp_min": chosen.get("risk_pred_clamp_min", 0.05),
            "risk_pred_clamp_max": chosen.get("risk_pred_clamp_max", 5.0),
            "score_tanh": chosen.get("score_tanh", False),
            "derived_score_clip": chosen.get("derived_score_clip", 10.0),
        }
        (pack_dir / "feature_schema.json").write_text(
            json.dumps(schema, indent=2, sort_keys=True)
        )

    # --- Splits ---
    splits_data = [s.to_dict() for s in run_manifest.splits]
    (pack_dir / "splits.json").write_text(
        json.dumps(splits_data, indent=2, sort_keys=True, default=str)
    )

    # --- Validation report ---
    if run_manifest.validation is not None:
        (pack_dir / "validation_report.json").write_text(
            json.dumps(run_manifest.validation.to_dict(), indent=2, sort_keys=True)
        )

    # --- Phase 1.5 reports ---
    if run_manifest.phase15 is not None:
        p15 = run_manifest.phase15

        if p15.session_semantics is not None:
            (pack_dir / "session_semantics_report.json").write_text(
                json.dumps(p15.session_semantics.to_dict(), indent=2, sort_keys=True)
            )

        if p15.feature_parity is not None:
            (pack_dir / "feature_parity_report.json").write_text(
                json.dumps(p15.feature_parity.to_dict(), indent=2, sort_keys=True, default=str)
            )

        if p15.coverage is not None:
            (pack_dir / "coverage_report.json").write_text(
                json.dumps(p15.coverage.to_dict(), indent=2, sort_keys=True)
            )

        if p15.trade_proxy is not None:
            (pack_dir / "trade_proxy_report.json").write_text(
                json.dumps(p15.trade_proxy.to_dict(), indent=2, sort_keys=True)
            )

    return pack_dir

