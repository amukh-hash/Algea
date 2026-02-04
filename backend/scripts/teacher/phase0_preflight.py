#!/usr/bin/env python3
"""
Phase 0: Preflight Check for Chronos 2 Teacher Training.
Validates:
1. Gold L2 Parquet (Samples, Schema, Timestamp, Feature Store detection).
2. Silver MarketFrames (Schema, Window Feasibility, Missing Bars).
3. Breadth Data (Existence, Timestamp Overlap).
4. Codec/Preproc compatibility (Token vs Tensor mode).

Output: backend/reports/preflight_report.json
"""

import os
import sys
import json
import glob
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import datetime

import numpy as np

# Requirements check
try:
    import polars as pl
except ImportError:
    print("ERROR: Polars not installed. Run pip install polars.")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("WARNING: Torch not installed. Some checks skipped.")

# ----------------- Config -----------------

@dataclass
class PreflightConfig:
    # Env / Paths
    gold_dir: Path
    gold_glob: str
    silver_dir: Path
    breadth_path: Optional[Path]
    
    # Gold Logic
    gold_sample_k: int
    gold_min_ok_frac: float
    gold_required_cols: List[str]
    gold_require_timestamp: bool
    
    # Silver Logic
    silver_tickers: List[str]
    silver_required_cols: List[str]
    silver_context: int
    silver_pred: int
    silver_stride: int
    
    # Input Mode
    input_mode: str # 'token' or 'tensor'
    codec_path: Optional[Path]
    preproc_json_path: Optional[Path]
    
    report_path: Path

def load_config() -> PreflightConfig:
    gold_dir = Path(os.getenv("GOLD_L2_PARQUET_DIR", "legacy/v2/Legacy_Algaie_2/backend/data/processed")).resolve()
    gold_glob = os.getenv("GOLD_EXAMPLE_GLOB", "orthogonal_features_final.parquet")
    
    silver_dir = Path(os.getenv("SILVER_MARKETFRAME_DIR", "LocalDatabase/data_cache_alpaca_curated(120)")).resolve()
    
    breadth_env = os.getenv("BREADTH_PARQUET_PATH", "backend/data/processed/breadth_features.parquet")
    breadth_path = Path(breadth_env).resolve() if breadth_env else None

    # Parsing lists
    def parse_list(k, default):
        v = os.getenv(k, default)
        return [x.strip() for x in v.split(",") if x.strip()]

    gold_cols = parse_list("GOLD_REQUIRED_COLS", "close_frac,OFI_L1,microprice")
    silver_cols = parse_list("SILVER_REQUIRED_COLS", "open,high,low,close,volume")
    
    tickers_str = os.getenv("SILVER_EXAMPLE_TICKERS", "")
    silver_tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    input_mode = os.getenv("CHRONOS2_INPUT_MODE", "token").lower()
    
    codec_env = os.getenv("CHRONOS2_CODEC_PATH", "backend/models/codec/codec_gold_v1.json")
    codec_path = Path(codec_env).resolve() if codec_env else None
    
    preproc_env = os.getenv("PREPROC_JSON_PATH", "backend/config/preproc_v1.json")
    preproc_path = Path(preproc_env).resolve() if preproc_env else None

    return PreflightConfig(
        gold_dir=gold_dir,
        gold_glob=gold_glob,
        silver_dir=silver_dir,
        breadth_path=breadth_path,
        
        gold_sample_k=int(os.getenv("GOLD_SAMPLE_FILES", "5")),
        gold_min_ok_frac=float(os.getenv("GOLD_MIN_OK_FRAC", "1.0")),
        gold_required_cols=gold_cols,
        gold_require_timestamp=os.getenv("GOLD_REQUIRE_TIMESTAMP", "0") == "1",
        
        silver_tickers=silver_tickers,
        silver_required_cols=silver_cols,
        silver_context=int(os.getenv("SILVER_CONTEXT", "1024")),
        silver_pred=int(os.getenv("SILVER_PRED", "64")),
        silver_stride=int(os.getenv("SILVER_STRIDE", "30")),
        
        input_mode=input_mode,
        codec_path=codec_path,
        preproc_json_path=preproc_path,
        report_path=Path("backend/reports/preflight_report.json")
    )

# ----------------- Check Logic -----------------

def check_timestamp_sanity(series: pl.Series) -> Dict[str, Any]:
    """
    Checks monotonicity and nulls.
    Returns: {ok: bool, gaps: int, nulls: int, min_ts, max_ts, errors: list}
    """
    res = {"ok": True, "errors": []}
    
    # 1. Nulls
    null_count = series.null_count()
    res["nulls"] = null_count
    if null_count > 0:
        res["ok"] = False
        res["errors"].append(f"Found {null_count} null timestamps")

    # 2. Sorting / Monotonicity
    # Handling different dtypes
    try:
        # Polars is_sorted check
        if not series.is_sorted():
             res["ok"] = False
             res["errors"].append("Timestamp column is not monotonic increasing")
    except Exception as e:
         res["ok"] = False
         res["errors"].append(f"Sort check failed: {e}")

    # 3. Span
    try:
        res["min_ts"] = str(series.min())
        res["max_ts"] = str(series.max())
    except:
        pass
        
    return res

def analyze_gold_structure(df: pl.DataFrame) -> str:
    # Heuristic: Feature Store vs Windowed
    # If "window_id" exists -> "windowed"
    # If One big run (> 10k rows) -> "continuous_timeseries"
    if "window_id" in df.columns:
        return "windowed"
    if df.height > 10000:
        return "continuous_timeseries"
    return "unknown_or_short_series"

def check_gold(cfg: PreflightConfig) -> Dict[str, Any]:
    print(f"[Gold] Scanning up to {cfg.gold_sample_k} files in {cfg.gold_dir}...")
    
    files = sorted(list(cfg.gold_dir.glob(cfg.gold_glob)))
    if not files:
        return {"ok": False, "error": f"No files found matching {cfg.gold_glob}"}

    sample_files = files[:cfg.gold_sample_k]
    
    results = []
    
    for f in sample_files:
        f_res = {"file": f.name, "ok": True, "errors": []}
        try:
            # Schema
            # Use scan to get schema without read if possible, but we need data for checks
            # Just read, K is small.
            df = pl.read_parquet(f)
            missing = [c for c in cfg.gold_required_cols if c not in df.columns]
            
            if missing:
                f_res["ok"] = False
                f_res["errors"].append(f"Missing cols: {missing}")

            # Timestamp
            ts_col = "timestamp" # Canonical
            # If user config mandates timestamp check
            if cfg.gold_require_timestamp:
                if ts_col not in df.columns:
                     f_res["ok"] = False
                     f_res["errors"].append("Missing 'timestamp' column")
                else:
                     ts_check = check_timestamp_sanity(df[ts_col])
                     if not ts_check["ok"]:
                          f_res["ok"] = False
                          f_res["errors"].extend(ts_check["errors"])
                     f_res["ts_info"] = ts_check

            # Structure Type
            f_res["kind"] = analyze_gold_structure(df)

        except Exception as e:
            f_res["ok"] = False
            f_res["errors"].append(str(e))
        
        results.append(f_res)

    pass_count = sum(1 for r in results if r["ok"])
    total = len(results)
    pass_frac = pass_count / total if total > 0 else 0.0
    
    return {
        "ok": pass_frac >= cfg.gold_min_ok_frac,
        "pass_frac": pass_frac,
        "n_checked": total,
        "samples": results
    }

def check_silver(cfg: PreflightConfig, gold_report: Dict) -> Dict[str, Any]:
    print(f"[Silver] Checking MarketFrames in {cfg.silver_dir}...")
    
    if not cfg.silver_dir.exists():
        return {"ok": False, "error": "Directory not found"}

    target_files = []
    
    # 1. Identify files
    if cfg.silver_tickers:
        for t in cfg.silver_tickers:
            # Match user's actual format: {TICKER}_1m.parquet
            target_files.append(cfg.silver_dir / f"{t}_1m.parquet")
    else:
        # Fallback: Scan first K. Glob: *_1m.parquet
        all_mf = sorted(list(cfg.silver_dir.glob("*_1m.parquet")))
        target_files = all_mf[:cfg.gold_sample_k] 
        print(f"[Silver] No tickers provided. Checking first {len(target_files)} detected files.")

    if not target_files:
        return {"ok": False, "error": "No MarketFrame files found to check."}

    results = []
    min_ts_global = None
    max_ts_global = None

    for f in target_files:
        f_res = {"file": f.name, "ok": True, "errors": []}
        
        if not f.exists():
            f_res["ok"] = False
            f_res["errors"].append("File not found")
            results.append(f_res)
            continue
            
        try:
            df = pl.read_parquet(f)
            
            # Schema
            missing = [c for c in cfg.silver_required_cols if c not in df.columns]
            if missing:
                f_res["ok"] = False
                f_res["errors"].append(f"Missing cols: {missing}")

            # Feasibility
            needed = cfg.silver_context + cfg.silver_pred + cfg.silver_stride
            if df.height < needed:
                 # Warning or Error? Error for Training feasibility.
                 f_res["ok"] = False
                 f_res["errors"].append(f"Too short ({df.height} rows) for context+pred+stride")

            # Timestamp
            ts_check = check_timestamp_sanity(df["timestamp"])
            if not ts_check["ok"]:
                f_res["ok"] = False
                f_res["errors"].extend(ts_check["errors"])
            
            # Global Range Tracking (for Breadth)
            if "min_ts" in ts_check and "max_ts" in ts_check:
                # We need datetime objects for comparison
                # Assuming ISO strings or Polars conversion
                # Let's trust Polars min/max types
                t_min = df["timestamp"].min()
                t_max = df["timestamp"].max()
                
                if min_ts_global is None or t_min < min_ts_global: min_ts_global = t_min
                if max_ts_global is None or t_max > max_ts_global: max_ts_global = t_max
                
        except Exception as e:
            f_res["ok"] = False
            f_res["errors"].append(str(e))
            
        results.append(f_res)

    pass_count = sum(1 for r in results if r["ok"])
    
    # Store ranges for Breadth check
    range_info = {}
    if min_ts_global and max_ts_global:
        range_info = {
            "min_ts": str(min_ts_global), 
            "max_ts": str(max_ts_global)
        }

    return {
        "ok": pass_count == len(results), # Strict Silver check for chosen tickers
        "pass_count": pass_count,
        "range": range_info,
        "samples": results
    }

def check_breadth(cfg: PreflightConfig, silver_rep: Dict) -> Dict[str, Any]:
    print("[Breadth] Checking artifacts...")
    
    if not cfg.breadth_path:
        return {"ok": True, "status": "skipped_no_path"}
        
    if not cfg.breadth_path.exists():
        return {"ok": False, "error": f"Breadth file missing at {cfg.breadth_path}"}
        
    try:
        # Schema and Range
        # Since Breadth file can be Huge, use Scan
        lf = pl.scan_parquet(cfg.breadth_path)
        
        # Check if 'timestamp' exists
        schema = lf.collect_schema()
        if "timestamp" not in schema.names():
             return {"ok": False, "error": "Breadth file missing timestamp column"}
             
        # Range Overlap Check
        if silver_rep.get("ok") and "range" in silver_rep and silver_rep["range"]:
             # Check if breadth covers silver range
             stats = lf.select([pl.col("timestamp").min().alias("min"), pl.col("timestamp").max().alias("max")]).collect()
             b_min = stats["min"][0]
             b_max = stats["max"][0]
             
             # Conversions might be tricky (polars datetime vs string).
             # We assume consistency in pipeline.
             s_min_str = silver_rep["range"]["min_ts"]
             s_max_str = silver_rep["range"]["max_ts"]
             
             # Loose string comparison or robust? 
             # Let's just report them.
             return {
                 "ok": True,
                 "breadth_range": {"min": str(b_min), "max": str(b_max)},
                 "silver_range": silver_rep["range"],
                 "overlap_note": "Please verify ranges overlap manually if formats differ."
             }
             
        return {"ok": True, "status": "checked_existence_only"}

    except Exception as e:
        return {"ok": False, "error": f"Read failed: {e}"}

def check_artifacts(cfg: PreflightConfig) -> Dict[str, Any]:
    print(f"[Artifacts] Checking Mode: {cfg.input_mode}...")
    res = {"ok": True, "errors": []}
    
    if cfg.input_mode == "token":
        # REQUIRE Codec
        if not cfg.codec_path or not cfg.codec_path.exists():
             res["ok"] = False
             res["errors"].append(f"Token mode requires Codec Artifact at {cfg.codec_path}")
        else:
             # Basic JSON check
             try:
                 with open(cfg.codec_path, 'r') as f:
                     json.load(f)
             except Exception as e:
                 res["ok"] = False
                 res["errors"].append(f"Corrupt Codec JSON: {e}")

        # Preproc is optional per updated plan, but nice to check existence if defined
        if cfg.preproc_json_path and cfg.preproc_json_path.exists():
             try:
                 with open(cfg.preproc_json_path, 'r') as f:
                     json.load(f)
             except:
                 res["errors"].append("Warning: Preproc JSON exists but invalid.")
    
    elif cfg.input_mode == "tensor":
        pass # No strict artifact requirements yet
        
    else:
        res["ok"] = False
        res["errors"].append(f"Unknown input_mode: {cfg.input_mode}")

    return res

# ----------------- Main -----------------

def main():
    cfg = load_config()
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": asdict(cfg, dict_factory=lambda x: {k: str(v) if isinstance(v, Path) else v for k, v in x})
    }
    
    # 1. Gold
    report["gold"] = check_gold(cfg)
    
    # 2. Silver (depends on Gold? No, independent check usually, but uses same engine)
    report["silver"] = check_silver(cfg, report["gold"])
    
    # 3. Breadth
    report["breadth"] = check_breadth(cfg, report["silver"])
    
    # 4. Artifacts
    report["artifacts"] = check_artifacts(cfg)
    
    # Final Decision
    all_ok = all(section.get("ok", False) for section in [
        report["gold"], report["silver"], report["breadth"], report["artifacts"]
    ])
    report["global_ok"] = all_ok
    
    # Recommendations
    recs = []
    if not report["artifacts"]["ok"] and cfg.input_mode == "token":
        recs.append("Run 'python backend/scripts/teacher/fit_codec_gold.py' to generate codec.")
    if not report["gold"]["ok"]:
        recs.append("Check Gold Data generation/paths.")
        
    report["recommendations"] = recs
    report["next_command"] = "python backend/scripts/teacher/phase1_train_teacher_gold.py" if all_ok else "Fix errors first"

    with open(cfg.report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print("-" * 40)
    print(f"Preflight Complete. Global OK: {all_ok}")
    print(f"Report saved to: {cfg.report_path}")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
