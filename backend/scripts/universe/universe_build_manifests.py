
import sys
import os
import shutil
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import argparse
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from backend.app.ops import bootstrap, pathmap, config, run_recorder
from backend.app.data import universe_selector as universe, security_master
from backend.app.data.ingest.ohlcv_daily import load_ohlcv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-02-03")
    parser.add_argument("--freq", default="monthly")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    run_id = run_recorder.init_run(
        pipeline_type="full_pipeline",
        trigger="manual",
        config={"start": args.start, "end": args.end, "freq": args.freq},
        data_versions={"gold": "unknown", "silver": "unknown", "macro": "unknown", "universe": "unknown"},
        tags=["universe"],
    )
    run_dir = run_recorder.run_paths.get_run_dir(run_id)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Master
    sec_master = security_master.load_security_master()
    
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    current = start_dt
    
    try:
        run_recorder.set_status(run_id, "RUNNING", stage="universe", step="build_manifests")
        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(f"Building Manifest for {date_str}...")
            run_recorder.emit_event(run_id, "universe", "build", "INFO", f"Building manifest for {date_str}")
        
        # Real Calculation using universe.py logic
        # Optimize: Pass only active symbols?
        # For now, pass all master symbols, the function loads OHLCV per symbol.
        # This allows accurate Point-in-Time calculation.
        
        # Define base rules
        rules = {
            "min_price": 5.0,
            "min_adv": 25e6, # $25M
            "min_ipo_days": 252,
            "top_n": 1000  # Broad universe
        }
        
        base_symbols = sec_master["symbol"].tolist()
        
        # Build Manifest
        # usage: build_universe_manifest(asof_date, base_symbols, rules)
        # Note: This loads OHLCV for every symbol. It might be slow.
        # But it's the correct "functional" path.
        manifest = universe.build_universe_manifest(date_str, base_symbols, rules)
        
        # Write
            if not manifest.empty:
                out = universe.write_universe_manifest(current, manifest, rules)
                run_output = outputs_dir / f"universe_manifest_{date_str}.parquet"
                run_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, run_output)
                logger.info(f"  Written {run_output} | Eligible: {manifest['eligible'].sum()}")
                run_recorder.register_artifact(
                    run_id,
                    name=f"universe_manifest_{date_str}",
                    type="parquet",
                    path=str(run_output),
                    tags=["universe", "manifest"],
                )
            else:
                logger.warning(f"  Empty manifest for {date_str}")
            
            current += relativedelta(months=1)
        run_recorder.finalize_run(run_id, "PASSED")
    except Exception as exc:
        run_recorder.set_status(
            run_id,
            "FAILED",
            stage="universe",
            step="error",
            error={"type": type(exc).__name__, "message": str(exc), "traceback": ""},
        )
        run_recorder.finalize_run(run_id, "FAILED")
        raise

if __name__ == "__main__":
    main()
