
import logging
import argparse
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from backend.app.ops import bootstrap, pathmap, config
from backend.app.teacher import priors, chronos_runner
from backend.app.data import security_master

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-02-03")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    # Config
    spec = {
        "model": "chronos-t5-small",
        "context": config.CONTEXT_LEN,
        "horizon": config.PRIOR_HORIZON
    }
    
    # Init Runner (Stub/Wrapper)
    runner = chronos_runner.ChronosRunner(
        model_id=spec["model"],
        context_len=spec["context"],
        horizon=spec["horizon"]
    )
    
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    current = start_dt
    
    # Iterate Days (Trading Days Only)
    from backend.app.data import calendar
    trading_days = calendar.get_trading_days(start_dt, end_dt)
    
    for current in trading_days:
        date_str = current.strftime("%Y-%m-%d")
        logger.info(f"Generating Priors for {date_str}...")
        
        # 1. Get Universe (from Manifest or simplistic Security Master)
        # Ideally: load manifest for this date.
        # Stub: use dummy list
        symbols = ["AAPL", "MSFT", "GOOGL"] 
        
        # 2. Generate
        try:
            df = priors.generate_priors_for_date(current, symbols, spec, runner)
            
            # 3. Write
            path = priors.write_priors(current, df, spec)
            logger.info(f"  Written to {path}")
            
        except Exception as e:
            logger.error(f"  Failed for {date_str}: {e}")
            if config.FAIL_ON_MISSING_DIRS: # Reuse flag for strictness?
                pass
        


if __name__ == "__main__":
    main()
