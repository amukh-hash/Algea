
import logging
import argparse
import os
from backend.app.ops import bootstrap, pathmap, config
from backend.app.selector import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_end", default="2025-12-31")
    parser.add_argument("--val_pct", type=float, default=0.1)
    parser.add_argument("--test_pct", type=float, default=0.1)
    parser.add_argument("--embargo_td", type=int, default=config.LABEL_HORIZON_TD)
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    logger.info("Starting Ranker Training...")
    
    # Run config
    cfg = {
        "train_end": args.train_end,
        "epochs": 10,
        "batch_size": 32,
        "val_pct": args.val_pct,
        "test_pct": args.test_pct,
        "embargo_td": args.embargo_td
    }
    
    try:
        result = train.train_selector(cfg)
        logger.info("Training complete.")
        
        # Promotion Gate would happen here or in next script? 
        # Requirement D says: train + calibrate + promote (if gate passes)
        
        # Logic for promotion gate call:
        # promote(result)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
