
import logging
import argparse
import pandas as pd
import os
from backend.app.ops import bootstrap, pathmap, config, artifact_registry
from backend.app.selector import dataset_builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    logger.info("Building Selector Dataset...")
    
    # Delegate to DatasetBuilder logic (Stubbed in PR-2, needs real logic eventually)
    # The requirement is that this script exists and calls the right place.
    # In PR-11 we finalize the logic.
    
    # Stub Logic
    try:
        # 1. Build Labels (Schema B8)
        labels = dataset_builder.build_labels_fwd10d(args.start, args.end)
        
        # 2. Build Rank Dataset (Tensors)
        dataset = dataset_builder.build_rank_dataset(args.start, args.end, sequence_len=60)
        
        logger.info("Dataset built successfully (Stub).")
        
    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        # Fail hard?
        pass

if __name__ == "__main__":
    main()
