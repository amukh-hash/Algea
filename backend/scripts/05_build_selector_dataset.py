
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
    parser.add_argument("--horizon", type=int, default=config.LABEL_HORIZON_TD)
    parser.add_argument("--seq_len", type=int, default=config.SEQUENCE_LEN)
    parser.add_argument("--min_group_size", type=int, default=200)
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    logger.info("Building Selector Dataset...")
    
    try:
        # 1. Build Labels (Schema B8)
        labels = dataset_builder.build_labels_fwd(args.start, args.end, horizon_td=args.horizon)
        labels_path = pathmap.resolve("labels", horizon=args.horizon)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        labels.to_parquet(labels_path)
        logger.info(f"Labels written: {labels_path}")
        
        # 2. Build Rank Dataset (Tensors)
        dataset = dataset_builder.build_rank_dataset(
            args.start,
            args.end,
            sequence_len=args.seq_len,
            min_group_size=args.min_group_size,
            horizon_td=args.horizon
        )
        dataset_path = pathmap.resolve("dataset_selector")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        import torch
        torch.save(dataset, dataset_path)
        logger.info(f"Dataset tensors written: {dataset_path}")

        groups_path = pathmap.resolve("dataset_selector_groups")
        groups_df = pd.DataFrame(dataset.get("groups", []))
        if not groups_df.empty:
            groups_df.to_parquet(groups_path)
            logger.info(f"Dataset groups written: {groups_path}")
        
        logger.info("Dataset built successfully.")
        
    except Exception as e:
        logger.error(f"Dataset build failed: {e}")
        # Fail hard?
        pass

if __name__ == "__main__":
    main()
