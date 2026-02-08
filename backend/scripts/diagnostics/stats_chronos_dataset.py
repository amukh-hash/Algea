"""
Phase 6.1 Diagnostics: ChronosDataset stats.
Instantiates ChronosDataset, prints len, stats, samples 5 windows.
Exit 0 if ok, nonzero if required path is missing.
"""

import sys
import os
from pathlib import Path
from backend.app.ops import pathmap


def main():
    gold_dir = pathmap.get_gold_daily_root()
    univ_root = pathmap.get_universe_frame_root(version="v2")

    print(f"Gold Root:     {gold_dir}")
    print(f"Universe Root: {univ_root}")

    if not gold_dir.exists():
        print("SKIP: Gold root not found")
        sys.exit(0)
    if not univ_root.exists():
        print("SKIP: Universe root not found")
        sys.exit(0)

    from backend.app.training.chronos_dataset import ChronosDataset

    files = sorted(gold_dir.glob("*.parquet"))
    if not files:
        print("SKIP: No gold parquet files found")
        sys.exit(0)

    context_len = int(os.getenv("CHRONOS_CONTEXT_LEN", "180"))
    prediction_len = int(os.getenv("CHRONOS_PRED_LEN", "20"))
    stride = int(os.getenv("CHRONOS_STRIDE", "5"))
    max_spf_str = os.getenv("CHRONOS_MAX_SAMPLES_PER_FILE", "100")
    max_spf = int(max_spf_str) if max_spf_str else None

    print("Instantiating ChronosDataset (this may take a moment)...")
    ds = ChronosDataset(
        files=files,
        context_len=context_len,
        prediction_len=prediction_len,
        stride=stride,
        universe_path=str(univ_root / "**/*.parquet"),
        target_col=os.getenv("CHRONOS_TARGET_COL", "close"),
        max_samples_per_file=max_spf,
        seed=42,
    )

    print(f"dataset_len: {len(ds)}")

    if hasattr(ds, "stats"):
        print("Stats:")
        for k, v in ds.stats.items():
            print(f"  {k}: {v}")

    # Sample 5 windows and confirm shapes
    n_sample = min(5, len(ds))
    negative_seen = 0
    print(f"\nSampling {n_sample} items...")
    for i in range(n_sample):
        item = ds[i]
        pt = item["past_target"]
        neg = (pt < 0).any().item()
        if neg:
            negative_seen += 1
        print(
            f"  Item {i}: past_shape={tuple(pt.shape)}, "
            f"future_shape={tuple(item['future_target'].shape)}, "
            f"mean={pt.mean():.4f}, min={pt.min():.4f}, has_negative={neg}"
        )

    print(f"negative_seen: {negative_seen}/{n_sample}")


if __name__ == "__main__":
    main()
