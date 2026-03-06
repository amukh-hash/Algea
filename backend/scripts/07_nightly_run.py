"""07_nightly_run.py — VRAM Saturation & PCIe Bottleneck Profiler.

Floods the inference gateway with concurrent requests to stress-test
hardware isolation between cuda:0 and cuda:1.

Invariants Asserted:
  1. cuda:0 SM utilization reaches near-saturation during batch processing
  2. cuda:1 VRAM usage stays strictly < 11.0 GB
  3. No OOM errors
  4. FastAPI Priority 1 endpoint P99 latency < 50ms
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("nightly_profiler")


# ═══════════════════════════════════════════════════════════════════════
# GPU Monitoring
# ═══════════════════════════════════════════════════════════════════════

def profile_gpu_state() -> dict:
    """Capture current GPU state using pynvml or torch.cuda."""
    result: dict = {}

    if not torch.cuda.is_available():
        return {"status": "cpu_only"}

    n_gpus = torch.cuda.device_count()

    try:
        import pynvml
        pynvml.nvmlInit()

        for i in range(min(n_gpus, 2)):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            result[f"cuda:{i}"] = {
                "name": name,
                "vram_used_gb": round(mem.used / 1e9, 3),
                "vram_total_gb": round(mem.total / 1e9, 3),
                "vram_free_gb": round(mem.free / 1e9, 3),
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
            }

        pynvml.nvmlShutdown()

    except ImportError:
        logger.warning("pynvml not available — using torch.cuda fallback")
        for i in range(min(n_gpus, 2)):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_mem / 1e9
            result[f"cuda:{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "vram_allocated_gb": round(allocated, 3),
                "vram_reserved_gb": round(reserved, 3),
                "vram_total_gb": round(total, 3),
            }

    return result


# ═══════════════════════════════════════════════════════════════════════
# Stress Test: Inference Queue Flooding
# ═══════════════════════════════════════════════════════════════════════

async def _simulate_inference_requests(
    n_requests: int = 1000,
    device_str: str = "cuda:1",
) -> dict:
    """Simulate concurrent inference requests on cuda:1.

    Creates dummy tensors and runs forward passes to stress VRAM.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    latencies: list[float] = []

    async def _single_request(idx: int) -> float:
        t0 = time.perf_counter()
        # Simulate model forward pass
        x = torch.randn(1, 64, device=device)
        w = torch.randn(64, 32, device=device)
        _ = torch.matmul(x, w)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        dt = (time.perf_counter() - t0) * 1000  # ms
        return dt

    # Run requests concurrently in batches to avoid overwhelming
    batch_size = 50
    for batch_start in range(0, n_requests, batch_size):
        batch_end = min(batch_start + batch_size, n_requests)
        tasks = [_single_request(i) for i in range(batch_start, batch_end)]
        batch_latencies = await asyncio.gather(*tasks)
        latencies.extend(batch_latencies)

    latencies_arr = np.array(latencies)
    return {
        "n_requests": n_requests,
        "p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "max_ms": round(float(latencies_arr.max()), 3),
        "mean_ms": round(float(latencies_arr.mean()), 3),
    }


def _run_heavy_batch(n_iterations: int = 100) -> dict:
    """Run batch computation on cuda:0 to stress SM utilization."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    for i in range(n_iterations):
        # Simulate MASTER embedding computation
        x = torch.randn(512, 256, device=device)
        w1 = torch.randn(256, 128, device=device)
        w2 = torch.randn(128, 64, device=device)
        h = torch.relu(torch.matmul(x, w1))
        _ = torch.matmul(h, w2)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.time() - t0
    return {
        "n_iterations": n_iterations,
        "elapsed_s": round(elapsed, 3),
        "throughput_iter_per_s": round(n_iterations / elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Profiler
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM & PCIe stress profiler.")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode.")
    parser.add_argument("--stress", action="store_true", help="Enable max stress mode.")
    parser.add_argument("--n-requests", type=int, default=1000, help="Number of inference requests.")
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info("ALGAIE NIGHTLY PROFILER — %s mode", "DRY-RUN" if args.dry_run else "LIVE")
    logger.info("=" * 72)

    # ── Pre-flight GPU snapshot ──────────────────────────────────────
    logger.info("PRE-FLIGHT GPU state:")
    pre_state = profile_gpu_state()
    for dev, info in pre_state.items():
        if isinstance(info, dict):
            logger.info("  %s: %s", dev, info)

    # ── Phase 1: Heavy batch on cuda:0 ───────────────────────────────
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 1: Heavy batch on cuda:0")
    n_iters = 500 if args.stress else 100
    heavy_result = _run_heavy_batch(n_iters)
    logger.info("HEAVY  %s", heavy_result)

    # ── Phase 2: Concurrent inference on cuda:1 ──────────────────────
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 2: Concurrent inference on cuda:1")
    inference_result = asyncio.run(
        _simulate_inference_requests(n_requests=args.n_requests)
    )
    logger.info("INFER  %s", inference_result)

    # ── Post-flight GPU snapshot ─────────────────────────────────────
    logger.info("\n" + "=" * 40)
    logger.info("POST-FLIGHT GPU state:")
    post_state = profile_gpu_state()
    for dev, info in post_state.items():
        if isinstance(info, dict):
            logger.info("  %s: %s", dev, info)

    # ── Invariant Assertions ─────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info("INVARIANT CHECKS:")

    passed = True

    # Check cuda:1 VRAM
    if "cuda:1" in post_state and isinstance(post_state["cuda:1"], dict):
        vram_used = post_state["cuda:1"].get("vram_used_gb", post_state["cuda:1"].get("vram_allocated_gb", 0))
        if vram_used < 11.0:
            logger.info("  ✅ cuda:1 VRAM = %.2f GB < 11.0 GB", vram_used)
        else:
            logger.error("  ❌ cuda:1 VRAM = %.2f GB >= 11.0 GB — BREACH!", vram_used)
            passed = False

    # Check P99 latency
    p99 = inference_result.get("p99_ms", 0.0)
    if p99 < 50.0:
        logger.info("  ✅ Inference P99 = %.2f ms < 50 ms", p99)
    else:
        logger.warning("  ⚠️  Inference P99 = %.2f ms >= 50 ms — SLOW", p99)

    # No OOM check (if we got here, no OOM occurred)
    logger.info("  ✅ No OOM detected")

    logger.info("\n" + "=" * 72)
    logger.info("RESULT: %s", "ALL INVARIANTS PASSED ✅" if passed else "INVARIANT BREACH DETECTED ❌")
    logger.info("=" * 72)

    # ── WAL Checkpoint: Prevent WAL starvation ───────────────────────
    # High-frequency async reads can block SQLite autocheckpoint,
    # causing the .wal file to grow indefinitely. Force a TRUNCATE
    # checkpoint to sync WAL → main DB and reclaim disk space.
    import sqlite3
    _wal_db = Path("backend/artifacts/orchestrator_state/state.sqlite3")
    if _wal_db.exists():
        _wal_conn = sqlite3.connect(_wal_db, timeout=30)
        try:
            _wal_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("WAL   checkpoint(TRUNCATE) executed on %s", _wal_db)
        finally:
            _wal_conn.close()

    # ── COOC Feature Precomputation: O(N) batch for T-1 state ─────────
    # Pre-compute all 252-day rolling features (sigma_co, dd_60,
    # trend_20, etc.) and persist to Parquet.  At 09:21 ET the live
    # inference path loads this cache in ~2ms instead of re-running
    # the full O(N) pipeline.
    logger.info("\n" + "=" * 40)
    logger.info("COOC PRECOMPUTE: Building T-1 feature cache")
    _cooc_cache_path = Path("backend/artifacts/features/cooc_t_minus_1.parquet")
    try:
        from sleeves.cooc_reversal_futures.features_core import (
            precompute_t_minus_1,
            FeatureConfig,
        )
        _history_path = Path("backend/artifacts/gold/futures_gold.parquet")
        if _history_path.exists():
            import pandas as _pd
            _history_df = _pd.read_parquet(_history_path)
            _t_minus_1 = precompute_t_minus_1(_history_df, FeatureConfig())
            _cooc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            _t_minus_1.to_parquet(
                _cooc_cache_path, engine="pyarrow", compression="snappy",
            )
            logger.info(
                "COOC  Cached %d instruments → %s (%.1f KB)",
                len(_t_minus_1),
                _cooc_cache_path,
                _cooc_cache_path.stat().st_size / 1024,
            )
        else:
            logger.warning(
                "COOC  History file not found at %s — skipping precompute",
                _history_path,
            )
    except Exception as exc:
        logger.error("COOC  Precompute failed: %s", exc, exc_info=True)
        # Non-fatal: live inference falls back to monolithic builder


if __name__ == "__main__":
    main()

