"""
Build per-date teacher priors cache for one teacher at a time.

Persists partitioned parquet:
    backend/data/selector/priors_cache/teacher={H}d/date={YYYY-MM-DD}/part.parquet

Each partition row contains the 8-key ChronosPriors fields with a
``_{H}`` suffix, plus metadata (run_id, context_len, horizon_days).

Usage
-----
::

    # Cache 10-day teacher over 2024-01-02..2024-01-31
    python backend/scripts/build_priors_cache.py \\
        --teacher-run RUN-2026-02-09-175844 --horizon 10 \\
        --start-date 2024-01-02 --end-date 2024-01-31 \\
        --context-len 252

    # Cache 30-day teacher
    python backend/scripts/build_priors_cache.py \\
        --teacher-run RUN-2026-02-09-181337 --horizon 30 \\
        --start-date 2024-01-02 --end-date 2024-01-31 \\
        --context-len 252
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Teacher loading (reuses eval_chronos_metrics pattern)
# ═══════════════════════════════════════════════════════════════════════════

def load_teacher(run_dir: Path, device: torch.device):
    """Load a Chronos2NativeWrapper with the LoRA checkpoint applied.

    The wrapper stores the full ``Chronos2Pipeline`` as ``self.model`` so
    that ``generate()`` can call ``self.model.predict()``.
    """
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, inject_adapter_in_model

    from algae.models.foundation.chronos2_teacher import Chronos2NativeWrapper

    config = json.loads((run_dir / "config.json").read_text())
    model_id = config["model_id"]

    logger.info(f"Loading base pipeline: {model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_id, device_map=None, dtype=torch.float32,
    )
    inner_model = pipeline.model.to(device)

    # LoRA injection
    target_modules = ["q", "v", "k", "o"]
    peft_cfg = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
    )
    inner_model = inject_adapter_in_model(peft_cfg, inner_model)

    # Load best/final checkpoint
    best_ckpt = run_dir / "checkpoints" / "best" / "model.pt"
    final_ckpt = run_dir / "checkpoints" / "final" / "model.pt"
    ckpt = best_ckpt if best_ckpt.exists() else final_ckpt
    state = torch.load(ckpt, map_location=device, weights_only=True)
    inner_model.load_state_dict(state, strict=False)
    logger.info(f"Loaded checkpoint: {ckpt}")

    inner_model.eval()
    pipeline.model = inner_model  # update pipeline's model ref

    # Build Chronos2NativeWrapper:
    # __init__ needs an object with .parameters() for device detection.
    # Pass inner_model for init, then swap self.model to pipeline for
    # generate() → pipeline.predict() compatibility.
    import inspect
    fwd_params = list(inspect.signature(inner_model.forward).parameters.keys())
    wrapper = Chronos2NativeWrapper(
        model=inner_model,  # has .parameters() for device detection
        model_type="chronos2",
        forward_params=fwd_params,
    )
    # nn.Module.__setattr__ rejects non-Module assignments, so bypass it
    object.__setattr__(wrapper, "model", pipeline)
    wrapper._enable_q10d_head = False  # NLL teachers
    wrapper.eval()

    return wrapper, config


# ═══════════════════════════════════════════════════════════════════════════
# Trading calendar helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_trading_dates(
    universe_frame: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> List[date]:
    """Extract sorted unique trading dates from universe_frame in range."""
    uf = universe_frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(uf["date"]):
        uf["date"] = pd.to_datetime(uf["date"])
    dates_in_range = uf[
        (uf["date"].dt.date >= start_date) & (uf["date"].dt.date <= end_date)
    ]["date"].dt.date.unique()
    return sorted(dates_in_range)


def get_tradable_symbols(
    universe_frame: pd.DataFrame,
    target_date: date,
) -> List[str]:
    """Return sorted list of tradable symbols on a given date."""
    uf = universe_frame
    if not pd.api.types.is_datetime64_any_dtype(uf["date"]):
        uf = uf.copy()
        uf["date"] = pd.to_datetime(uf["date"])
    mask = (uf["date"].dt.date == target_date) & (uf["is_tradable"] == True)
    symbols = sorted(uf.loc[mask, "symbol"].unique().tolist())
    return symbols


# ═══════════════════════════════════════════════════════════════════════════
# Context loading
# ═══════════════════════════════════════════════════════════════════════════

def load_ticker_context(
    symbol: str,
    target_date: date,
    context_len: int,
    ticker_dir: Path,
) -> Optional[torch.Tensor]:
    """Load last ``context_len`` closes up to ``target_date`` (inclusive).

    Returns
    -------
    torch.Tensor of shape ``[context_len]`` or None if insufficient history.
    """
    fpath = ticker_dir / f"{symbol}.parquet"
    if not fpath.exists():
        return None
    df = pd.read_parquet(fpath)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= target_date].sort_values("date")
    if len(df) < context_len:
        return None
    closes = df["close"].iloc[-context_len:].values.astype(np.float32)
    if np.any(~np.isfinite(closes)) or np.any(closes <= 0):
        return None
    return torch.tensor(closes, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Main cache-building logic
# ═══════════════════════════════════════════════════════════════════════════

def build_cache_for_date(
    teacher,
    symbols: List[str],
    target_date: date,
    horizon_days: int,
    context_len: int,
    ticker_dir: Path,
    run_id: str,
    batch_size: int = 16,
) -> Optional[pd.DataFrame]:
    """Compute teacher priors for all ``symbols`` on ``target_date``.

    Uses the canonical ``compute_teacher_priors`` function.
    """
    from algae.data.priors.chronos_priors_compute import (
        compute_teacher_priors,
        priors_to_row,
    )

    # Load contexts
    contexts = {}
    for sym in symbols:
        ctx = load_ticker_context(sym, target_date, context_len, ticker_dir)
        if ctx is not None:
            contexts[sym] = ctx

    if not contexts:
        logger.warning(f"  {target_date}: no tickers with sufficient history")
        return None

    skipped = len(symbols) - len(contexts)
    if skipped > 0:
        logger.info(f"  {target_date}: {len(contexts)} tickers loaded, {skipped} skipped (insufficient history)")

    # Sort for determinism
    sym_list = sorted(contexts.keys())
    # Process in batches
    h_suffix = str(horizon_days)
    rows: List[Dict] = []

    for batch_start in range(0, len(sym_list), batch_size):
        batch_syms = sym_list[batch_start : batch_start + batch_size]
        batch_tensor = torch.stack(
            [contexts[s].unsqueeze(-1) for s in batch_syms]
        )  # [B, T, 1]

        try:
            priors_list = compute_teacher_priors(
                teacher=teacher,
                input_tensor=batch_tensor,
                horizon_days=horizon_days,
                strict=True,
                mode="native_nll",
            )
        except Exception as e:
            logger.warning(f"  {target_date} batch starting {batch_syms[0]}: priors failed: {e}")
            continue

        for sym, p in zip(batch_syms, priors_list):
            row = priors_to_row(p, h_suffix)
            row["date"] = target_date
            row["symbol"] = sym
            row["run_id"] = run_id
            row["context_len"] = context_len
            row["horizon_days"] = horizon_days
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # ── Item A: strict row-level validation ────────────────────────────
    # Refuse to write invalid rows rather than silently fixing them.
    h = str(horizon_days)
    q10_col, q50_col, q90_col = f"q10_{h}", f"q50_{h}", f"q90_{h}"
    disp_col = f"dispersion_{h}"

    valid_mask = (
        df[q10_col].le(df[q50_col] + 1e-8)
        & df[q50_col].le(df[q90_col] + 1e-8)
        & df[disp_col].ge(-1e-8)
        & df[[q10_col, q50_col, q90_col, disp_col]].apply(
            lambda s: np.isfinite(s).all(), axis=1)
    )
    n_rejected = (~valid_mask).sum()
    if n_rejected > 0:
        logger.warning(f"  {target_date}: REJECTING {n_rejected}/{len(df)} rows "
                       f"with invalid priors (monotonicity/dispersion/finite)")
        df = df[valid_mask].reset_index(drop=True)

    if df.empty:
        return None

    # ── Item E: symbols metadata ──────────────────────────────────────
    syms_sorted = sorted(df["symbol"].unique().tolist())
    df["universe_count"] = len(syms_sorted)
    df["symbols_hash"] = hashlib.sha256(
        ",".join(syms_sorted).encode()
    ).hexdigest()[:12]

    return df


def main():
    parser = argparse.ArgumentParser(description="Build teacher priors cache")
    parser.add_argument("--teacher-run", required=True, help="Run ID, e.g. RUN-2026-02-09-175844")
    parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon in trading days")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--context-len", type=int, default=252)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-root", default=None, help="Override cache output dir")
    args = parser.parse_args()

    from algae.core.device import get_device
    device = get_device()

    runs_dir = ROOT / "backend" / "data" / "runs"
    run_dir = runs_dir / args.teacher_run
    if not run_dir.exists():
        logger.error(f"Run not found: {run_dir}")
        sys.exit(1)

    # Output directory
    cache_root = Path(args.cache_root) if args.cache_root else (
        ROOT / "backend" / "data" / "selector" / "priors_cache"
    )
    teacher_cache_dir = cache_root / f"teacher={args.horizon}d"

    # Load universe frame
    uf_path = ROOT / "backend" / "data" / "canonical" / "universe_frame.parquet"
    logger.info(f"Loading universe frame: {uf_path}")
    uf = pd.read_parquet(uf_path)
    uf["date"] = pd.to_datetime(uf["date"])

    # Get trading dates
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    trading_dates = get_trading_dates(uf, start, end)
    logger.info(f"Trading dates in range: {len(trading_dates)}")

    # Load teacher
    teacher, config = load_teacher(run_dir, device)

    ticker_dir = ROOT / "backend" / "data" / "canonical" / "per_ticker"

    processed = 0
    skipped_existing = 0
    for i, td in enumerate(trading_dates):
        date_str = td.isoformat()
        out_dir = teacher_cache_dir / f"date={date_str}"
        out_file = out_dir / "part.parquet"

        # Skip if already cached
        if out_file.exists():
            skipped_existing += 1
            continue

        symbols = get_tradable_symbols(uf, td)
        if not symbols:
            logger.info(f"  {date_str}: no tradable symbols, skipping")
            continue

        logger.info(f"[{i+1}/{len(trading_dates)}] {date_str}: {len(symbols)} symbols")

        df = build_cache_for_date(
            teacher=teacher,
            symbols=symbols,
            target_date=td,
            horizon_days=args.horizon,
            context_len=args.context_len,
            ticker_dir=ticker_dir,
            run_id=args.teacher_run,
            batch_size=args.batch_size,
        )

        if df is not None and len(df) > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_file, index=False)
            processed += 1
            logger.info(f"  → Wrote {len(df)} rows to {out_file}")

    logger.info(f"Done. Processed={processed}, SkippedExisting={skipped_existing}")


if __name__ == "__main__":
    main()
