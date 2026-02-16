"""Time-based splits and walk-forward cross-validation."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .types import SplitSpec


# ---------------------------------------------------------------------------
# Time-based single split
# ---------------------------------------------------------------------------

def time_based_split(
    dataset: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    embargo_days: int = 2,
) -> SplitSpec:
    """Chronological train / val / test split.  No shuffling.

    Parameters
    ----------
    dataset : must have a ``trading_day`` column (or index level).
    train_frac : fraction of unique dates for training.
    val_frac : fraction of unique dates for validation.
    embargo_days : gap (in calendar days) between sets.

    Returns
    -------
    SplitSpec
    """
    days = _extract_sorted_days(dataset)
    n = len(days)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    train_days = days[:train_end_idx]
    val_days = days[train_end_idx + embargo_days: val_end_idx]
    test_days = days[val_end_idx + embargo_days:]

    universe: Dict[str, List[str]] = _snapshot_universe(dataset, days)

    return SplitSpec(
        train_start=train_days[0].isoformat() if train_days else "",
        train_end=train_days[-1].isoformat() if train_days else "",
        val_start=val_days[0].isoformat() if val_days else "",
        val_end=val_days[-1].isoformat() if val_days else "",
        test_start=test_days[0].isoformat() if test_days else None,
        test_end=test_days[-1].isoformat() if test_days else None,
        embargo_days=embargo_days,
        fold_index=0,
        universe_snapshot=universe,
    )


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def walk_forward_cv(
    dataset: pd.DataFrame,
    fold_size_days: int = 40,
    embargo_days: int = 2,
    min_train_days: int = 60,
) -> List[SplitSpec]:
    """Expanding-window walk-forward cross-validation.

    For each fold, the training window expands from the start of the
    dataset up to the fold boundary, then the validation window is the
    next ``fold_size_days`` trading days.  An embargo gap of
    ``embargo_days`` trading days separates train and val.

    No overlap.  Deterministic ordering.

    Parameters
    ----------
    dataset : must have ``trading_day`` column (or index level).
    fold_size_days : number of trading days per validation fold.
    embargo_days : trading days gap between train end and val start.
    min_train_days : minimum training days before first fold.

    Returns
    -------
    list of SplitSpec (one per fold)
    """
    days = _extract_sorted_days(dataset)
    n = len(days)
    splits: List[SplitSpec] = []
    fold_idx = 0

    start = min_train_days
    while start + embargo_days + fold_size_days <= n:
        train_days = days[:start]
        val_start_idx = start + embargo_days
        val_end_idx = min(val_start_idx + fold_size_days, n)
        val_days = days[val_start_idx:val_end_idx]

        if not val_days:
            break

        universe = _snapshot_universe(dataset, train_days + val_days)

        splits.append(SplitSpec(
            train_start=train_days[0].isoformat(),
            train_end=train_days[-1].isoformat(),
            val_start=val_days[0].isoformat(),
            val_end=val_days[-1].isoformat(),
            test_start=None,
            test_end=None,
            embargo_days=embargo_days,
            fold_index=fold_idx,
            universe_snapshot=universe,
        ))

        fold_idx += 1
        start += fold_size_days

    return splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sorted_days(dataset: pd.DataFrame) -> List[date]:
    """Extract sorted unique trading days from dataset."""
    if "trading_day" in dataset.columns:
        raw = dataset["trading_day"]
    elif hasattr(dataset.index, "get_level_values"):
        try:
            raw = dataset.index.get_level_values("trading_day")
        except KeyError:
            raw = dataset.index.get_level_values(0)
    else:
        raise ValueError("Cannot find trading_day in dataset")
    unique = sorted(set(d if isinstance(d, date) else d.date() if hasattr(d, "date") else d for d in raw))
    return unique


def _snapshot_universe(
    dataset: pd.DataFrame,
    days: List[date],
) -> Dict[str, List[str]]:
    """Snapshot which roots are present for each day."""
    if "root" in dataset.columns:
        root_col = dataset["root"]
        day_col = dataset["trading_day"] if "trading_day" in dataset.columns else dataset.index.get_level_values("trading_day")
    else:
        root_col = dataset.index.get_level_values("root")
        day_col = dataset.index.get_level_values("trading_day")

    # Build a quick lookup
    day_root: Dict[str, List[str]] = {}
    for d, r in zip(day_col, root_col):
        key = str(d) if not isinstance(d, str) else d
        day_root.setdefault(key, [])
        if r not in day_root[key]:
            day_root[key].append(r)
    # Sort each day's roots
    for k in day_root:
        day_root[k] = sorted(day_root[k])
    return day_root


# ---------------------------------------------------------------------------
# F5: Contiguous OOS split (longer out-of-sample)
# ---------------------------------------------------------------------------

def contiguous_oos_split(
    dataset: pd.DataFrame,
    oos_months: int = 6,
    embargo_days: int = 2,
) -> SplitSpec:
    """Reserve last ``oos_months`` calendar-months as a contiguous OOS slice.

    The remainder is the training set.  Unlike walk-forward CV, there is
    exactly one fold and the OOS slice is long and contiguous — better for
    spotting regime degradation.
    """
    days = _extract_sorted_days(dataset)
    if not days:
        raise ValueError("Empty dataset — cannot split")

    last_day = days[-1]
    oos_cutoff = last_day

    # Walk backward to find the start of the OOS slice
    import calendar as _cal
    months_left = oos_months
    y, m = last_day.year, last_day.month
    while months_left > 0:
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        months_left -= 1
    oos_start_date = date(y, m, 1)

    train_days = [d for d in days if d < oos_start_date]
    oos_days = [d for d in days if d >= oos_start_date]

    if embargo_days > 0 and len(train_days) > embargo_days:
        train_days = train_days[:-embargo_days]

    if not train_days or not oos_days:
        raise ValueError(
            f"Insufficient data for {oos_months}-month OOS split "
            f"({len(train_days)} train, {len(oos_days)} OOS days)"
        )

    universe = _snapshot_universe(dataset, days)

    return SplitSpec(
        train_start=train_days[0].isoformat(),
        train_end=train_days[-1].isoformat(),
        val_start=oos_days[0].isoformat(),
        val_end=oos_days[-1].isoformat(),
        test_start=None,
        test_end=None,
        embargo_days=embargo_days,
        fold_index=0,
        universe_snapshot=universe,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_splits(
    splits: Sequence[SplitSpec],
    output_dir: str | Path,
) -> Path:
    """Write split specs as JSON array and return path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "splits.json"
    data = [s.to_dict() for s in splits]
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    return path
