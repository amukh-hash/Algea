"""LiveGuard Concept Drift Baselines — MMD with Gaussian RBF kernel.

Computes offline training feature distributions and kernel mean embeddings,
storing baselines to ``state.sqlite3``.  Exposes ``check_mmd()`` for
real-time DAG comparison during production inference.

Halt Condition: MMD_current > MMD_baseline × 1.5 → HALTED_DRIFT
"""
from __future__ import annotations

import io
import logging
import sqlite3
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

STATE_DB = Path("backend/artifacts/orchestrator_state/state.sqlite3")


# ═══════════════════════════════════════════════════════════════════════
# Gaussian RBF Kernel & MMD
# ═══════════════════════════════════════════════════════════════════════

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute Gaussian RBF kernel: k(x,y) = exp(-||x-y||² / 2σ²)."""
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True)
    dist = xx - 2 * x @ y.t() + yy.t()
    return torch.exp(-dist / (2 * sigma ** 2))


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> float:
    """Compute Maximum Mean Discrepancy with Gaussian RBF kernel.

    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N, D)`` — samples from reference distribution P.
    y : torch.Tensor
        Shape ``(M, D)`` — samples from test distribution Q.
    sigma : float
        RBF kernel bandwidth.

    Returns
    -------
    float — MMD² (non-negative).
    """
    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)

    # Remove diagonal self-comparisons
    n = x.size(0)
    m = y.size(0)

    mmd = (k_xx.sum() - k_xx.diag().sum()) / max(n * (n - 1), 1)
    mmd += (k_yy.sum() - k_yy.diag().sum()) / max(m * (m - 1), 1)
    mmd -= 2 * k_xy.mean()

    return max(0.0, mmd.item())


def _adaptive_bandwidth(x: torch.Tensor) -> float:
    """Compute adaptive kernel bandwidth using median heuristic."""
    dists = torch.cdist(x, x)
    median_dist = dists[dists > 0].median().item()
    return max(median_dist, 1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Baseline Storage
# ═══════════════════════════════════════════════════════════════════════

def _tensor_to_blob(t: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes for SQLite BLOB storage."""
    buf = io.BytesIO()
    np.save(buf, t.cpu().numpy())
    return buf.getvalue()


def _blob_to_tensor(blob: bytes) -> torch.Tensor:
    """Deserialize a SQLite BLOB back to a tensor."""
    buf = io.BytesIO(blob)
    arr = np.load(buf)
    return torch.from_numpy(arr)


def save_baseline(
    sleeve: str,
    reference_data: torch.Tensor,
    db_path: Path = STATE_DB,
) -> dict[str, float]:
    """Compute and save MMD baseline for a sleeve.

    Parameters
    ----------
    sleeve : str
        Sleeve identifier (``'kronos'``, ``'mera'``, ``'vrp'``).
    reference_data : torch.Tensor
        Shape ``(N, D)`` — training feature distribution.
    db_path : Path
        Path to the SQLite database.

    Returns
    -------
    dict with baseline statistics.
    """
    sigma = _adaptive_bandwidth(reference_data)

    # Compute kernel mean embedding
    n = reference_data.size(0)
    kernel_mean = _rbf_kernel(reference_data, reference_data, sigma).mean(dim=1)
    feature_mean = reference_data.mean(dim=0)
    feature_var = reference_data.var(dim=0)

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            INSERT OR REPLACE INTO liveguard_baselines
            (sleeve, mmd_kernel_mean, feature_mean, feature_variance, sample_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sleeve,
            _tensor_to_blob(kernel_mean),
            _tensor_to_blob(feature_mean),
            _tensor_to_blob(feature_var),
            n,
        ))
        conn.commit()
    finally:
        conn.close()

    baseline_mmd = compute_mmd(reference_data, reference_data, sigma)
    logger.info(
        "BASELINE %s — σ=%.4f, self-MMD=%.6f, N=%d, D=%d",
        sleeve, sigma, baseline_mmd, n, reference_data.size(1),
    )

    return {
        "sleeve": sleeve,
        "sigma": sigma,
        "self_mmd": baseline_mmd,
        "n_samples": n,
        "feature_dim": reference_data.size(1),
    }


# ═══════════════════════════════════════════════════════════════════════
# Real-Time Drift Check
# ═══════════════════════════════════════════════════════════════════════

def check_mmd(
    sleeve: str,
    current_data: torch.Tensor,
    reference_data: torch.Tensor | None = None,
    threshold_multiplier: float = 1.5,
    db_path: Path = STATE_DB,
) -> dict:
    """Check current data against baseline for concept drift.

    Halt Condition: MMD_current > self_MMD × threshold_multiplier → HALTED_DRIFT

    Parameters
    ----------
    sleeve : str
        Sleeve identifier.
    current_data : torch.Tensor
        Shape ``(M, D)`` — recent production feature data (past 24h).
    reference_data : torch.Tensor or None
        If provided, use as reference directly.  Otherwise load from DB.
    threshold_multiplier : float
        Factor applied to baseline self-MMD to compute halt threshold.
    db_path : Path
        Path to SQLite database.

    Returns
    -------
    dict with ``mmd_score``, ``threshold``, ``is_drifted``.
    """
    if reference_data is not None:
        ref = reference_data
    else:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            row = conn.execute(
                "SELECT feature_mean, feature_variance, sample_count FROM liveguard_baselines WHERE sleeve = ?",
                (sleeve,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            raise ValueError(
                f"No baseline found for sleeve '{sleeve}'. "
                "Run save_baseline() first."
            )

        feature_mean = _blob_to_tensor(row[0])
        feature_var = _blob_to_tensor(row[1])
        n_samples = row[2]

        # Reconstruct approximate reference from stored statistics
        rng = torch.Generator()
        rng.manual_seed(42)
        ref = feature_mean.unsqueeze(0) + torch.sqrt(feature_var + 1e-8).unsqueeze(0) * torch.randn(
            min(n_samples, 1000), feature_mean.size(0), generator=rng
        )

    sigma = _adaptive_bandwidth(ref)
    baseline_mmd = compute_mmd(ref, ref, sigma)
    current_mmd = compute_mmd(ref, current_data, sigma)

    # Floor threshold to prevent zero-threshold false positives when
    # baseline_mmd ≈ 0 (expected for self-comparisons of same distribution)
    min_threshold = 1e-3
    threshold = max(baseline_mmd * threshold_multiplier, min_threshold)

    is_drifted = current_mmd > threshold

    if is_drifted:
        logger.warning(
            "🚨 DRIFT %s — MMD=%.6f > threshold=%.6f (baseline=%.6f × %.1f)",
            sleeve, current_mmd, threshold, baseline_mmd, threshold_multiplier,
        )
    else:
        logger.info(
            "DRIFT OK %s — MMD=%.6f < threshold=%.6f",
            sleeve, current_mmd, threshold,
        )

    return {
        "sleeve": sleeve,
        "mmd_score": current_mmd,
        "baseline_mmd": baseline_mmd,
        "threshold": threshold,
        "is_drifted": is_drifted,
    }
