"""Wasserstein Optimal-Transport regime detection.

Replaces Gaussian HMMs with non-parametric optimal transport to handle
fat tails and heteroskedasticity.  Returns **continuous probability
vectors** — no discrete regime flags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import ot  # Python Optimal Transport (POT)
except ImportError:  # pragma: no cover
    ot = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class WassersteinRegimeCluster:
    """Non-parametric regime detector using Wasserstein-1 distances.

    Parameters
    ----------
    n_regimes : int
        Number of regime centroids to discover.
    window : int
        Rolling window size (business days) for building empirical
        distributions.
    max_iter : int
        Maximum K-Medoids iterations.
    """

    n_regimes: int = 4
    window: int = 20
    max_iter: int = 50
    _centroids: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    _distance_matrix: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    _labels: Optional[np.ndarray] = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wasserstein_1(u: np.ndarray, v: np.ndarray) -> float:
        """Compute W₁ (Earth Mover's Distance) between two 1-D samples."""
        if ot is None:
            # Fallback: scipy-based 1-D EMD
            from scipy.stats import wasserstein_distance

            return float(wasserstein_distance(u, v))

        n, m = len(u), len(v)
        a = np.ones(n) / n
        b = np.ones(m) / m
        M = ot.dist(u.reshape(-1, 1), v.reshape(-1, 1), metric="euclidean")
        return float(ot.emd2(a, b, M))

    def _build_windows(self, returns: np.ndarray) -> List[np.ndarray]:
        """Slice rolling windows from a (T, n_assets) return matrix."""
        T = returns.shape[0]
        windows: List[np.ndarray] = []
        for t in range(self.window, T + 1):
            windows.append(returns[t - self.window : t].flatten())
        return windows

    def _pairwise_distance_matrix(self, windows: List[np.ndarray]) -> np.ndarray:
        """Build symmetric distance matrix of W₁ distances."""
        n = len(windows)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._wasserstein_1(windows[i], windows[j])
                D[i, j] = d
                D[j, i] = d
        return D

    def _kmedoids(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple K-Medoids on a precomputed distance matrix.

        Returns
        -------
        labels : (n,) array of cluster assignments
        medoid_indices : (n_regimes,) array of medoid row indices
        """
        n = D.shape[0]
        rng = np.random.RandomState(42)
        medoids = rng.choice(n, size=self.n_regimes, replace=False)

        for _ in range(self.max_iter):
            # Assign each point to nearest medoid
            dists_to_medoids = D[:, medoids]  # (n, k)
            labels = np.argmin(dists_to_medoids, axis=1)

            # Update medoids
            new_medoids = np.copy(medoids)
            for k in range(self.n_regimes):
                members = np.where(labels == k)[0]
                if len(members) == 0:
                    continue
                sub_D = D[np.ix_(members, members)]
                costs = sub_D.sum(axis=1)
                new_medoids[k] = members[np.argmin(costs)]

            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        return labels, medoids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "WassersteinRegimeCluster":
        """Fit regime clusters on a (T, n_assets) return matrix.

        Parameters
        ----------
        returns : np.ndarray
            Shape ``(T, n_assets)`` of daily multi-asset returns.

        Returns
        -------
        self
        """
        windows = self._build_windows(returns)
        if len(windows) < self.n_regimes:
            raise ValueError(
                f"Not enough windows ({len(windows)}) for "
                f"{self.n_regimes} regimes. Need at least "
                f"{self.n_regimes + self.window - 1} rows."
            )

        self._distance_matrix = self._pairwise_distance_matrix(windows)
        self._labels, medoid_idx = self._kmedoids(self._distance_matrix)
        self._centroids = np.array([windows[i] for i in medoid_idx])
        logger.info("Fitted %d regimes on %d windows", self.n_regimes, len(windows))
        return self

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Return continuous probability vector for each window.

        Uses inverse-distance weighting to the fitted centroids so the
        output is a *continuous* soft membership — no discrete flags.

        Parameters
        ----------
        returns : np.ndarray
            Shape ``(T, n_assets)`` of daily returns.

        Returns
        -------
        probs : np.ndarray
            Shape ``(n_windows, n_regimes)`` probability matrix.
            Each row sums to 1.
        """
        if self._centroids is None:
            raise RuntimeError("Must call .fit() before .predict_proba()")

        windows = self._build_windows(returns)
        n = len(windows)
        dists = np.zeros((n, self.n_regimes))
        for i, w in enumerate(windows):
            for k in range(self.n_regimes):
                dists[i, k] = self._wasserstein_1(w, self._centroids[k])

        # Inverse-distance weighting → probabilities
        inv = 1.0 / (dists + 1e-10)
        probs = inv / inv.sum(axis=1, keepdims=True)
        return probs

    def compute_barycenters(
        self, weights: np.ndarray, n_support: int = 50
    ) -> np.ndarray:
        """Compute Wasserstein barycenter of the fitted centroids.

        Parameters
        ----------
        weights : np.ndarray
            Shape ``(n_regimes,)`` convex combination weights.
        n_support : int
            Number of support points for the barycenter.

        Returns
        -------
        barycenter : np.ndarray
            Shape ``(n_support,)`` barycenter distribution.
        """
        if self._centroids is None:
            raise RuntimeError("Must call .fit() before .compute_barycenters()")

        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()  # normalise

        if ot is not None:
            # Build list of uniform-weight distributions on each centroid
            distributions = []
            for c in self._centroids:
                distributions.append(np.ones(len(c)) / len(c))

            support = np.linspace(
                min(c.min() for c in self._centroids),
                max(c.max() for c in self._centroids),
                n_support,
            )

            # Cost matrices between support and each centroid's support
            A_list = []
            for c in self._centroids:
                M = ot.dist(
                    support.reshape(-1, 1), c.reshape(-1, 1), metric="euclidean"
                )
                A_list.append(M)

            bary = ot.barycenter(
                A=np.column_stack(distributions) if len(distributions) > 0 else np.empty((0, 0)),
                M=A_list[0],  # reference cost
                weights=weights,
                reg=1e-3,
            )
            return bary
        else:
            # Simple weighted average fallback
            result = np.zeros(n_support)
            for k, c in enumerate(self._centroids):
                interp = np.interp(
                    np.linspace(0, 1, n_support),
                    np.linspace(0, 1, len(c)),
                    np.sort(c),
                )
                result += weights[k] * interp
            return result
