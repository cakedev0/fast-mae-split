import numpy as np
from numba import njit

from ..ds.fenwick import WeightedFenwick


@njit
def rankdata(x):
    """
    0-based rank of each element in x
    """
    rank = np.empty(x.size, dtype=np.int32)
    rank[np.argsort(x)] = np.arange(0, rank.size, dtype=np.int32)
    return rank


@njit
def compute_left_loss_fenwick(y, w, q=0.5, ranks=None) -> np.ndarray:
    """
    Compute the left-child pinball loss after each activation (left->right sweep),
    using a Fenwick tree. 'q' is the quantile level alpha in [0,1].

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Targets.
    w : np.ndarray, shape (n,)
        Nonnegative sample weights.
    q : float
        Quantile level alpha.

    Returns
    -------
    loss : np.ndarray, shape (n,)
        Left-child loss after activating the first i+1 samples (in sweep order).
    """
    # Map each sample to its rank in y-sorted order (0-based)
    if ranks is None:
        ranks = rankdata(y)

    ft = WeightedFenwick(y.size)
    loss = np.empty_like(y)

    for i in range(y.size):
        # Activate sample i at its y-rank
        ft.add(ranks[i], w[i], y[i])

        # Weighted alpha-quantile by cumulative weight
        t = q * ft.total_w
        w_left, wy_left, quantile = ft.search(t)

        # Right-side aggregates include the quantile position
        w_right  = ft.total_w  - w_left
        wy_right = ft.total_wy - wy_left

        # O(1) pinball loss formula
        loss[i] = (
            q * (wy_right - quantile * w_right)
            + (1 - q) * (quantile * w_left - wy_left)
        )

    return loss
