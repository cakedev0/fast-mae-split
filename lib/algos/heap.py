import numpy as np
from numba import njit

from ..ds.heap import WeightedHeap
from ..ds.pythonheap import WeightedHeap as PythonWeightedHeap


def _compute_prefix_maes_heap(y: np.ndarray, w: np.ndarray, alpha: float, above: WeightedHeap, below: WeightedHeap):
    n = y.size
    maes = np.empty(n, dtype=y.dtype)
    for i in range(n):
        # Insert y[i] into the appropriate heap
        if above.empty() or below.empty() or y[i] > below.top():
            above.push(y[i], w[i])
        else:
            below.push(y[i], w[i])

        split_weight = (above.total_weight + below.total_weight) * (1 - alpha)
        # Rebalance the heaps
        while above.total_weight < split_weight:
            yt, wt = below.pop()
            above.push(yt, wt)
        while above.total_weight - above.top_weight() > split_weight:
            yt, wt = above.pop()
            below.push(yt, wt)

        q = above.top()  # Current weighted quantile
        # Compute the loss:
        maes[i] = (
            alpha * (above.weighted_sum - q * above.total_weight)
            + (1 - alpha) * (q * below.total_weight - below.weighted_sum)
        )
    return maes


def compute_prefix_loss_python_heap(y, w, alpha=0.5) -> np.ndarray:
    above = PythonWeightedHeap(min_heap=True)
    below = PythonWeightedHeap(min_heap=False)
    return _compute_prefix_maes_heap(y, w, alpha, above, below)


_compute_prefix_maes_heap_numba = njit(_compute_prefix_maes_heap)


@njit
def compute_prefix_loss_heap(y, w, alpha=0.5) -> np.ndarray:
    above = WeightedHeap(y.size ,True)
    below = WeightedHeap(y.size, False)
    return _compute_prefix_maes_heap_numba(y, w, alpha, above, below)
