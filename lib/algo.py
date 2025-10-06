import numpy as np
from numba import njit

from .ds.heap import WeightedHeap
from .ds.pythonheap import WeightedHeap as PythonWeightedHeap
from .ds.segmenttree import WeightedSegmentTree


def _compute_prefix_maes_heap(y: np.ndarray, w: np.ndarray, q: float, above: WeightedHeap, below: WeightedHeap):
    n = y.size
    maes = np.empty(n, dtype=y.dtype)
    for i in range(n):
        # Insert y[i] into the appropriate heap
        if above.empty() or below.empty() or y[i] > below.top():
            above.push(y[i], w[i])
        else:
            below.push(y[i], w[i])

        split_weight = (above.total_weight + below.total_weight) * (1 - q)
        # Rebalance the heaps
        while above.total_weight < split_weight:
            yt, wt = below.pop()
            above.push(yt, wt)
        while above.total_weight - above.top_weight() > split_weight:
            yt, wt = above.pop()
            below.push(yt, wt)

        quantile = above.top()  # Current weighted quantile
        # Compute the loss:
        maes[i] = (
            q * (above.weighted_sum - quantile * above.total_weight)
            + (1 - q) * (quantile * below.total_weight - below.weighted_sum)
        )
    return maes


def compute_prefix_loss_python_heap(y, w, q=0.5) -> np.ndarray:
    above = PythonWeightedHeap(min_heap=True)
    below = PythonWeightedHeap(min_heap=False)
    return _compute_prefix_maes_heap(y, w, q, above, below)


_compute_prefix_maes_heap_numba = njit(_compute_prefix_maes_heap)


@njit
def compute_prefix_loss_heap(y, w, q=0.5) -> np.ndarray:
    above = WeightedHeap(y.size ,True)
    below = WeightedHeap(y.size, False)
    return _compute_prefix_maes_heap_numba(y, w, q, above, below)


@njit
def rankdata(x):
    # TODO: is it slower with numba?
    rank = np.empty(x.size, dtype=np.int32)
    rank[np.argsort(x)] = np.arange(0, rank.size)
    return rank


@njit
def compute_prefix_loss_segmenttree(y, w, q=0.5, ranks=None, st=None) -> np.ndarray:
    if ranks is None:
        ranks = rankdata(y)
    if st is None:
        st = WeightedSegmentTree(y.size)
    maes = np.empty(y.size)
    for i in range(y.size):
        st.set(ranks[i], w[i], y[i])
        total_weight = st.weights[0]
        weight_left, weighted_sum_left, quantile = st.search(total_weight * q)
        weight_right = total_weight - weight_left
        weighted_sum_right = st.weighted[0] - weighted_sum_left
        maes[i] = (
            q * (weighted_sum_right - quantile * weight_right)
            + (1 - q) * (quantile * weight_left - weighted_sum_left)
        )
    return maes
