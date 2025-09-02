import numpy
from numba import njit

from .weightedheap import WeightedHeap


def mae_min(y, w):
    return min((numpy.abs(y - yi) * w).sum() for yi in y)

def leaves_mae(l, y, w):
    return sum(mae_min(y[l == i], w[l == i]) for i in numpy.unique(l))


def min_mae_split(x, y, w, debug=False):
    """
    Find the best split of x that minimizes the sum of left and right MAEs.

    Sorts and deduplicates x, y, w, then computes the MAE for all possible splits using splits_left_mae.
    Returns the split value and the corresponding MAE.

    Parameters
    ----------
    x : np.ndarray
        Feature values.
    y : np.ndarray
        Target values.
    w : np.ndarray
        Sample weights.

    Returns
    -------
    x_split : float
        The value of x at which to split.
    split_mae : float
        The minimum sum of left and right MAEs.
    """
    sorter = numpy.argsort(x)
    x = x[sorter]
    y = y[sorter]
    w = w[sorter]
    prefix_maes = compute_prefix_maes(y, w)
    suffix_maes = compute_prefix_maes(y[::-1], w[::-1])[::-1]
    maes = prefix_maes + suffix_maes  # size: n-1
    maes[x[:-1] == x[1:]] = numpy.inf  # impossible to split between 2 points that are exactly equals
    if debug:
        print(maes.round(3))
        # print(numpy.array([mae_min(y[:split], w[:split]) + mae_min(y[split:], w[split:]) for split in range(1, y.size)]).round(3))
    best_split = numpy.argmin(maes)
    # Choose split point between best_split and its neighbor with lower MAE
    x_split = (x[best_split] + x[best_split + 1]) / 2
    split_mae = maes[best_split]
    if debug:
        print(x)
        print(split_mae, leaves_mae(x < x_split, y, w))
        print(best_split, (x < x_split).astype(int))
    return x_split, split_mae


@njit
def compute_prefix_maes(y: numpy.ndarray, w: numpy.ndarray):
    """
    Compute the minimum mean absolute error (MAE) for all (y[:i], w[:i]) with i ranging in [1, n-1]
    O(n log n) complexity, expect for patological cases (w growing faster than x^2)

    Parameters
    ----------
    y : numpy.ndarray
        Array of target values (assumed sorted).
    w : numpy.ndarray
        Array of sample weights.
    Returns
    -------
    maes : numpy.ndarray
        Prefix array of MAE values
    """
    n = y.size
    above = WeightedHeap(n, True)  # Min-heap for values above the median
    below = WeightedHeap(n, False)  # Max-heap for values below the median
    maes = numpy.full(n-1, numpy.inf)
    for i in range(n - 1):
        # Insert y[i] into the appropriate heap
        if y[i] > below.top():
            above.push(y[i], w[i])
        else:
            below.push(y[i], w[i])

        half_weight = (above.total_weight + below.total_weight) / 2
        # Rebalance the heaps, we want to ensure that:
        # above.total_weight >= 1/2 and above.total_weight - above.top_weight() <= 1/2
        # which ensures that above.top() is a weighted median of the heap
        # and in particular, an argmin for the MAE
        while above.total_weight < half_weight:
            yt, wt = below.pop()
            above.push(yt, wt)
        while above.total_weight - above.top_weight() > half_weight:
            yt, wt = above.pop()
            below.push(yt, wt)

        median = above.top()  # Current weighted median
        # Compute MAE for this split
        maes[i] = (
            (below.total_weight - above.total_weight) * median
            - below.weighted_sum
            + above.weighted_sum
        )
    return maes
