import numpy as np

from .heap import compute_prefix_loss_heap
from .fenwick import compute_left_loss_fenwick
from .segment_tree import compute_left_loss_segmenttree
from .utils import rankdata


METHODS = {
    "heap": compute_prefix_loss_heap,
    "segment-tree": compute_left_loss_segmenttree,
    "fenwick": compute_left_loss_fenwick,
}


def find_best_split(x, y, w, alpha=0.5, method="heap"):
    """
    Returns
    -------
    x_split : float
        The value of x at which to split.
    split_loss : float
        The loss of the split (sum of left node loss and right node loss)
    """
    func = METHODS[method]

    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]
    w = w[sorter]

    kwargs = {}
    if method != "heap":
        kwargs['ranks'] = rankdata(y)
    left_loss = func(y, w, alpha, **kwargs)
    if method != "heap":
        kwargs['ranks'] = kwargs['ranks'][::-1]
    right_loss = func(y[::-1], w[::-1], alpha, **kwargs)[::-1]
    loss = left_loss[:-1] + right_loss[1:]
    # impossible to split between 2 points that are exactly equals:
    loss[x[:-1] == x[1:]] = np.inf

    best_split = np.argmin(loss)
    x_split = (x[best_split] + x[best_split + 1]) / 2
    split_loss = loss[best_split]
    return x_split, split_loss
