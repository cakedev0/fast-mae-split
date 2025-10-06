import numpy as np

from .heap import compute_prefix_loss_heap
from .segment_tree import compute_left_loss_segmenttree


def find_best_split(x, y, w, alpha=0.5, method="heap"):
    """
    Returns
    -------
    x_split : float
        The value of x at which to split.
    split_loss : float
        The loss of the split (sum of left node loss and right node loss)
    """
    assert method in ["heap", "segment-tree"]
    func = (
        compute_prefix_loss_heap if method == "heap"
        else compute_left_loss_segmenttree
    )

    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]
    w = w[sorter]
    
    left_loss = func(y, w, alpha)
    right_loss = func(y[::-1], w[::-1], alpha)[::-1]
    loss = left_loss + right_loss
    # impossible to split between 2 points that are exactly equals:
    loss[x[:-1] == x[1:]] = np.inf

    best_split = np.argmin(loss)
    x_split = (x[best_split] + x[best_split + 1]) / 2
    split_loss = loss[best_split]
    return x_split, split_loss
