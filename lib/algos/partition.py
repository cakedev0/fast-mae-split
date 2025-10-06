import numpy as np


def weighted_quantile_and_pinball_loss(y: np.ndarray, w: np.ndarray, alpha: float):
    """
    O(n) weighted quantile calculation and aggregates maintenance
    using recursive partitioning
    + O(1) loss calculation
    """
    total_weight = w.sum()
    target_sum = total_weight * alpha
    weight_below = 0
    weighted_sum_below = 0 
    while y.size > 1:
        i = y.size // 2
        partitioner = np.argpartition(y, i)
        w_left = w[partitioner[:i]]
        sum_left = w_left.sum()
        if target_sum >= sum_left:
            target_sum -= sum_left
            y = y[partitioner[i:]]
            w = w[partitioner[i:]]
            weight_below += sum_left
            weighted_sum_below += (y * w).sum()
        else:
            y = y[partitioner[:i]]
            w = w_left
    q = y[0]
    weight_above = total_weight - weight_below
    weighted_sum_above = (y * w).sum() - weighted_sum_below
    return q, (
        alpha * (weighted_sum_above - q * weight_above)
        + (1 - alpha) * (q * weight_below - weighted_sum_below)
    )


def find_best_split_partition(x, y, w, alpha=0.5):
    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]
    w = w[sorter]

    splits_idx, = np.where(x[:-1] != x[:1])
    splits_idx += 1
    if splits_idx.size == 0:
        raise ValueError("constant x")

    splits = []
    for i in splits_idx:
        x_split = (x[i - 1] + x[i]) / 2
        q_left, loss_left = weighted_quantile_and_pinball_loss(y[:i], w[:i], alpha)
        q_right, loss_right = weighted_quantile_and_pinball_loss(y[i:], w[i:], alpha)
        splits.append((loss_left + loss_right, x_split, (q_left, q_right)))

    return min(splits)
