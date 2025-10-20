import numpy as np
from numba import njit

from ..ds.segment_tree import WeightedSegmentTree
from .utils import rankdata


@njit
def compute_left_loss_segmenttree(y, w, alpha=0.5, ranks=None) -> np.ndarray:
    if ranks is None:
        ranks = rankdata(y)
    st = WeightedSegmentTree(y.size)
    loss = np.empty(y.size)
    for i in range(y.size):
        st.set(ranks[i], w[i], y[i])
        total_weight = st.weights[0]
        weight_below, weighted_sum_below, q = st.search(total_weight * alpha)
        weight_above = total_weight - weight_below
        weighted_sum_above = st.weighted[0] - weighted_sum_below
        loss[i] = (
            alpha * (weighted_sum_above - q * weight_above)
            + (1 - alpha) * (q * weight_below - weighted_sum_below)
        )
    return loss
