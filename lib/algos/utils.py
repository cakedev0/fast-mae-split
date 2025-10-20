from numba import njit
import numpy as np

@njit
def rankdata(x: np.ndarray):
    rank = np.empty(x.size, dtype=np.int32)
    rank[np.argsort(x)] = np.arange(0, rank.size)
    return rank
