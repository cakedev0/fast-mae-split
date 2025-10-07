import numpy as np
from numba import types as ntypes
from numba.experimental import jitclass


@jitclass([
    ('n', ntypes.int32),            # number of leaves (ranks)
    ('tree_w', ntypes.float64[:]),  # BIT for weights
    ('tree_wy', ntypes.float64[:]), # BIT for weighted targets
    ('vals', ntypes.float64[:]),    # y-value at each rank (1-based)
    ('max_pow2', ntypes.int32),     # highest power of two <= n
    ('total_w', ntypes.float64),    # running total weight
    ('total_wy', ntypes.float64),   # running total weighted target
])
class WeightedFenwick:
    """
    Fenwick tree (Binary Indexed Tree) for maintaining:
      - prefix sums of weights, and
      - prefix sums of weight*value (targets),
    indexed by the rank of y in sorted order (1-based internally).

    Supports:
      - add(rank, w, y): point update at 'rank'
      - search(t): find the smallest rank with cumulative weight > t,
                   also returns prefix aggregates excluding that rank.
    """

    def __init__(self, size):
        self.n = size
        # 1-based arrays of length n+1
        self.tree_w  = np.zeros(self.n + 1)
        self.tree_wy = np.zeros(self.n + 1)
        self.vals    = np.empty(self.n + 1)

        # highest power of two <= n
        p = 1
        while p <= self.n:
            p <<= 1
        self.max_pow2 = p >> 1

        self.total_w = 0.0
        self.total_wy = 0.0

    def add(self, rank0, w, y):
        """
        Add a sample with weight w and value y at 0-based rank 'rank0'.
        """
        i = rank0 + 1  # 1-based
        self.vals[i] = y
        wy = w * y

        j = i
        while j <= self.n:
            self.tree_w[j]  += w
            self.tree_wy[j] += wy
            j += j & -j

        self.total_w  += w
        self.total_wy += wy

    def search(self, t):
        """
        Find the leaf (rank) such that:
          prefix_weight(rank-1) <= t < prefix_weight(rank)
        and return:
          - cw: prefix weight up to (rank-1)
          - cwv: prefix weighted sum up to (rank-1)
          - q: the y-value at 'rank' (the weighted alpha-quantile)

        Notes:
          * Assumes there is at least one active (positive-weight) item.
          * If t >= total weight (can happen with alpha ~ 1), we clamp t slightly.
        """
        idx = 0
        cw = 0.0
        cwv = 0.0
        bit = self.max_pow2

        # Standard Fenwick lower-bound with simultaneous prefix accumulation
        while bit != 0:
            nxt = idx + bit
            if nxt <= self.n:
                w_here = self.tree_w[nxt]
                if t >= w_here:
                    t -= w_here
                    idx = nxt
                    cw  += w_here
                    cwv += self.tree_wy[nxt]
            bit >>= 1

        rank1 = idx + 1          # 1-based rank
        q = self.vals[rank1]     # y at the quantile position
        return cw, cwv, q
