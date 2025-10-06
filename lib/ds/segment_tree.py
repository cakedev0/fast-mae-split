import numpy as np
from numba import types as ntypes
from numba import njit
from numba.experimental import jitclass


@njit
def _build_pos(pos, i, l, s):
    """
    Recursively builds position mapping for segment tree nodes.

    Args:
        pos: Array to store leaf positions
        i: Current internal node index
        l: Left boundary in pos array
        s: Size of current segment
    """
    if s == 1:
        # Base case: single element, store the internal node index
        pos[l] = i
    else:
        # Split segment into left and right parts
        sl = (s + 1) >> 1  # Size of left segment (ceil division)
        sr = s >> 1        # Size of right segment
        i <<= 1            # Move to children level: i*2
        # Recursively build left and right subtrees
        _build_pos(pos, i + 1, l, sl)        # Left child: i*2+1
        _build_pos(pos, i + 2, l + sl, sr)   # Right child: i*2+2


@jitclass([
    ('size', ntypes.int32),      # Number of leaf elements
    ('pos', ntypes.int32[:]),    # Maps leaf index to internal node index
    ('weights', ntypes.float64[:]),   # Sum of weights in subtree
    ('weighted', ntypes.float64[:]),  # Sum of weight*value in subtree
    ('vals', ntypes.float64[:]),      # Values stored at nodes
    ('is_leaf', ntypes.bool_[:]),     # Boolean array marking leaf nodes
])
class WeightedSegmentTree:
    """
    A segment tree that maintains weighted sums for efficient range queries
    and weighted sampling. Each leaf has a weight and value, and internal
    nodes store aggregated weights and weighted values.
    """

    def __init__(self, size):
        """Initialize segment tree with given number of leaves."""
        self.size = size
        # Build mapping from leaf indices to internal node positions
        self.pos = np.empty(size, dtype=np.int32)
        _build_pos(self.pos, 0, 0, size)
        
        # Determine maximum internal node index
        m = self.pos.max() + 1

        # Initialize arrays for tree nodes
        self.weights = np.zeros(m)    # Sum of weights in subtree
        self.weighted = np.zeros(m)   # Sum of weight*value in subtree
        self.vals = np.empty(m)       # Values at nodes
        self.is_leaf = np.zeros(m, dtype=np.bool_)
        self.is_leaf[self.pos] = True  # Mark leaf positions

    def set(self, i, w, val):
        """
        Set weight and value for leaf i, updating all ancestors.
        
        Args:
            i: Leaf index
            w: Weight to add
            val: Value to store
        """
        i = self.pos[i]  # Get internal node index for leaf i
        self.vals[i] = val
        wv = w * val
        
        # Propagate changes up the tree
        while i >= 0:
            self.weights[i] += w      # Add weight to current node
            self.weighted[i] += wv    # Add weighted value to current node
            i = (i - 1) >> 1          # Move to parent: (i-1)/2

    def search(self, w):
        """
        Search for the leaf such that:
        - the cumulative weight up to this leaf excluded is <= w
        - the cumulative weight up to this leaf included is > w

        Returns:
            - cumulative_weight: the cumulative weight up to the leaf excluded
            - cumulative_weighted_value:
            - selected_value: the value at the leaf
        """
        i = 0        # Start at root
        cw = 0.      # Cumulative weight
        cwv = 0.     # Cumulative weighted value

        # Traverse down the tree
        while not self.is_leaf[i]:
            i <<= 1           # Move to children level: i*2
            il = i + 1        # Left child
            ir = i + 2        # Right child
            wl = self.weights[il]  # Weight in right subtree

            if cw + wl <= w:
                # Go to the right subtree,  account for left subtree weight
                i = ir
                cw += wl
                cwv += self.weighted[il]
            else:
                # Go to left subtree
                i = il

        return cw, cwv, self.vals[i]
