import numpy
from numba import njit, types as ntypes
from numba.experimental import jitclass


@jitclass(
    [
        ("heap", ntypes.float64[:]),
        ("weights", ntypes.float64[:]),
        ("total_weight", ntypes.float64),
        ("weighted_sum", ntypes.float64),
        ("size", ntypes.uint32),
        ("min_heap", ntypes.boolean),
    ]
)
class WeightedHeap:

    def __init__(self, max_size, min_heap=True):
        self.heap = numpy.zeros(max_size, dtype=numpy.float64)
        self.weights = numpy.zeros(max_size, dtype=numpy.float64)
        self.total_weight = 0
        self.weighted_sum = 0
        self.size = 0
        self.min_heap = min_heap

    def empty(self):
        return self.size == 0

    def push(self, val, weight):
        self.heap[self.size] = val if self.min_heap else -val
        self.weights[self.size] = weight
        self.total_weight += weight
        self.weighted_sum += val * weight
        self.size += 1
        self._perc_up(self.size - 1)

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]

    def top(self):
        return self.heap[0] if self.min_heap else -self.heap[0]

    def top_weight(self):
        return self.weights[0]

    def pop(self):
        retv = self.top()
        retw = self.top_weight()
        self.size -= 1
        self.total_weight -= retw
        self.weighted_sum -= retv * retw
        self.heap[0] = self.heap[self.size]
        self.weights[0] = self.weights[self.size]
        self._perc_down(0)
        return retv, retw

    def _perc_up(self, i):
        p = (i - 1) >> 1
        while p >= 0:
            if self.heap[i] < self.heap[p]:
                self.swap(i, p)
            i = p
            p = (i - 1) >> 1

    def _perc_down(self, i):
        while (i << 1) + 2 <= self.size:
            mc_i = self._min_child_node(i)
            if self.heap[i] > self.heap[mc_i]:
                self.swap(i, mc_i)
            i = mc_i

    def _min_child_node(self, i):
        if (i << 1) + 2 == self.size:
            return (i << 1) | 1
        else:
            if self.heap[(i << 1) | 1] < self.heap[(i << 1) + 2]:
                return (i << 1) | 1
            else:
                return (i << 1) + 2
