from heapq import heappush, heappop

class WeightedHeap:

    def __init__(self, min_heap=True):
        self.heap = []
        self.total_weight = 0
        self.weighted_sum = 0
        self.min_heap = min_heap

    def empty(self):
        return len(self.heap) == 0

    def push(self, val, weight):
        self.total_weight += weight
        self.weighted_sum += val * weight
        val = val if self.min_heap else -val
        heappush(self.heap, [(val, weight)])

    def top(self):
        val, _ = self.heap[0]
        return val if self.min_heap else -val

    def top_weight(self):
        _, w = self.heap[0]
        return w

    def pop(self):
        val, w = heappop(self.heap)
        val = val if self.min_heap else -val
        self.total_weight -= w
        self.weighted_sum -= val * w
        return val, w


import numpy

def min_mae_split(x, y, w):
    sorter = numpy.argsort(x)
    x = x[sorter]
    y = y[sorter]
    w = w[sorter]
    left_loss = compute_prefix_loss(y, w)
    right_loss = compute_prefix_loss(y[::-1], w[::-1])[::-1]
    total_loss = left_loss + right_loss
    total_loss[x[:-1] == x[1:]] = numpy.inf

    best_split = numpy.argmin(total_loss)
    x_split = (x[best_split] + x[best_split + 1]) / 2
    return x_split, total_loss[best_split]


def compute_prefix_loss(y: numpy.ndarray, w: numpy.ndarray):
    n = y.size
    above = WeightedHeap(n, True)  # Min-heap for values above the median
    below = WeightedHeap(n, False)  # Max-heap for values below the median
    maes = numpy.full(n-1, numpy.inf)
    for i in range(n - 1):
        # Insert y[i] into the appropriate heap:
        if above.empty() or y[i] > below.top():
            above.push(y[i], w[i])
        else:
            below.push(y[i], w[i])

        # Rebalance:
        half_weight = (above.total_weight + below.total_weight) / 2
        while above.total_weight < half_weight:
            above.push(*below.pop())
        while above.total_weight - above.top_weight() >= half_weight:
            below.push(*above.pop())

        # Compute AE for this split
        median = above.top()  # Current weighted median
        maes[i] = (
            (below.total_weight - above.total_weight) * median
            - below.weighted_sum
            + above.weighted_sum
        )
    return maes
