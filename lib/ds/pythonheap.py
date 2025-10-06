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
        heappush(self.heap, (val, weight))

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
