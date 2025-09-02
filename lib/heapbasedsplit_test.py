import numpy

from .heapbasedsplit import compute_prefix_maes


def min_mae_naive(y, w):
    return min((numpy.abs(y - yi) * w).sum() for yi in y)

def test_compute_prefix_maes():
    y = numpy.random.rand(100)
    w = numpy.random.rand(y.size)
    maes = compute_prefix_maes(y, w)
    for i in range(maes.size):
        assert numpy.isclose(maes[i], min_mae_naive(y[:i+1], w[:i+1]))

    # add some duplicates:
    y[y.size//2:] = y[:y.size//2]
    maes = compute_prefix_maes(y, w)
    for i in range(maes.size):
        assert numpy.isclose(maes[i], min_mae_naive(y[:i+1], w[:i+1]))

    # add even more duplicates:
    y = y.round(1)
    maes = compute_prefix_maes(y, w)
    for i in range(maes.size):
        assert numpy.isclose(maes[i], min_mae_naive(y[:i+1], w[:i+1]))
