import pytest
import numpy as np

from sklearn.metrics import mean_pinball_loss

from .algo import (
    compute_prefix_loss_python_heap,
    compute_prefix_loss_heap,
    compute_left_loss_segmenttree,
)

@pytest.mark.parametrize("q", [0.5, 0.2, 0.9, 0.4, 0.75])
@pytest.mark.parametrize("compute_prefix_loss", [
    compute_prefix_loss_python_heap,
    compute_prefix_loss_heap,
    compute_left_loss_segmenttree,
])
def test_pinball_loss_precomputation_function(q, compute_prefix_loss):
    """
    Test the main bit of logic of the MAE(RegressionCriterion) class
    (used by DecisionTreeRegressor(criterion="absolute_error")).

    The implementation of the criterion relies on an efficient precomputation
    of left/right children absolute error for each split. This test verifies this
    part of the computation, in case of major refactor of the MAE class,
    it can be safely removed.
    """

    def compute_prefix_losses_naive(y, w):
        """
        Computes the pinball loss for all (y[:i], w[:i])
        Naive: O(n^2 log n)
        """
        quantiles = [
            np.quantile(y[:i], q, weights=w[:i], method="inverted_cdf")
            for i in range(1, y.size + 1)
        ]
        losses = [
            mean_pinball_loss(y[:i], np.full(i, quantile), sample_weight=w[:i], alpha=q)
            * w[:i].sum()
            for i, quantile in zip(range(1, y.size + 1), quantiles)
        ]
        return np.array(losses), np.array(quantiles)

    def assert_same_results(y, w):
        losses = compute_prefix_loss(y, w, q)
        losses_, _ = compute_prefix_losses_naive(y, w)
        assert np.allclose(losses, losses_, atol=1e-12)

    rng = np.random.default_rng()

    ns = np.concat((
        np.repeat([3, 5, 10, 20, 50, 100], 3),
        [300]
    ))
    for n in ns:
        y = rng.random(n)
        w = rng.random(n)
        w *= 10.0 ** rng.uniform(-5, 5)
        assert_same_results(y, w)
        assert_same_results(y, np.ones(n))
        assert_same_results(y, w.round() + 1)
