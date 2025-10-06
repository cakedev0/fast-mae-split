<!--
🙌 Thanks for contributing a pull request!

👀 Please ensure you have taken a look at the contribution guidelines:
https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md

✅ In particular following the pull request checklist will increase the likelihood
of having maintainers review your PR:
https://scikit-learn.org/dev/developers/contributing.html#pull-request-checklist

📋 If your PR is likely to affect users, you will need to add a changelog entry
describing your PR changes, see:
https://github.com/scikit-learn/scikit-learn/blob/main/doc/whats_new/upcoming_changes/README.md
-->

This PR re-implements the way `DecisionTreeRegressor(criterion='absolute_error')` works underneath for optimization purposes. The current algorithm for calculating the AE of a split incures a O(n^2) overall complexity for building a tree which quickly becomes impractical. My implementation makes it O(n log n) making it tremendously faster.

For instance with d=2, n=100_000 and max_depth=1 (just one split), the execution time went from ~30s to ~100ms on my machine.

#### Referenced Issues

Fixes #9626 by reducing the complexity from O(n^2) to O(n log n).
Also fixes #32099. But that's more of the side effect of re-implementing completely the criterion logic.

#### Explanation of my changes

The work focuses solely on the class `MAE(RegressionCriterion)`.

Previous implementation had O(n^2) overall complexity emerging from several methods in this class
- in `update`: O(n) cost due to updating a data structure that maintains data sorted (`WeightedMedianCalculator`/`WeightedPQueue`). Called O(n) times to find the best split => incures O(n^2) overall
- in `children_impurity`: O(n) due to looping over all the data points. Called O(n) times to find the best split => incures O(n^2) overall

Those can't really be fix by small local changes, as overall, the algorithm is O(n^2) independently of how you implement it. Hence a complete rewrite was needed. I think there are several efficient algorithms to solve the problem (computing the absolute errors for all the possible splits along one feature). The one I chose is an adaptation of the well-known two-heap solution of the "find median from a data stream" problem (see for instance this [leetcode solution](https://leetcode.com/problems/find-median-from-data-stream/solutions/7146165/o-logn-2-heaps-python)).

The adaptations to make this work with weighted data (and to compute the AE along the way) are the following:
- instead of balancing the heaps based on their number of elements, rebalance them to keep one with just slightly more than half the total weight and the other with slighlty less.
- rewrite the AE computation by taking advantage of the following calculations:
    $$
    \sum_i w_i | y_i - m | = \sum_{y_i >= m} w_i(y_i - m) + \sum_{y_i < m} w_i(m - y_i) 
    = \sum_{y_i >= m} w_i y_i - m \sum_{y_i >= m} w_i + m \sum_{y_i < m} w_i - \sum_{y_i < m} w_i y_i 
    $$ 
    maintaining the value of those 4 sums as attributes of the heaps is easy, so the computation becomes O(1).

By applying the algorithm on iteration on the data from left to right, you are able to compute the AE for every possible left child.
And by applying it from right to left, you are able to compute the AE for every possible right child.

This logic is implemented in `tree/_utils.pyx::precompute_absolute_errors` as I wanted to be able to unit test it.

