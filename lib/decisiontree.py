import numpy

from .heapbasedsplit import min_mae_split


def fit_apply(X, y, X_apply=None, sample_weights=None, algo="heaps", debug=False):
    if sample_weights is None:
        sample_weights = numpy.ones(y.size)
    X_apply = X if X_apply is None else X_apply
    best_mae, best_feature, best_threshold = numpy.inf, -1, numpy.nan
    for k, x in enumerate(X.T):
        if algo == "heaps":
            threshold, split_mae = min_mae_split(x, y, sample_weights, debug)
        else:
            raise NotImplementedError(algo)
        if split_mae < best_mae:
            best_mae = split_mae
            best_feature = k
            best_threshold = threshold
    if debug:
        print("split axis:", best_feature)
    return (X_apply[:, best_feature] >= best_threshold) + 1
