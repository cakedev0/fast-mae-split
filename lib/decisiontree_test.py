import numpy
from sklearn.tree import DecisionTreeRegressor

from .decisiontree import fit_apply


def sample_X_y_w(n):
    x_true = (numpy.random.rand(n) > 0.5).astype(float)
    X = numpy.array([
        numpy.random.randn(n) + 2*x_true,
        numpy.round(2*numpy.random.rand(n) + 2*x_true, 2)
    ]).T
    X_pred = numpy.array([
        numpy.random.randn(n) + 2*x_true,
        2*numpy.random.rand(n) + 2*x_true
    ]).T
    y = numpy.random.rand(n) + (numpy.random.rand(n) + 0.5) * x_true
    w = 0.5 + numpy.random.rand(n)
    return X, y, w, X_pred


def mae_min(y, w):
    return min((numpy.abs(y - yi) * w).sum() for yi in y)

def leaves_mae(l, y, w=None):
    if w is None:
        w = numpy.ones(y.size)
    return sum(mae_min(y[l == i], w[l == i]) for i in numpy.unique(l))


def test_against_sklearn_no_weights():
    for _ in range(1):
        for n in [5]*100 + [10]*100 + [100]*100 + [1000]*10 + [10_000]*3:
            X, y, _, _ = sample_X_y_w(n)
            reg = DecisionTreeRegressor(max_depth=1, criterion='absolute_error')
            sk_leaves = reg.fit(X, y).apply(X)
            h_leaves = fit_apply(X, y, X, algo='heaps')
            are_leaves_the_same = (sk_leaves == h_leaves).all() or (sk_leaves == (3 - h_leaves)).all()
            if not are_leaves_the_same:
                sk_mae = leaves_mae(sk_leaves, y)
                h_mae = leaves_mae(h_leaves, y)
                assert numpy.isclose(sk_mae, h_mae)

def test_against_sklearn_with_weights():
    print('with weights:')
    for n in [5]*100:# + [10]*100 + [100]*100 + [1000]*10 + [10_000]*3:
        X, y, w, _ = sample_X_y_w(n)
        reg = DecisionTreeRegressor(max_depth=1, criterion='absolute_error')
        sk_leaves = reg.fit(X, y, sample_weight=w).apply(X)
        h_leaves = fit_apply(X, y, X_apply=X, sample_weights=w, algo='heaps')
        are_leaves_the_same = (sk_leaves == h_leaves).all() or (sk_leaves == (3 - h_leaves)).all()
        if not are_leaves_the_same:
            sk_mae = leaves_mae(sk_leaves, y, w)
            h_mae = leaves_mae(h_leaves, y, w)
            if not numpy.isclose(sk_mae, h_mae):
                assert sk_mae > h_mae
                print(sk_mae, h_mae)
                print(X)
                print(y)
                print(w)
