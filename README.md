This project provides several **fast $O(n\log n)$ implementations** of **weighted quantile-based splits** (pinball loss) for decision tree regression.

It replaces the previous **$O(n^2)$** absolute-error path in scikit-learn (see PR [#32100](https://github.com/scikit-learn/scikit-learn/pull/32100)), reducing training time by **hundreds of times** on large datasets.

* **Two-heaps algorithm:** expected $O(n\log n)$, fast and simple to grasp.
* **Segment tree / Fenwick tree variants:** guaranteed $O(n\log n)$ worst case; slower but theoretically clean.
* Supports **arbitrary quantiles** ($\alpha\in[0,1]$) and **weighted samples**.
* Implemented in pure Python + Numba, easy to integrate or adapt. See scikit-learn PR for a Cython implementation of the two-heaps algorithm.

**Full technical report:** [report.ipynb](https://github.com/cakedev0/fast-mae-split/blob/main/report.ipynb)
