**TL;DR**: Maintain the weighted $\alpha$-quantile with two heaps, update four aggregates, and evaluate pinball loss in $O(1)$ per threshold. You get $O(n\log n)$ expected time under mild conditions, with a clean $O(n\log n)$ worst-case alternative via a segment tree when needed.

# Efficient Weighted Quantile–Based Splitting for Decision Trees


This repository provides an efficient implementation of quantile-based impurities (pinball loss)
for decision-tree regression with **weighted samples** and **arbitrary quantile levels** $\alpha\in[0,1]$.

* **Two-heaps method (default):** Expected $O(n\log n)$ per feature under mild assumptions; very fast in practice.
  This was implemented in scikit-learn in PR [#32100](https://github.com/scikit-learn/scikit-learn/pull/32100)
  for the absolute error case ($\alpha = 0.5$) replacing an $O(n^2)$ implementation.
  A future PR will generalize to **arbitrary quantile levels**.
* **Segment tree alternative:** Worst-case $O(n\log n)$ per feature by design; typically ~3× slower than two-heaps in benchmarks, but provides a clean theoretical upper bound.

---

## 1. Pinball loss with weights and an $O(1)$ formula

For weighted data ${(y_i,w_i)}*{i=1}^n$ with $w_i > 0$, the pinball loss at level $\alpha\in[0,1]$ for a prediction $q\in\mathbb{R}$ is

$$
L*\alpha(q)
= \sum_{i} w_i!\left(\alpha\max(y_i-q,0) + (1-\alpha)\max(q-y_i,0)\right).
$$

Splitting by whether $y_i\ge q$ or $y_i < q$, define the aggregates
$$
W^+(q)= \sum_{y_i\ge q}! w_i,\quad
Y^+(q)= \sum_{y_i\ge q}! w_i y_i,\qquad
W^-(q)= \sum_{y_i< q}! w_i,\quad
Y^-(q)= \sum_{y_i< q}! w_i y_i.
$$

Then
$$
\boxed{L_\alpha(q)
= \alpha\big(Y^+(q) - q W^+(q)\big)
+ (1-\alpha)\big(q W^-(q)-Y^-(q)\big)}
$$
which is **$O(1)$** to evaluate once the four aggregates are maintained.

**Code (notation: `alpha` is the level, `q` is the value):**

```python
loss = alpha * (above.weighted_sum - q * above.total_weight) \
     + (1 - alpha) * (q * below.total_weight - below.weighted_sum)
# "above": indices with y_i >= q; "below": indices with y_i < q
```

$q$ is a **weighted $\alpha$-quantile** of the current set.

---

## 2. Algorithms

### 2.1 Two-heaps (weighted $\alpha$-quantile maintenance)

This algorithm is a weighted adaptation of the two-heaps solution of the median of a data-stream problem (see for instance this [leetcode solution](https://leetcode.com/problems/find-median-from-data-stream/solutions/7146165/o-logn-2-heaps-python))

Maintain two heaps keyed by $y$: a max-heap for items below $q$ and a min-heap for items at/above $q$. Balance by **total weight** (not count) so that
$$
W^- \approx \alpha W \quad \text{with } W^-= \sum_{y_i<q} w_i \, , \; W=W^- + W^+.
$$
During a left→right sweep over samples sorted by the candidate feature:

1. Insert $(y_i,w_i)$ into the appropriate heap. Complexity: $O(\log n)$
2. **Rebalance by weight** by moving boundary items across heaps until $W^- \approx \alpha W$. **Expected** complexity: $O(\log n)$, see section 3 for more details.
3. Read off $q$ at the boundary and compute the child loss with the $O(1)$ formula.

A symmetric right→left sweep yields the right-child losses; summing gives the impurity at each threshold.

**Implementation:**
- A simple python WeightedHeap (wraps stdlib's `heapq`)
- A numba WeightedHeap
- The left→right sweep loop:

### 2.2 Segment tree (guaranteed $O(n\log n)$)


**Idea.** Maintain a static binary tree over the **sorted order of targets $y$**. Each leaf corresponds to one **rank** position in that sorted order (smallest $y_i$ has rank $0$ and so on); every internal node stores two aggregates for its subtree:

* total weight $W$ and
* total weighted target $Y=\sum w_i y_i$.

The tree is initialized with all aggregates set to zero—that is, all leaves start “empty.” 

This lets us (i) **set** the current sample’s $(y_i, w_i)$ at its leaf (i.e. at the **rank** of $y_i$), updating $W$ and $Y$ up the path, and (ii) **search** by **cumulative weight** to find the weighted $\alpha$-quantile and the prefix aggregates needed for the $O(1)$ pinball-loss formula.

**Operations.**

* **SET(rank, $w$, $y$)**: add $(w,y)$ to the leaf at `rank`; bubble updates to the root, maintaining subtree aggregates $(W,Y)$. Cost $O(\log n)$.
* **SEARCH($t$)**: given a weight target $t=\alpha \cdot W_{\text{total}}$, descend from the root choosing left/right by comparing (t) to the left child’s weight. This finds the **leaf** where the weighted rank crosses (t); along the way you accumulate the **prefix** (W^-) and (Y^-). Return ((W^-, Y^-, q)) where (q) is the leaf’s (y)-value (the current weighted (\alpha)-quantile). Cost (O(\log n)).

(A symmetric right→left sweep yields right-child losses; add left+right to score each threshold.)

**Complexity and performance.**

* Each step performs one `SET` and one `SEARCH`: **$O(\log n)$** each ⇒ **$O(n\log n)$** per sweep, worst-case by design.
* In practice this variant is typically **~3× slower** than the two-heaps method on the same data (larger constants, more memory traffic), but it provides a clean upper bound and deterministic per-step work.

**Implementation.**

* The weighted segment tree class:
* The left→right sweep loop:

---

## 3. Complexity of the two-heaps algorithm

See the notebook [complexity_experiments](https://github.com/cakedev0/fast-mae-split/blob/main/complexity_experiments.ipynb) for experiments and plots illustrating the different statements of this section.

### 3.1 Worst-case for two-heaps

In adversarial orders of weights, a single insertion can force the $\alpha$-quantile boundary to traverse $\Theta(n)$ items (each move is a heap pop+push, $O(\log n)$). Thus **worst-case** per insertion is $O(n\log n)$ and a full sweep can be **$O(n^2\log n)$**.

This does **not** occur in typical data; it requires systematically placing extreme weights to repeatedly push the boundary across many tiny items.


### 3.2 Expected $O(n\log n)$ per feature — two versions

We give two simple sets of conditions under which the **expected** number of boundary moves per insertion is **$O(1)$**, yielding $O(n\log n)$ total expected time (insert + moves, each $O(\log n)$).

---

#### 3.2A Independence model: $Y \perp W$

**Assumptions.**

* $Y_1, \dots, Y_n\;$ i.i.d. from a continuous distribution (or consistent tie-breaking).
* $W_1, \dots, W_n\;$ i.i.d., non-negative, independent of the $Y_i$, with $0<\mu=\mathbb{E}[W_i]<\infty$.

**Intuition.** Insert $(Y_{t+1},W_{t+1})$. The target balance changes by at most the new weight, so the boundary must “absorb” weight
$$
B_t\in{\alpha W_{t+1}, (1-\alpha)W_{t+1}}\le W_{t+1}.
$$

To re-balance, we move boundary items one by one. By independence and continuity, the weights encountered at the boundary look like fresh draws from $W$, so a **typical boundary item contributes $\approx \mu$ weight**. Hence the expected number of moves is about $\mathbb{E}[B_t]/\mu\le \mu/\mu=O(1)$ (up to a constant for the final overshoot). Each move costs $O(\log n)$; adding the $O(\log n)$ insertion yields $O(\log n)$ expected time per sample and $O(n\log n)$ per feature.

**Remark.** Heavy tails with finite mean (e.g., log-normal, Pareto with $\alpha>1$) increase variance but not the expectation; infinite-mean tails would break the bound. However, in practice, even with infinite-mean Pareto distributions, performance does not degrade significantly.

---

#### 3.2B Lower-bound model: $f(y)=\mathbb{E}[W\mid Y=y]\ge c>0$

**Assumptions.**

* $(Y_i,W_i)$ are i.i.d.; $\mu=\mathbb{E}[W_i]<\infty$.
* $Y$ has a continuous distribution (or consistent tie-breaking).
* There exists $c>0$ such that for all $y$ in the support of $Y$,
  $$
  f(y):=\mathbb{E}[W\mid Y=y]\ \ge\ c.
  $$

**Intuition.** As above, each insertion requires shifting at most $B_t\le W_{t+1}$ weight across the boundary. The items encountered while sliding the boundary have an **expected weight of at least $c$**. Therefore, the **expected number of moves** satisfies
$$
\mathbb{E}[K_t]\ \lesssim\ \frac{\mathbb{E}[B_t]}{c} + 1 \ \le\ \frac{\mu}{c}+1 \ =\ O(1).
$$

Again, with $O(\log n)$ per move and insertion, we obtain $O(n\log n)$ expected time, but with possibly a much bigger constant than above, if $\frac{\mu}{c}$ is big. See graph *small_weights_around_median* in `complexity_experiments.ipynb` for an example of such a setting.

---

## 4. Empirical performance (illustrative)

* **Setup:** $n=100{,}000$
* **Previous $O(n^2)$ MAE path:** $\sim 10$ s.
* **Two-heaps:** $\sim 0.2$ s.
* **Segment tree:** typically ~3× slower than two-heaps on the same data, expect for adversial input,
  or low mean around quantile case: $\mathbb{E}[W|y \approx q] << \mathbb{E}[W]$.

---

## 5. Related work (brief)

This approach adapts the classic two-heaps streaming median to **weighted $\alpha$-quantiles** and couples it with a simple $O(1)$ pinball-loss evaluation. Some [prior work](https://faculty.ucmerced.edu/hbhat/BhatKumarVaz_final.pdf) also proposes a two-heaps algorithm for quantile-regression trees, but it omits weights and rely on more elaborate derivations for pinball-loss evaluation ($O(1)$ too, but might not be easy to adapt for the weighted case). The present formulation is simple and easy to integrate into open-source libraries such as scikit-learn.
