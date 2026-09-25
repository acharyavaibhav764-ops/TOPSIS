"""
weighting_bwm.py — V2: Best-Worst Method (BWM) via linear program.

Concept
-------
Instead of the n(n-1)/2 = 28 pairwise comparisons AHP demands for 8 criteria
(which causes cognitive fatigue and logical contradictions), BWM asks the
decision-maker only two things:

  1. Which criterion is the BEST (most important)?  → call it B
  2. Which criterion is the WORST (least important)? → call it W
  3. Rate every criterion vs. B on a 1-9 scale   → vector a_B  (a_B[B] = 1)
  4. Rate every criterion vs. W on a 1-9 scale   → vector a_W  (a_W[W] = 1)

That is only 2(n-1) - 1 = 13 judgments for 8 criteria.

The optimization problem: find weight vector w (non-negative, summing to 1)
that minimizes the maximum deviation ξ between stated ratios and implied ratios:

    minimize  ξ
    subject to:
        |w_B / w_j  - a_B[j]| ≤ ξ   for all j ≠ B
        |w_j / w_W  - a_W[j]| ≤ ξ   for all j ≠ W
        Σ w_j = 1,  w_j ≥ 0

ξ (the output consistency ratio) is your data-quality signal.
  ξ ≈ 0   → perfectly consistent judgments
  ξ large → contradictory survey; ask the decision-maker to redo it.
"""

import numpy as np
from scipy.optimize import linprog


def bwm_weights(
    criteria: list[str],
    best_idx: int,
    worst_idx: int,
    a_B: np.ndarray,
    a_W: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Solve the BWM linear program and return the weight vector and consistency ratio.

    Parameters
    ----------
    criteria  : list of criterion codes, e.g. ["C1", ..., "C8"]
    best_idx  : index into `criteria` of the Best (most important) criterion
    worst_idx : index into `criteria` of the Worst (least important) criterion
    a_B       : 1-D array length n; a_B[j] = "how many times more important
                is Best than criterion j?" — must have a_B[best_idx] = 1
    a_W       : 1-D array length n; a_W[j] = "how many times more important
                is criterion j than Worst?" — must have a_W[worst_idx] = 1

    Returns
    -------
    weights : np.ndarray shape (n,) — normalized weight vector summing to 1
    xi      : float — consistency ratio (lower is better; 0 = perfect)
    """
    n = len(criteria)
    # Variables: w[0..n-1], xi = x[n]
    num_vars = n + 1
    c = np.zeros(num_vars)
    c[-1] = 1.0  # minimize xi

    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # |w_B - a_B[j]*w_j| ≤ ξ   →   two linear inequalities per j ≠ B
    for j in range(n):
        if j == best_idx:
            continue
        # w_B - a_B[j]*w_j - ξ ≤ 0
        row = np.zeros(num_vars)
        row[best_idx] = 1.0
        row[j] = -a_B[j]
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # -(w_B - a_B[j]*w_j) - ξ ≤ 0
        row2 = np.zeros(num_vars)
        row2[best_idx] = -1.0
        row2[j] = a_B[j]
        row2[-1] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)

    # |w_j - a_W[j]*w_W| ≤ ξ   →   two linear inequalities per j ≠ W
    for j in range(n):
        if j == worst_idx:
            continue
        # w_j - a_W[j]*w_W - ξ ≤ 0
        row = np.zeros(num_vars)
        row[j] = 1.0
        row[worst_idx] = -a_W[j]
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)
        # -(w_j - a_W[j]*w_W) - ξ ≤ 0
        row2 = np.zeros(num_vars)
        row2[j] = -1.0
        row2[worst_idx] = a_W[j]
        row2[-1] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)

    # Σ w_j = 1
    A_eq = [np.concatenate([np.ones(n), [0.0]])]
    b_eq = [1.0]

    bounds = [(0.0, None)] * num_vars

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"BWM LP did not converge: {res.message}")

    weights = res.x[:n]
    xi = float(res.x[-1])
    return weights / weights.sum(), xi  # re-normalize for floating-point safety
