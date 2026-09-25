"""
weighting_critic_m.py — V2: CRITIC-M objective weights.

Concept
-------
Classic CRITIC (Diakoulaki et al., 1995) derives weights from the decision
matrix itself — no stakeholder survey needed.  The underlying logic:

  "A criterion earns high weight when it
   (a) actually discriminates between markets (high variance), AND
   (b) is not just echoing another criterion (low correlation with peers)."

Information content per criterion j:
    C_j = σ_j × Σ_k (1 - r_{jk})

where σ_j is the standard deviation of column j after normalization, and r_{jk}
is the Pearson correlation between columns j and k.

The final weight: W_j = C_j / Σ_k C_k

The problem with the original CRITIC is that it uses min-max normalization,
which forces the best and worst-observed jurisdiction to exactly 0 and 1.  One
extreme outlier stretches the whole scale, artificially inflating σ and hence
that criterion's weight — for the wrong reason.

CRITIC-M's fix: normalize around a robust center using logistic squash over
median/IQR instead of min-max.  A single outlier jurisdiction no longer
dominates the variance calculation.

Direction contract
------------------
This function receives a decision matrix that has ALREADY been direction-
corrected by build_decision_matrix() / criterion_mapping.py.  In that matrix,
a HIGHER score always means "more desirable" for every column, including the
three cost criteria (C6 capital burden, C7 presence friction, C8 volatility) —
their mapping functions already invert the raw values so that
  low capital → high score, low friction → high score, low volatility → high score.

Therefore this function must NOT re-apply benefit_mask negation.  Doing so
would perform a second inversion: a market with genuinely low capital burden
(score=8, good) would become -8 (treated as bad), which is wrong.

The benefit_mask parameter is kept in the signature for documentation and
so callers can see what the criteria's RAW semantics are, but it is not
used for any computation here.

Zero-variance guard (§14.1 of implementation plan)
---------------------------------------------------
When a criterion column has zero variance after the logistic squash (e.g. all
jurisdictions score identically because the criterion is a constant or was
median-imputed), np.corrcoef divides by that column's std and produces NaN
for the *entire* correlation matrix — poisoning ALL weights, not just that
criterion's weight.

The fix: identify zero-variance columns *after* the squash, exclude them from
the corrcoef call entirely, and assign them zero information content.
Do NOT try to jitter them — jitter on the raw column is overwritten by the
logistic squash, and jitter on the post-squash column still leaves std ≈ 0,
which corrcoef can't tolerate.
"""

import numpy as np
import pandas as pd


def critic_m_weights(
    matrix: pd.DataFrame,
    benefit_mask: np.ndarray,      # kept for documentation; NOT used for direction flip
) -> np.ndarray:
    """
    Compute objective CRITIC-M weights from the decision matrix.

    Parameters
    ----------
    matrix       : pd.DataFrame (jurisdictions × criteria).
                   IMPORTANT: values must already be direction-corrected so that
                   higher score = more desirable for ALL columns.
                   build_decision_matrix() guarantees this.
    benefit_mask : bool array, length n_criteria.
                   Documents the raw-data semantics (True=benefit, False=cost).
                   NOT used for internal direction correction — the matrix is
                   assumed to be already corrected upstream.

    Returns
    -------
    np.ndarray shape (n_criteria,) summing to 1.0
    """
    X = matrix.values.astype(float).copy()
    n_cols = X.shape[1]

    # Step 1: Logistic squash — robust normalization around median/IQR.
    # All columns are treated identically (no direction flip) because the
    # decision matrix is already fully direction-corrected.
    for j in range(n_cols):
        col = X[:, j]                           # higher already = better for every column
        med = np.median(col)
        iqr = float(np.subtract(*np.percentile(col, [75, 25])))
        if iqr < 1e-9:
            # Zero-IQR: all jurisdictions score identically on this criterion.
            # Logistic squash with IQR≈0 would overflow or produce a constant 0.5.
            # Force the constant explicitly — its std will be 0 and it gets zero weight.
            X[:, j] = 0.5
        else:
            # Logistic squash: robust to outliers, bounded in (0,1)
            X[:, j] = 1.0 / (1.0 + np.exp(-(col - med) / iqr))

    # Step 2: information content = std × conflict (sum of 1-corr with every other crit)
    std = X.std(axis=0, ddof=1)

    # Identify active (non-zero-variance) columns.
    # np.corrcoef divides by std internally. If ANY column std == 0, the entire
    # correlation matrix becomes NaN, poisoning ALL weights — not just that column's.
    # Fix (plan §14.1): exclude zero-variance columns from corrcoef entirely;
    # assign them zero information content (correct: they add no discriminating signal).
    active = np.where(std > 1e-10)[0]
    information = np.zeros(n_cols)

    if len(active) >= 2:
        X_active = X[:, active]
        std_active = std[active]
        corr_active = np.corrcoef(X_active, rowvar=False)
        # Guard residual NaN (shouldn't occur after active filter, but be defensive)
        corr_active = np.nan_to_num(corr_active, nan=0.0)
        conflict_active = (1.0 - corr_active).sum(axis=0)
        information[active] = std_active * conflict_active
    elif len(active) == 1:
        # Only one criterion has any variance — it gets all the weight
        information[active[0]] = std[active[0]]
    # else: all zero-variance → information stays zeros → raises below

    if information.sum() == 0:
        raise ValueError(
            "CRITIC-M: all criteria have zero information content — "
            "check that your matrix has more than one row and that "
            "not all criteria are completely constant across jurisdictions."
        )

    return information / information.sum()


def fuse_weights(
    w_bwm: np.ndarray,
    w_critic_m: np.ndarray,
    lam: float = 0.6,
) -> np.ndarray:
    """
    Convex combination of subjective (BWM) and objective (CRITIC-M) weights.

    w_fused = lam * w_bwm + (1 - lam) * w_critic_m

    Parameters
    ----------
    w_bwm      : BWM weight vector (subjective, from stakeholder survey)
    w_critic_m : CRITIC-M weight vector (objective, from matrix variance)
    lam        : float ∈ [0, 1]; lam=1 → pure expert opinion, lam=0 → pure data.
                 Expose this to the business as a single tunable knob, not buried.

    Returns
    -------
    np.ndarray summing to 1.0
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    w = lam * w_bwm + (1.0 - lam) * w_critic_m
    return w / w.sum()  # re-normalize for floating-point safety
