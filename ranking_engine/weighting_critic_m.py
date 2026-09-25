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

    # Step 1: Logistic squash — robust normalization around median/IQR.
    # All columns are treated identically (no direction flip) because the
    # decision matrix is already fully direction-corrected.
    for j in range(X.shape[1]):
        col = X[:, j]                           # higher already = better for every column
        med = np.median(col)
        iqr = float(np.subtract(*np.percentile(col, [75, 25]))) or 1e-9  # guard zero-IQR
        # Logistic squash: robust to outliers, bounded in (0,1)
        X[:, j] = 1.0 / (1.0 + np.exp(-(col - med) / iqr))

    # Step 2: information content = std × conflict (sum of 1-corr with every other crit)
    std = X.std(axis=0, ddof=1)
    corr_matrix = np.corrcoef(X, rowvar=False)
    conflict = (1.0 - corr_matrix).sum(axis=0)   # shape (n_criteria,)
    information = std * conflict

    if information.sum() == 0:
        raise ValueError("CRITIC-M: all criteria have zero information content — "
                         "check that your matrix has more than one row.")

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
