"""
topsis_rad.py — V3: TOPSIS-RAD (Reference points and Acceptability Dimensions).

Concept — The rank-reversal problem
-------------------------------------
Standard TOPSIS defines its Positive Ideal Solution (PIS) and Negative Ideal
Solution (NIS) from the best/worst scores *actually observed* in the current
dataset.  This means:

  1. You evaluate UAE, Brazil, Indonesia → PIS is anchored to UAE's scores.
  2. You add Kenya (much weaker) → NIS shifts → distances change for UAE/Brazil/Indonesia.
  3. UAE and Brazil may swap ranks, even though their underlying data never changed.

This is rank reversal.  It's a well-documented failure mode of standard TOPSIS
(see the literature review: doi.org/10.1016/j.omega.2014.11.009) and it's
especially dangerous here because we *constantly add new markets* to the pipeline.

The fix — TOPSIS-RAD
---------------------
Instead of deriving PIS/NIS from the observed dataset, we anchor them to
*analyst-defined, fixed reference levels* that never change:

  VPL (Vetoed Performance Level) — the minimum tolerable score per criterion.
      A jurisdiction scoring below VPL on any criterion should already be knocked
      out.  VPL effectively defines the "anti-ideal" independently of the data.

  DPL (Desired Performance Level) — the aspirational "good enough" score per
      criterion.  Market scores are *capped at DPL before normalizing*.
      This means one jurisdiction that scores 10/10 cannot stretch the frontier
      and inflate everyone else's apparent distance.

Because VPL and DPL never move when markets are added or removed, the geometric
space is frozen.  Adding Kenya does not shift the origin.  Rank reversal is
structurally impossible, not just less likely.

Direction contract
------------------
This function receives a decision matrix that has ALREADY been direction-
corrected by build_decision_matrix() / criterion_mapping.py.  In that matrix,
a HIGHER score always means "more desirable" for every column, including the
three cost criteria (C6 capital burden, C7 presence friction, C8 volatility).

Therefore:
  - Capping is applied uniformly as np.minimum(X, dpl) for all columns —
    no benefit/cost branch needed.
  - PIS is always the DPL anchor (higher = ideal), NIS is always VPL (lower = anti-ideal) —
    no benefit/cost branch needed.

The benefit_mask parameter is retained in the function signature so callers
and tests can see the raw-data semantics, but it is not consumed for any
computation.  It must NOT be passed to np.where() for direction branching.

Reference: Costa et al. (2026), "TOPSIS-RAD: Ranking According to Desires",
           arXiv:2606.07253
"""

import numpy as np
import pandas as pd


def topsis_rad(
    matrix: pd.DataFrame,
    weights: np.ndarray,
    benefit_mask: np.ndarray,      # kept for documentation; NOT used for direction branching
    vpl: np.ndarray,
    dpl: np.ndarray,
) -> pd.Series:
    """
    Compute TOPSIS-RAD closeness coefficients for each jurisdiction.

    Parameters
    ----------
    matrix       : pd.DataFrame (jurisdictions × criteria), 1-10 scores.
                   IMPORTANT: must be fully direction-corrected — higher score = better
                   for ALL columns.  build_decision_matrix() guarantees this.
    weights      : np.ndarray (n_criteria,), must sum to 1
    benefit_mask : bool array (n_criteria,) — documents raw-data semantics only.
                   NOT used for any internal computation; see "Direction contract" above.
    vpl          : np.ndarray (n_criteria,) — Vetoed Performance Level (anti-ideal floor).
                   All values should be at or below the minimum possible matrix score.
    dpl          : np.ndarray (n_criteria,) — Desired Performance Level (aspirational cap).
                   Any market score above DPL is capped to DPL (uniform, no branching).
                   Set DPL strictly below the scale maximum (e.g. 8.5 not 10.0) so the
                   cap is actually active; setting DPL = scale_max makes capping a no-op.

    Returns
    -------
    pd.Series  — closeness coefficient per jurisdiction (higher = better), index = jurisdiction names
    """
    X = matrix.values.astype(float)

    # Step 1: cap scores at DPL — applied uniformly (no benefit/cost branch)
    # Every column is already "higher = better", so DPL is always the upper cap.
    X_capped = np.minimum(X, dpl)

    # Step 2: normalize against FIXED anchors (VPL, DPL), not the data's own extremes
    anchors = np.vstack([vpl, dpl])                        # shape (2, n_criteria)
    denom = np.sqrt((anchors ** 2).sum(axis=0))            # scalar per column
    denom[denom == 0] = 1e-12                              # guard zero-denom
    Xn   = X_capped / denom
    Wn   = Xn * weights

    # Step 3: fixed PIS and NIS from DPL/VPL, not the data.
    # Since higher = better for all columns: PIS = DPL (anchors[1]), NIS = VPL (anchors[0]).
    # No benefit_mask branch — the direction is uniform.
    anchors_n = anchors / denom                            # shape (2, n_criteria)
    pis = anchors_n[1] * weights                           # DPL, always — the ideal
    nis = anchors_n[0] * weights                           # VPL, always — the anti-ideal

    # Step 4: Euclidean distances to fixed PIS and NIS
    d_pos = np.sqrt(((Wn - pis) ** 2).sum(axis=1))
    d_neg = np.sqrt(((Wn - nis) ** 2).sum(axis=1))

    # Step 5: closeness coefficient — 0 = at anti-ideal, 1 = at ideal
    denom2 = d_pos + d_neg
    denom2[denom2 == 0] = 1e-12
    closeness = d_neg / denom2

    return pd.Series(closeness, index=matrix.index, name="closeness")
