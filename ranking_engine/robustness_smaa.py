"""
robustness_smaa.py — V4: Stochastic Multicriteria Acceptability Analysis (SMAA).

Concept
-------
A single weight vector (even one carefully derived via BWM + CRITIC-M) still
represents one opinion.  SMAA answers a different question:

  "Across *all* plausible weight vectors — every possible combination of
   priorities that sums to 1 and is non-negative — what fraction of those
   worlds puts UAE in first place?"

This is the Rank Acceptability Index (RAI).

How it works:
  1. Sample N weight vectors uniformly from the weight simplex.
     Dirichlet(1,...,1) is exactly the uniform distribution over non-negative
     vectors summing to 1 — the fair "no opinion" prior.
  2. Run TOPSIS-RAD for each sampled weight vector.
  3. Tally: for market A, count how many trials assign it rank 1, rank 2, etc.
  4. Divide by N → probability matrix.

Two outputs:
  - Rank Acceptability Index (RAI): p[market, rank] = P(market achieves rank)
  - Central Weight Vector (CWV): for each market, the *average* weight vector
    among all trials where that market won.  Show this to the executive team —
    it reveals "this is the implied philosophy that makes UAE win."  If that
    philosophy doesn't match their actual risk appetite, the model is catching a
    blind spot.

Reference: Lahdelma et al. (2001), "Implementing SMAA", doi:10.1016/S0305-0548(00)00026-2
"""

import numpy as np
import pandas as pd

from .topsis_rad import topsis_rad
from .criterion_mapping import BENEFIT_MASK


def smaa_topsis(
    matrix: pd.DataFrame,
    benefit_mask: np.ndarray,
    vpl: np.ndarray,
    dpl: np.ndarray,
    n_trials: int = 20_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | None]]:
    """
    Run SMAA over TOPSIS-RAD and compute the Rank Acceptability Index.

    Parameters
    ----------
    matrix       : pd.DataFrame (jurisdictions × criteria), 1-10 scores
    benefit_mask : bool array (n_criteria,)
    vpl          : np.ndarray (n_criteria,) — fixed anti-ideal anchors
    dpl          : np.ndarray (n_criteria,) — fixed ideal anchors
    n_trials     : number of random weight vectors to sample
    seed         : numpy random seed for reproducibility

    Returns
    -------
    rai : pd.DataFrame  shape (n_jurisdictions, n_jurisdictions)
          columns = rank positions 1..n; values = probabilities.
          rai.loc["UAE", 1] = P(UAE ranks 1st) across all plausible weights.

    central_weight_vectors : dict[jurisdiction_name → np.ndarray | None]
          Average weight vector in the trials where that market won rank 1.
          None if the market never won rank 1 across all trials (i.e. dominated).
    """
    rng = np.random.default_rng(seed)
    n_crit = len(vpl)
    alts = matrix.index.tolist()
    n_alts = len(alts)

    rank_counts = pd.DataFrame(0, index=alts, columns=range(1, n_alts + 1))
    weight_sum = {a: np.zeros(n_crit) for a in alts}
    win_count  = {a: 0 for a in alts}

    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(n_crit))
        scores = topsis_rad(matrix, w, benefit_mask, vpl, dpl)
        # rank 1 = highest closeness coefficient
        order = scores.sort_values(ascending=False).index.tolist()
        for rank_pos, alt in enumerate(order, start=1):
            rank_counts.loc[alt, rank_pos] += 1
        winner = order[0]
        weight_sum[winner] += w
        win_count[winner]  += 1

    rai = rank_counts / n_trials

    central_weight_vectors = {
        a: (weight_sum[a] / win_count[a] if win_count[a] > 0 else None)
        for a in alts
    }

    return rai, central_weight_vectors
