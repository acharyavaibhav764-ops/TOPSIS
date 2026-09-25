"""
robustness_mc.py — V4: Evidence-calibrated Monte Carlo simulation.

Concept
-------
A legally verified capital floor from a central bank gazette and a subjective
guess about payment-rail readiness do not deserve the same amount of injected
noise.  Applying uniform ±X% noise across the entire matrix is mathematically
dishonest — it implies equal confidence in every data point.

The LLM extraction pipeline assigns each fact an EvidenceTag:
  VERIFIED_FACT     → tight Gaussian noise (the fact is law; minor transcription error only)
  ANALYST_ESTIMATE  → wider Gaussian noise (proxy / secondary source; meaningful uncertainty)
  UNRESOLVED_POLICY → bimodal flip: the jurisdiction is genuinely in one of two
                      discrete regulatory regimes.  Score doesn't jitter uniformly —
                      it probabilistically flips between the two regimes (e.g.,
                      "crypto regulator allows synthetics" vs "securities regulator bans them").

Procedure:
  For each Monte Carlo trial:
    1. Perturb every cell in the decision matrix according to its EvidenceTag.
    2. Re-run TOPSIS-RAD with the perturbed matrix and the fused weight vector.
    3. Record the closeness coefficient per jurisdiction.
  After N trials, compute the 5th–95th percentile of closeness per jurisdiction.
  This is the output confidence interval that travels with the point rank.
"""

import numpy as np
import pandas as pd

from .schema import EvidenceTag, RawFact
from .topsis_rad import topsis_rad
from .criterion_mapping import MAPPING_FUNCTIONS


# ---------------------------------------------------------------------------
# Per-cell perturbation
# ---------------------------------------------------------------------------

def perturb_score(
    value: float,
    tag: EvidenceTag,
    lo: float,
    hi: float,
    rng: np.random.Generator,
    alt_value: float | None = None,
) -> float:
    """
    Apply evidence-calibrated noise to one decision-matrix cell.

    Parameters
    ----------
    value     : the current 1-10 score for this (jurisdiction, criterion) cell
    tag       : EvidenceTag governing the noise level
    lo, hi    : score bounds (typically 1.0, 10.0)
    rng       : numpy random Generator (for reproducibility)
    alt_value : the alternative regime score (required when tag = UNRESOLVED_POLICY)
                e.g. value=8 (lenient crypto regulator) vs alt_value=2 (strict securities regulator)

    Returns
    -------
    float clipped to [lo, hi]
    """
    scale = hi - lo  # normalizing factor for noise magnitude

    if tag == EvidenceTag.VERIFIED_FACT:
        # Tight noise: ±5% of the score range — accounts for minor transcription uncertainty
        draw = rng.normal(value, scale=0.05 * scale)

    elif tag == EvidenceTag.ANALYST_ESTIMATE:
        # Wide noise: ±15% of score range — proxy source could be materially off
        draw = rng.normal(value, scale=0.15 * scale)

    elif tag == EvidenceTag.UNRESOLVED_POLICY:
        # Bimodal flip: genuinely uncertain which regulatory regime applies.
        # 50-50 coin flip between the two discrete outcomes, then small Gaussian jitter.
        chosen = value if rng.random() < 0.5 else (alt_value if alt_value is not None else lo)
        draw = rng.normal(chosen, scale=0.05 * scale)

    else:
        draw = value  # unknown tag — leave score untouched

    return float(np.clip(draw, lo, hi))


# ---------------------------------------------------------------------------
# Monte Carlo orchestration
# ---------------------------------------------------------------------------

def monte_carlo_robustness(
    base_matrix: pd.DataFrame,
    facts: list[RawFact],
    weights: np.ndarray,
    benefit_mask: np.ndarray,
    vpl: np.ndarray,
    dpl: np.ndarray,
    n_trials: int = 10_000,
    seed: int = 42,
    score_lo: float = 1.0,
    score_hi: float = 10.0,
) -> pd.DataFrame:
    """
    Run evidence-calibrated Monte Carlo and return closeness-coefficient
    percentile intervals per jurisdiction.

    Parameters
    ----------
    base_matrix  : the point-estimate decision matrix (jurisdictions × criteria)
    facts        : original RawFact list — used to look up EvidenceTag and alt_raw_value
    weights      : fused weight vector (from V2 BWM + CRITIC-M)
    benefit_mask : bool array (n_criteria,)
    vpl, dpl     : fixed TOPSIS-RAD anchors
    n_trials     : number of perturbed re-rankings to run
    seed         : numpy seed
    score_lo/hi  : score scale bounds for clip (default 1–10)

    Returns
    -------
    pd.DataFrame with columns [p5, p50, p95, std] per jurisdiction (index)

    Notes
    -----
    alt_raw_value on RawFact is in the SAME raw units as raw_value (e.g. USD,
    leverage ratio). It must be mapped through MAPPING_FUNCTIONS before use
    as the alternate bimodal flip target — passing raw values directly to
    perturb_score() as a 1-10 score would be meaningless (a dollar amount
    clipped to [1,10] is always 1.0 or 10.0, not a meaningful regime score).
    """
    rng = np.random.default_rng(seed)
    alts = base_matrix.index.tolist()
    criteria = base_matrix.columns.tolist()

    # Build lookup: (jurisdiction, criterion) → (EvidenceTag, mapped_alt_score)
    # alt_raw_value is in raw units — convert to 1-10 score space here, once,
    # so perturb_score always receives values on the same scale as the matrix.
    tag_lookup: dict[tuple[str, str], tuple[EvidenceTag, float | None]] = {}
    for f in facts:
        mapped_alt: float | None = None
        if f.alt_raw_value is not None and f.criterion_code in MAPPING_FUNCTIONS:
            try:
                mapped_alt = float(MAPPING_FUNCTIONS[f.criterion_code](f.alt_raw_value))
            except Exception:
                mapped_alt = None  # mapping failed — fall back to None (uses lo in perturb)
        tag_lookup[(f.jurisdiction, f.criterion_code)] = (f.evidence_tag, mapped_alt)

    all_closeness = np.zeros((n_trials, len(alts)))

    for t in range(n_trials):
        # Explicitly cast to float64 to avoid pandas FutureWarning when
        # assigning float values into a column that was inferred as int64.
        noisy = base_matrix.astype(float).copy()
        for jur in alts:
            for crit in criteria:
                tag, alt_score = tag_lookup.get(
                    (jur, crit),
                    (EvidenceTag.ANALYST_ESTIMATE, None),  # default if not in facts
                )
                noisy.loc[jur, crit] = perturb_score(
                    value=float(base_matrix.loc[jur, crit]),
                    tag=tag,
                    lo=score_lo,
                    hi=score_hi,
                    rng=rng,
                    alt_value=alt_score,
                )
        closeness = topsis_rad(noisy, weights, benefit_mask, vpl, dpl)
        all_closeness[t] = closeness.loc[alts].values

    results = pd.DataFrame(
        {
            "p5":  np.percentile(all_closeness, 5, axis=0),
            "p50": np.percentile(all_closeness, 50, axis=0),
            "p95": np.percentile(all_closeness, 95, axis=0),
            "std": all_closeness.std(axis=0),
        },
        index=alts,
    )
    return results
