"""
data_feeds/scoring.py — Map raw feed values to the 1-10 scale the matrix expects.

Design constraints
------------------
These thresholds are human-defined policy decisions, not computed from the data.
The same principle as VPL/DPL: they must be documented here, not buried, and
changes to them should be deliberate — edit this file, not a config somewhere.

All functions return float in [1.0, 10.0] unless the input is None (→ returns None).
The 1-10 range matches the mapping scale used by criterion_mapping.py's MAPPING_FUNCTIONS
so Tier-4 facts slot directly into the same matrix cells without any further conversion.

Thresholds are documented with their rationale. Change them here when the management
team revises their scoring philosophy — it's a one-line edit with a clear audit trail.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# C2 — Revenue Depth
# ---------------------------------------------------------------------------

# GNI per capita, PPP (current international USD) → 1-10 score
# Breakpoints reflect purchasing power tiers relevant to a derivatives platform:
#   <$2,000   → very low user spending power (score 1)
#   $2-8K     → low-middle (score 4)
#   $8-20K    → middle (score 6)
#   $20-40K   → upper-middle (score 8)
#   >$40K     → high (UAE, Singapore tier, score 10)
_GNI_BREAKPOINTS: list[tuple[float, float]] = [
    (2_000,  1.0),
    (8_000,  4.0),
    (20_000, 6.0),
    (40_000, 8.0),
]
_GNI_MAX_SCORE = 10.0

def score_gni(gni_ppp: float | None) -> float | None:
    """GNI per capita PPP (USD) → C2 score [1, 10]."""
    if gni_ppp is None:
        return None
    for cap, score in _GNI_BREAKPOINTS:
        if gni_ppp <= cap:
            return score
    return _GNI_MAX_SCORE


# Account ownership / financial-inclusion proxy (% of adults)
# Used as a secondary C2 signal; blended with GNI in score_revenue_depth().
_FINDEX_BREAKPOINTS: list[tuple[float, float]] = [
    (20.0,  1.0),
    (40.0,  3.0),
    (60.0,  6.0),
    (80.0,  8.0),
]
_FINDEX_MAX_SCORE = 10.0

def score_findex(pct: float | None) -> float | None:
    """Account ownership % of adults → financial-inclusion sub-score [1, 10]."""
    if pct is None:
        return None
    for cap, score in _FINDEX_BREAKPOINTS:
        if pct <= cap:
            return score
    return _FINDEX_MAX_SCORE


def score_revenue_depth(gni_ppp: float | None, account_ownership_pct: float | None) -> float | None:
    """
    Blend GNI and financial-inclusion scores into the final C2 raw value.

    Weighting: 70% GNI (capital depth), 30% Findex (access breadth).
    If only one source is available, use that one alone.
    Returns None if neither is available.
    """
    gni_score    = score_gni(gni_ppp)
    findex_score = score_findex(account_ownership_pct)

    if gni_score is not None and findex_score is not None:
        return round(0.70 * gni_score + 0.30 * findex_score, 4)
    if gni_score is not None:
        return gni_score
    if findex_score is not None:
        return findex_score
    return None


# ---------------------------------------------------------------------------
# C1 — Demand Scale (HHI)
# ---------------------------------------------------------------------------

# HHI is in [0, 10,000]. Lower = more fragmented = more contestable.
# C1 is a benefit criterion, so the mapping inverts HHI:
#   high HHI (monopoly)           → low C1 score  (bad for entry)
#   low  HHI (fragmented market)  → high C1 score  (good for entry)
# Thresholds derived from standard DoJ/FTC HHI competitive-analysis brackets:
#   >2,500 → concentrated   (score 1–3)
#   1,500–2,500 → moderate  (score 4–6)
#   <1,500 → fragmented     (score 7–9)
#   <500   → highly frag.   (score 10)

def score_hhi(hhi: float | None) -> float | None:
    """HHI [0, 10,000] → C1 score [1, 10] (inverted — lower HHI = higher score)."""
    if hhi is None:
        return None
    if hhi <   500: return 10.0
    if hhi < 1_500: return  8.0
    if hhi < 2_500: return  5.0
    if hhi < 5_000: return  3.0
    return 1.0


# ---------------------------------------------------------------------------
# C8 — Regulatory Volatility (WGI Regulatory Quality)
# ---------------------------------------------------------------------------

# WGI Regulatory Quality estimate is in [-2.5, +2.5].
# C8 is a cost criterion (high volatility = bad), but criterion_mapping.py
# handles the inversion. The raw value we produce here is the WGI score itself
# (on the 1-10 scale), which criterion_mapping.py then maps further.
# Linear rescale: -2.5 → 1.0, +2.5 → 10.0

def score_wgi_regulatory_quality(estimate: float | None) -> float | None:
    """WGI RQ estimate [-2.5, +2.5] → 1-10 raw value."""
    if estimate is None:
        return None
    score = ((estimate + 2.5) / 5.0) * 9.0 + 1.0
    return round(min(10.0, max(1.0, score)), 4)


# ---------------------------------------------------------------------------
# C5 — Local Rail Fit (base signal only)
# ---------------------------------------------------------------------------

# The automated base signal uses World Bank account-ownership as a structural
# floor indicator. The VASP-openness flag is NOT automated (that's Tier 3).
# Output is a 1-10 score that feeds C5's raw_value; the same breakpoints as
# Findex apply since the same indicator is used.
#
# Callers in raw_fact_adapter.py must tag these ANALYST_ESTIMATE, not
# VERIFIED_FACT, because the VASP-openness component is missing.

def score_c5_base_signal(account_ownership_pct: float | None) -> float | None:
    """
    Digital-payments infrastructure proxy for C5 base signal.
    Reuses Findex breakpoints — same indicator (FX.OWN.TOTL.ZS).
    Returns None if no data; caller skips producing a fact.
    """
    return score_findex(account_ownership_pct)
