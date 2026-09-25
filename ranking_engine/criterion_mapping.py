"""
criterion_mapping.py — deterministic raw_value → 1-10 score mapping.

OWNED BY THE RANKING TEAM, NOT THE EXTRACTION TEAM.

This is where domain expertise is encoded as code, not as model output.
A hallucinated LLM number arriving as raw_value cannot silently become a
ranking score — it must pass through a human-reviewed threshold table here.

Each mapping function takes the *unmapped* extracted number (RawFact.raw_value)
and returns a float in [1.0, 10.0].
"""

import numpy as np

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

CRITERIA = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# True  = benefit criterion (higher score is better, e.g. Demand Scale)
# False = cost criterion    (lower score is better, e.g. Capital Burden)
BENEFIT_MASK = np.array([True, True, True, True, True, False, False, False])


# ---------------------------------------------------------------------------
# Individual criterion mappings
# ---------------------------------------------------------------------------

def map_c1_demand_scale(hhi_index: float) -> float:
    """
    C1 Demand Scale — HHI of derivative exchange concentration.
    Lower HHI (more fragmented, more contestable market) → higher score.
    HHI range: 0.0 (perfectly fragmented) to 1.0 (pure monopoly).
    """
    # Invert: fragmented = good, monopoly = bad
    score = 10.0 * (1.0 - hhi_index)
    return float(np.clip(score, 1.0, 10.0))


def map_c2_revenue_depth(gni_per_capita_ppp_usd: float) -> float:
    """
    C2 Revenue Depth — GNI per capita (PPP, USD).
    Higher income = deeper per-user revenue potential.
    Breakpoints calibrated to global derivative-trading population.
    """
    breakpoints = [
        (2_000,  1.0),
        (5_000,  3.0),
        (10_000, 5.0),
        (20_000, 6.5),
        (35_000, 8.0),
        (55_000, 9.0),
        (80_000, 10.0),
    ]
    for threshold, score in breakpoints:
        if gni_per_capita_ppp_usd <= threshold:
            return score
    return 10.0


def map_c3_pathway_clarity(framework_exists: float) -> float:
    """
    C3 Pathway Clarity — existence of an explicit in-force framework for retail
    leveraged derivatives / synthetic equities.
    Categorical input encoded as:
        0 = banned / no framework
        1 = draft / consultation only
        2 = framework exists but ambiguous on synthetic equities
        3 = explicit, in-force framework (e.g. VARA Part V, BCB Resolutions)
    """
    mapping = {0: 1.0, 1: 4.0, 2: 7.0, 3: 10.0}
    return mapping.get(int(framework_exists), 1.0)


def map_c4_leverage_fit(cap_ratio: float) -> float:
    """
    C4 Leverage Fit — maximum permitted retail leverage cap.
    Higher cap = better fit for the platform's product (which targets high leverage).
    """
    breakpoints = [(5, 2.0), (10, 4.0), (20, 5.0), (50, 7.0), (100, 8.0), (200, 9.0), (300, 10.0)]
    for cap, score in breakpoints:
        if cap_ratio <= cap:
            return score
    return 10.0


def map_c5_local_rail_fit(openness_level: float) -> float:
    """
    C5 Local Rail Fit — availability/openness of instant fiat payment rails to VASPs.
    Categorical: 0=Low, 1=Medium, 2=High.
    """
    mapping = {0: 2.0, 1: 6.0, 2: 10.0}
    return mapping.get(int(openness_level), 2.0)


def map_c6_capital_burden(usd_required: float) -> float:
    """
    C6 Capital Burden (COST criterion) — minimum statutory paid-up capital + licensing fees.
    The KNOCKOUT threshold is checked separately in knockout.py — this returns a
    1-10 score used in TOPSIS for markets that survive the knockout.
    Lower capital required = higher (better) score.
    """
    breakpoints = [
        (100_000,   10.0),
        (250_000,    8.0),
        (500_000,    6.5),
        (1_000_000,  5.0),
        (3_000_000,  3.5),
        (10_000_000, 2.0),
    ]
    for threshold, score in breakpoints:
        if usd_required <= threshold:
            return score
    return 1.0


def map_c7_presence_friction(friction_score: float) -> float:
    """
    C7 Presence Friction (COST criterion) — aggregated mandate score:
    local physical office + resident director + data-localization laws + custody reqs.
    Categorical integer 0–4 (one point per mandate present).
    More mandates = higher friction = lower score (cost criterion).
    """
    mapping = {0: 10.0, 1: 8.0, 2: 5.5, 3: 3.0, 4: 1.0}
    return mapping.get(int(np.clip(friction_score, 0, 4)), 1.0)


def map_c8_regulatory_volatility(volatility_index: float) -> float:
    """
    C8 Regulatory Volatility (COST criterion) — frequency of policy reversals,
    draft-stage vs enacted laws, stabilized by World Bank WGI Regulatory Quality.
    Input: composite index 0 (stable) to 10 (extreme volatility).
    Higher volatility = lower score (cost criterion).
    """
    score = 10.0 - volatility_index
    return float(np.clip(score, 1.0, 10.0))


# ---------------------------------------------------------------------------
# Mapping registry — used by build_decision_matrix() in knockout.py
# ---------------------------------------------------------------------------

MAPPING_FUNCTIONS: dict = {
    "C1": map_c1_demand_scale,
    "C2": map_c2_revenue_depth,
    "C3": map_c3_pathway_clarity,
    "C4": map_c4_leverage_fit,
    "C5": map_c5_local_rail_fit,
    "C6": map_c6_capital_burden,
    "C7": map_c7_presence_friction,
    "C8": map_c8_regulatory_volatility,
}
