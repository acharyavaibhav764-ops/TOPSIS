"""
knockout.py — V0: Decision matrix assembly + hard capital knockout screen.

Concept
-------
Before any weighting or ranking runs, purge any jurisdiction whose capital
requirement exceeds what the business can actually fund.  This is not a low
score — it is a hard binary constraint (Vietnam's ~$380M charter capital vs.
a startup's runway).  Running the knockout first keeps the expensive downstream
stages (BWM, SMAA, Monte Carlo) from wasting cycles on markets that were never
viable.

Two responsibilities:
1. build_decision_matrix  — pivot a flat list of RawFact rows into a
                            jurisdiction × criterion DataFrame of 1-10 scores.
2. knockout_screen        — drop jurisdictions whose C6 raw capital figure
                            exceeds the liquid runway.
"""

import pandas as pd
import numpy as np
from typing import Callable

from .schema import RawFact
from .criterion_mapping import CRITERIA, BENEFIT_MASK, MAPPING_FUNCTIONS


# ---------------------------------------------------------------------------
# V0-A: Decision matrix assembly
# ---------------------------------------------------------------------------

def build_decision_matrix(
    facts: list[RawFact],
    mapping_fns: dict[str, Callable[[float], float]] | None = None,
) -> pd.DataFrame:
    """
    Pivot a flat list of RawFact rows into a (jurisdictions × criteria) DataFrame
    of 1-10 scores.

    Parameters
    ----------
    facts       : list of RawFact (the full extraction pipeline output)
    mapping_fns : dict mapping criterion_code → callable(raw_value) → float[1,10]
                  Defaults to MAPPING_FUNCTIONS from criterion_mapping.py.

    Returns
    -------
    pd.DataFrame  shape (n_jurisdictions, 8), columns = CRITERIA, index = jurisdiction names
    """
    if mapping_fns is None:
        mapping_fns = MAPPING_FUNCTIONS

    rows: dict[str, dict[str, float]] = {}
    missing: list[tuple[str, str]] = []

    for f in facts:
        fn = mapping_fns.get(f.criterion_code)
        if fn is None:
            # Unknown criterion — skip, but surface the problem.
            missing.append((f.jurisdiction, f.criterion_code))
            continue
        score = fn(f.raw_value)
        rows.setdefault(f.jurisdiction, {})[f.criterion_code] = score

    if missing:
        print(f"[WARNING] build_decision_matrix: no mapping for {missing}; those cells will be NaN")

    df = pd.DataFrame(rows).T   # shape: (jurisdictions, criteria)
    # Reorder columns to canonical CRITERIA order; fill any gaps with NaN
    df = df.reindex(columns=CRITERIA)
    return df


# ---------------------------------------------------------------------------
# V0-B: Hard knockout screen (C6 capital burden)
# ---------------------------------------------------------------------------

def knockout_screen(
    matrix: pd.DataFrame,
    capital_required: dict[str, float],
    liquid_runway_usd: float,
) -> pd.DataFrame:
    """
    Remove any jurisdiction whose raw C6 capital requirement (in USD) exceeds
    the business's liquid runway.

    Note: capital_required uses the *raw* dollar figure from RawFact, not the
    1-10 score.  This ensures the knockout fires at the correct absolute threshold
    regardless of how the score mapping is calibrated.

    Parameters
    ----------
    matrix            : output of build_decision_matrix
    capital_required  : dict jurisdiction → minimum statutory capital (USD)
    liquid_runway_usd : the startup's deployable capital ceiling (USD)

    Returns
    -------
    pd.DataFrame with ineligible jurisdictions removed
    """
    survivors = [
        j for j in matrix.index
        if capital_required.get(j, 0.0) <= liquid_runway_usd
    ]
    dropped = sorted(set(matrix.index) - set(survivors))
    if dropped:
        for j in dropped:
            cap = capital_required.get(j, 0.0)
            print(
                f"[KNOCKOUT] {j}: capital required ${cap:,.0f} "
                f"> runway ${liquid_runway_usd:,.0f} — removed from pipeline"
            )
    return matrix.loc[survivors].copy()


# ---------------------------------------------------------------------------
# Convenience: build capital_required dict from RawFact list
# ---------------------------------------------------------------------------

def extract_capital_required(facts: list[RawFact]) -> dict[str, float]:
    """Extract the raw C6 dollar figure per jurisdiction from the fact list."""
    return {f.jurisdiction: f.raw_value for f in facts if f.criterion_code == "C6"}
