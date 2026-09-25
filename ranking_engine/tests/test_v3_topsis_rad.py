"""
test_v3_topsis_rad.py — Tests for V3: TOPSIS-RAD.

Key tests in this file:
  1. Rank-reversal regression (the primary TOPSIS-RAD guarantee).
  2. DPL capping stability.
  3. Fix C monotonicity tests for C6/C7/C8 (the direct regression for bug #3).

Run:  python -m pytest ranking_engine/tests/test_v3_topsis_rad.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from ranking_engine.criterion_mapping import CRITERIA, BENEFIT_MASK, MAPPING_FUNCTIONS
from ranking_engine.knockout import build_decision_matrix
from ranking_engine.topsis_rad import topsis_rad

# ---------------------------------------------------------------------------
# Fixed anchors — NOW UNIFORM (higher = better for all 8 columns)
# VPL = 1.0 for all, DPL = 8.5 for all.
# Old split values ([1,1,1,1,1,10,10,10] / [10,10,10,10,10,1,1,1]) were
# for the buggy direction-branching code; they do NOT apply here.
# ---------------------------------------------------------------------------
VPL = np.array([1.0] * 8, dtype=float)
DPL = np.array([8.5] * 8, dtype=float)

# The verified weight vector from the implementation plan (§8.3)
WEIGHTS = np.array([0.1243, 0.0932, 0.3336, 0.1864, 0.0746, 0.0327, 0.0621, 0.0932])

# Three-market matrix — already direction-corrected (higher = better for ALL columns).
# C6/C7/C8 scores here represent mapped values: a score of 7 means "low burden" (good).
THREE_MARKETS = pd.DataFrame(
    {"UAE": [8,7,8,9,7,7,7,7], "Brazil": [7,6,7,6,6,6,5,5], "Indonesia": [6,5,5,4,5,5,4,4]},
    index=CRITERIA
).T

# Four-market matrix (adding Kenya — a weak alternative)
KENYA_ROW = pd.DataFrame({"Kenya": [2, 2, 3, 2, 3, 2, 2, 2]}, index=CRITERIA).T
FOUR_MARKETS = pd.concat([THREE_MARKETS, KENYA_ROW])


class TestTOPSISRAD:

    def test_closeness_in_range(self):
        """Closeness coefficients must be in [0, 1]."""
        cc = topsis_rad(THREE_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)
        assert (cc.values >= 0).all() and (cc.values <= 1).all()

    def test_uae_ranks_first_three_markets(self):
        """UAE should have the highest closeness coefficient in the 3-market set."""
        cc = topsis_rad(THREE_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)
        assert cc.idxmax() == "UAE"

    def test_order_is_uae_brazil_indonesia(self):
        """Verified order: UAE > Brazil > Indonesia."""
        cc = topsis_rad(THREE_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)
        order = cc.sort_values(ascending=False).index.tolist()
        assert order == ["UAE", "Brazil", "Indonesia"]

    # ------------------------------------------------------------------
    # THE KEY TEST: Rank-Reversal Regression
    # ------------------------------------------------------------------

    def test_rank_reversal_proof_adding_kenya(self):
        """
        Adding Kenya (a weak alternative) must NOT change the closeness
        coefficients or rankings of UAE, Brazil, and Indonesia.

        This is the primary guarantee TOPSIS-RAD provides over standard TOPSIS.
        If this test fails, the VPL/DPL anchors are incorrectly derived from
        the data instead of being fixed analyst-defined constants.
        """
        cc_3 = topsis_rad(THREE_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)
        cc_4 = topsis_rad(FOUR_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)

        for market in ["UAE", "Brazil", "Indonesia"]:
            np.testing.assert_almost_equal(
                cc_3[market], cc_4[market], decimal=6,
                err_msg=(
                    f"Rank reversal detected: {market}'s closeness changed when Kenya was added.\n"
                    f"Before: {cc_3[market]:.6f}, After: {cc_4[market]:.6f}"
                )
            )

    def test_rank_reversal_proof_order_unchanged(self):
        """The ranking order of original markets must be identical with and without Kenya."""
        cc_3 = topsis_rad(THREE_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)
        cc_4 = topsis_rad(FOUR_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)

        original_order_3 = cc_3.sort_values(ascending=False).index.tolist()
        original_order_4 = [m for m in cc_4.sort_values(ascending=False).index if m != "Kenya"]

        assert original_order_3 == original_order_4, (
            f"Ranking order changed when Kenya was added.\n"
            f"3-market order: {original_order_3}\n"
            f"4-market order (Kenya removed): {original_order_4}"
        )

    def test_capping_at_dpl_works(self):
        """
        The DPL capping guarantee is about STABILITY of OTHER markets.

        When UAE's C1 score is inflated to 15 (above DPL=8.5), it is capped.
        Brazil's and Indonesia's closeness must be completely unchanged — the
        geometric space for other markets is frozen.
        """
        above_dpl = THREE_MARKETS.copy()
        above_dpl.loc["UAE", "C1"] = 15.0   # above DPL[0] = 8.5

        cc_normal = topsis_rad(THREE_MARKETS, WEIGHTS, BENEFIT_MASK, VPL, DPL)
        cc_above  = topsis_rad(above_dpl, WEIGHTS, BENEFIT_MASK, VPL, DPL)

        for market in ["Brazil", "Indonesia"]:
            np.testing.assert_almost_equal(
                cc_normal[market], cc_above[market], decimal=6,
                err_msg=(
                    f"DPL capping failed: {market}'s closeness changed when UAE's score "
                    f"was inflated beyond DPL. Before={cc_normal[market]:.6f}, "
                    f"After={cc_above[market]:.6f}"
                )
            )


# ---------------------------------------------------------------------------
# Fix C — Monotonicity tests for C6 / C7 / C8 (cost criteria)
#
# This is the test class that would have caught bug #3.
#
# Design: two jurisdictions identical on all criteria except one cost criterion.
# Jurisdiction A has genuinely lower cost (better in real-world terms) — after
# direction correction by build_decision_matrix, A's MAPPED score is HIGHER.
# Assertion: A must have a strictly higher closeness than B, and the difference
# must be non-trivial (not "equal" — equal means the criterion was silently zeroed).
# ---------------------------------------------------------------------------

def _two_jurisdiction_raw_facts(crit_diffs: dict[str, tuple[float, float]]) -> tuple:
    """
    Build a pair of RawFact lists for JurA and JurB, identical except for
    the criteria in crit_diffs.  crit_diffs maps criterion_code →
    (raw_value_for_A, raw_value_for_B).

    Default raw values are the midpoint of each criterion's natural range so
    they map to a mid-range score.
    """
    from ranking_engine.schema import RawFact, EvidenceTag

    midpoints = {
        "C1": 0.40,       # HHI midpoint
        "C2": 20_000,     # GNI PPP midpoint
        "C3": 1,          # "partial" framework
        "C4": 50,         # leverage cap midpoint
        "C5": 1,          # rail tier 1
        "C6": 1_000_000,  # moderate capital requirement
        "C7": 2,          # moderate presence friction
        "C8": 4.0,        # moderate volatility
    }

    facts_a, facts_b = [], []
    for c in CRITERIA:
        raw_a, raw_b = crit_diffs.get(c, (midpoints[c], midpoints[c]))
        facts_a.append(RawFact(
            jurisdiction="JurA", criterion_code=c, raw_value=raw_a,
            evidence_tag=EvidenceTag.VERIFIED_FACT, source_citation="mono-test",
            as_of=date(2026, 1, 1),
        ))
        facts_b.append(RawFact(
            jurisdiction="JurB", criterion_code=c, raw_value=raw_b,
            evidence_tag=EvidenceTag.VERIFIED_FACT, source_citation="mono-test",
            as_of=date(2026, 1, 1),
        ))
    return facts_a, facts_b


class TestMonotonicityC6C7C8:
    """
    Fix C: Direct regression test for bug #3.

    Bug #3 was: topsis_rad() applied np.maximum(X, dpl_cost) for cost columns,
    which forced every C6/C7/C8 score to a constant 10.0 → two jurisdictions
    differing ONLY in C6 got identical closeness.

    Each test here isolates one criterion and verifies that the better jurisdiction
    (lower cost in real-world terms → higher mapped score) gets a strictly
    higher closeness coefficient, with a non-trivial margin.
    """

    def _run(self, crit_diffs):
        facts_a, facts_b = _two_jurisdiction_raw_facts(crit_diffs)
        facts_all = facts_a + facts_b
        matrix = build_decision_matrix(facts_all, MAPPING_FUNCTIONS)
        w = np.ones(8) / 8.0
        cc = topsis_rad(matrix, w, BENEFIT_MASK, VPL, DPL)
        return cc["JurA"], cc["JurB"], matrix

    def test_c6_lower_capital_scores_higher(self):
        """
        C6: JurA requires $100k capital, JurB requires $5M.
        After mapping: JurA scores much higher (low capital = high score).
        topsis_rad must rank JurA strictly above JurB, not tie them.
        Under bug #3, np.maximum(X, dpl=10.0) forced BOTH to 10.0 → tie.
        """
        cc_a, cc_b, matrix = self._run({"C6": (100_000, 5_000_000)})
        assert matrix.loc["JurA", "C6"] > matrix.loc["JurB", "C6"], \
            "Pre-condition failed: JurA's C6 mapped score should be > JurB's"
        assert cc_a > cc_b, (
            f"C6 monotonicity FAILED: lower capital should → higher closeness. "
            f"JurA(cc={cc_a:.4f}) <= JurB(cc={cc_b:.4f}). "
            f"Possible bug #3 regression: cost criteria capped to constant."
        )
        assert (cc_a - cc_b) > 0.01, \
            f"Margin too small ({cc_a-cc_b:.4f}): C6 may be silently zeroed"

    def test_c7_lower_friction_scores_higher(self):
        """
        C7: JurA friction tier 1 (lowest), JurB friction tier 4 (high).
        After mapping: JurA scores much higher.
        """
        cc_a, cc_b, matrix = self._run({"C7": (1, 4)})
        assert matrix.loc["JurA", "C7"] > matrix.loc["JurB", "C7"], \
            "Pre-condition failed: JurA's C7 mapped score should be > JurB's"
        assert cc_a > cc_b, (
            f"C7 monotonicity FAILED: lower friction should → higher closeness. "
            f"JurA(cc={cc_a:.4f}) <= JurB(cc={cc_b:.4f})."
        )
        assert (cc_a - cc_b) > 0.01, \
            f"Margin too small ({cc_a-cc_b:.4f}): C7 may be silently zeroed"

    def test_c8_lower_volatility_scores_higher(self):
        """
        C8: JurA volatility 1.5 (stable), JurB volatility 8.0 (chaotic).
        After mapping: JurA scores much higher.
        """
        cc_a, cc_b, matrix = self._run({"C8": (1.5, 8.0)})
        assert matrix.loc["JurA", "C8"] > matrix.loc["JurB", "C8"], \
            "Pre-condition failed: JurA's C8 mapped score should be > JurB's"
        assert cc_a > cc_b, (
            f"C8 monotonicity FAILED: lower volatility should → higher closeness. "
            f"JurA(cc={cc_a:.4f}) <= JurB(cc={cc_b:.4f})."
        )
        assert (cc_a - cc_b) > 0.01, \
            f"Margin too small ({cc_a-cc_b:.4f}): C8 may be silently zeroed"

    def test_all_benefit_criteria_monotonic(self):
        """
        Fix C also covers C1-C5. Run one monotonicity check per benefit criterion.
        """
        diffs = {
            "C1": (0.10, 0.70),    # low HHI (competitive) vs high HHI (monopoly)
            "C2": (60_000, 5_000), # high GNI vs low GNI
            "C3": (3, 0),          # explicit framework vs no framework
            "C4": (200, 5),        # high leverage vs no leverage
            "C5": (2, 0),          # rail tier 2 vs no access
        }
        for crit, (raw_a, raw_b) in diffs.items():
            cc_a, cc_b, matrix = self._run({crit: (raw_a, raw_b)})
            mapped_a = matrix.loc["JurA", crit]
            mapped_b = matrix.loc["JurB", crit]
            assert mapped_a > mapped_b, \
                f"{crit}: pre-condition failed — JurA mapped {mapped_a:.2f}, JurB {mapped_b:.2f}"
            assert cc_a > cc_b, (
                f"{crit} monotonicity FAILED: better raw value → lower closeness. "
                f"JurA(cc={cc_a:.4f}) <= JurB(cc={cc_b:.4f})"
            )
