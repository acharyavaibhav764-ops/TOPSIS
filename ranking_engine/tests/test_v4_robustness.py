"""
test_v4_robustness.py — Tests for V4: SMAA + Evidence-Calibrated Monte Carlo.

Run:  python -m pytest ranking_engine/tests/test_v4_robustness.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from ranking_engine.schema import RawFact, EvidenceTag
from ranking_engine.criterion_mapping import CRITERIA, BENEFIT_MASK
from ranking_engine.robustness_smaa import smaa_topsis
from ranking_engine.robustness_mc import perturb_score, monte_carlo_robustness

# Shared fixtures
VPL = np.array([1, 1, 1, 1, 1, 10, 10, 10], dtype=float)
DPL = np.array([10, 10, 10, 10, 10, 1, 1, 1], dtype=float)
WEIGHTS = np.array([0.1243, 0.0932, 0.3336, 0.1864, 0.0746, 0.0327, 0.0621, 0.0932])

MATRIX = pd.DataFrame(
    {"UAE": [8,7,8,9,7,3,4,3], "Brazil": [7,6,7,6,6,4,5,5], "Indonesia": [6,5,5,4,5,3,6,6]},
    index=CRITERIA
).T


def make_fact(j, c, v, tag=EvidenceTag.VERIFIED_FACT, alt=None):
    return RawFact(j, c, v, tag, "test", date(2026,1,1), alt)


# -----------------------------------------------------------------------
# SMAA Tests
# -----------------------------------------------------------------------

class TestSMAA:

    def test_rai_rows_sum_to_one(self):
        """For each jurisdiction, probabilities across all rank positions must sum to 1."""
        rai, _ = smaa_topsis(MATRIX, BENEFIT_MASK, VPL, DPL, n_trials=1000, seed=0)
        for jur in rai.index:
            assert abs(rai.loc[jur].sum() - 1.0) < 1e-9, \
                f"{jur} RAI row sums to {rai.loc[jur].sum():.6f}, expected 1.0"

    def test_rank1_acceptabilities_sum_to_one(self):
        """The probability of *someone* being rank 1 across all jurisdictions = 1.0."""
        rai, _ = smaa_topsis(MATRIX, BENEFIT_MASK, VPL, DPL, n_trials=1000, seed=0)
        assert abs(rai[1].sum() - 1.0) < 1e-9, \
            f"Rank-1 column sums to {rai[1].sum():.6f}, expected 1.0"

    def test_uae_most_likely_rank_1(self):
        """UAE should win rank 1 most often across the weight simplex (known result)."""
        rai, _ = smaa_topsis(MATRIX, BENEFIT_MASK, VPL, DPL, n_trials=5000, seed=42)
        assert rai.loc["UAE", 1] > rai.loc["Brazil", 1]
        assert rai.loc["UAE", 1] > rai.loc["Indonesia", 1]

    def test_dominated_market_has_low_rank1_acceptability(self):
        """Indonesia (lowest closeness) should rarely reach rank 1."""
        rai, _ = smaa_topsis(MATRIX, BENEFIT_MASK, VPL, DPL, n_trials=5000, seed=42)
        assert rai.loc["Indonesia", 1] < 0.30

    def test_central_weight_vectors_not_none_for_winners(self):
        """Any market that wins rank 1 at least once must have a central weight vector."""
        rai, cwv = smaa_topsis(MATRIX, BENEFIT_MASK, VPL, DPL, n_trials=1000, seed=42)
        for jur in rai.index:
            if rai.loc[jur, 1] > 0:
                assert cwv[jur] is not None, f"{jur} won rank 1 but has no central weight vector"
                assert abs(cwv[jur].sum() - 1.0) < 1e-9, \
                    f"{jur} CWV sums to {cwv[jur].sum():.6f}, expected 1.0"


# -----------------------------------------------------------------------
# Monte Carlo / perturb_score Tests
# -----------------------------------------------------------------------

class TestPerturbScore:
    """Unit tests for the per-cell noise function."""

    N = 10_000
    rng = np.random.default_rng(seed=0)

    def _draw_many(self, value, tag, alt=None):
        rng = np.random.default_rng(seed=1)
        return np.array([perturb_score(value, tag, 1.0, 10.0, rng, alt) for _ in range(self.N)])

    def test_verified_fact_tight_std(self):
        """VERIFIED_FACT noise must be much tighter than ANALYST_ESTIMATE."""
        std_vf = self._draw_many(7.0, EvidenceTag.VERIFIED_FACT).std()
        std_ae = self._draw_many(7.0, EvidenceTag.ANALYST_ESTIMATE).std()
        assert std_vf < std_ae, f"Expected std_vf ({std_vf:.3f}) < std_ae ({std_ae:.3f})"

    def test_analyst_estimate_wider_than_verified(self):
        std_ae = self._draw_many(7.0, EvidenceTag.ANALYST_ESTIMATE).std()
        assert std_ae > 0.3, f"ANALYST_ESTIMATE std too tight: {std_ae:.3f}"

    def test_unresolved_policy_is_bimodal(self):
        """
        An UNRESOLVED_POLICY score should produce a bimodal distribution
        when alt_value is far from value — verifiable by checking that the
        distribution has non-negligible density near BOTH outcomes.
        """
        draws = self._draw_many(8.0, EvidenceTag.UNRESOLVED_POLICY, alt=2.0)
        # Count draws close to each regime
        near_high = np.sum(draws > 6.0)
        near_low  = np.sum(draws < 4.0)
        assert near_high > self.N * 0.3, f"Too few draws near high regime: {near_high}"
        assert near_low  > self.N * 0.3, f"Too few draws near low regime: {near_low}"

    def test_all_draws_clipped_to_valid_range(self):
        """All perturbed values must stay in [1.0, 10.0]."""
        for tag in EvidenceTag:
            draws = self._draw_many(5.0, tag, alt=2.0)
            assert (draws >= 1.0).all() and (draws <= 10.0).all(), \
                f"Tag {tag}: draw out of [1, 10] range"


class TestMonteCarloRobustness:

    def _make_facts(self, tags: dict[str, EvidenceTag]) -> list[RawFact]:
        """Helper: build a complete fact list with specified evidence tags per criterion."""
        base_scores = {"C1": 0.3, "C2": 45000, "C3": 3, "C4": 200,
                       "C5": 2, "C6": 250000, "C7": 1, "C8": 2.0}
        facts = []
        for jur in ["UAE", "Brazil", "Indonesia"]:
            for c in CRITERIA:
                facts.append(make_fact(jur, c, base_scores[c], tags.get(c, EvidenceTag.VERIFIED_FACT)))
        return facts

    def test_verified_facts_have_smaller_std_than_estimates(self):
        """Markets evaluated with VERIFIED_FACT data should have tighter MC confidence bands."""
        facts_tight = self._make_facts({c: EvidenceTag.VERIFIED_FACT for c in CRITERIA})
        facts_wide  = self._make_facts({c: EvidenceTag.ANALYST_ESTIMATE for c in CRITERIA})

        mc_tight = monte_carlo_robustness(MATRIX, facts_tight, WEIGHTS, BENEFIT_MASK, VPL, DPL, n_trials=500, seed=0)
        mc_wide  = monte_carlo_robustness(MATRIX, facts_wide, WEIGHTS, BENEFIT_MASK, VPL, DPL, n_trials=500, seed=0)

        assert mc_tight["std"].mean() < mc_wide["std"].mean(), \
            "VERIFIED_FACT data should produce tighter MC intervals than ANALYST_ESTIMATE"

    def test_output_shape(self):
        """MC output must have one row per jurisdiction and p5/p50/p95/std columns."""
        facts = self._make_facts({c: EvidenceTag.VERIFIED_FACT for c in CRITERIA})
        mc = monte_carlo_robustness(MATRIX, facts, WEIGHTS, BENEFIT_MASK, VPL, DPL, n_trials=200, seed=0)
        assert mc.shape == (3, 4)
        assert list(mc.columns) == ["p5", "p50", "p95", "std"]

    def test_p5_le_p50_le_p95(self):
        """Percentiles must be monotonically ordered."""
        facts = self._make_facts({c: EvidenceTag.VERIFIED_FACT for c in CRITERIA})
        mc = monte_carlo_robustness(MATRIX, facts, WEIGHTS, BENEFIT_MASK, VPL, DPL, n_trials=200, seed=0)
        assert (mc["p5"] <= mc["p50"]).all()
        assert (mc["p50"] <= mc["p95"]).all()
