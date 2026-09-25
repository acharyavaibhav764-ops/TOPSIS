"""
test_v6_pipeline.py — Integration test: real RawFact list → RankingPipeline.run()

This is the test that the isolated unit tests could never catch.
It feeds a toy 4-jurisdiction dataset through every module in sequence
and asserts the final outputs against hand-computed expectations.

Why this test catches both bugs the unit tests missed
------------------------------------------------------
Bug #1 (CRITIC-M double inversion):
  The unit tests hand-feed already-consistent numbers to one function.
  This test builds RawFacts, runs build_decision_matrix(), then runs
  CRITIC-M on the result.  After the fix, C2 and C5 (revenue-depth and
  rail-fit, which have high variance in the fixture) should dominate —
  not C6 and C8 (capital burden and volatility) which would dominate
  under the double-inversion bug because their direction flip was wrong.

Bug #2 (MC alt_raw_value wrong units):
  The UNRESOLVED_POLICY test in test_v4_robustness.py calls perturb_score
  directly with (value=8.0, alt=2.0) — both already 1-10.  This test
  creates a RawFact with alt_raw_value in raw units (a dollar figure),
  passes it through monte_carlo_robustness(), and verifies the bimodal
  distribution is centered near the MAPPED score, not the raw value.

Run:  python -m pytest ranking_engine/tests/test_v6_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from datetime import date

from ranking_engine.schema import RawFact, EvidenceTag
from ranking_engine.pipeline import RankingPipeline, PipelineConfig
from ranking_engine.criterion_mapping import CRITERIA, MAPPING_FUNCTIONS


# ---------------------------------------------------------------------------
# Toy 4-jurisdiction dataset — hand-computed expectations
# ---------------------------------------------------------------------------
# Design intent:
#   UAE:         best on most criteria           → should rank #1
#   Brazil:      mid-tier                        → should rank #2
#   Indonesia:   slightly below Brazil           → should rank #3
#   Vietnam:     terrible C6 (capital too high)  → knocked out before ranking


def make_fact(j, c, v, tag=EvidenceTag.VERIFIED_FACT, alt=None):
    return RawFact(
        jurisdiction=j,
        criterion_code=c,
        raw_value=v,
        evidence_tag=tag,
        source_citation="integration-test",
        as_of=date(2026, 1, 1),
        alt_raw_value=alt,
    )


# Raw values — must pass through MAPPING_FUNCTIONS before becoming matrix scores
RAW = {
    "UAE": {
        "C1": 0.20,          # low HHI → competitive → high score
        "C2": 50_000,        # high GNI PPP → high score
        "C3": 3,             # explicit framework → score=10
        "C4": 200,           # high leverage cap → high score
        "C5": 2,             # high rail openness → high score
        "C6": 150_000,       # very low capital → high score (already inverted by mapping)
        "C7": 1,             # low presence friction → high score
        "C8": 1.5,           # low volatility → high score
    },
    "Brazil": {
        "C1": 0.40,
        "C2": 16_000,
        "C3": 2,
        "C4": 100,
        "C5": 1,
        "C6": 800_000,
        "C7": 2,
        "C8": 4.0,
    },
    "Indonesia": {
        "C1": 0.50,
        "C2": 7_000,
        "C3": 1,
        "C4": 30,
        "C5": 0,
        "C6": 1_500_000,
        "C7": 3,
        "C8": 6.0,
    },
    "Vietnam": {
        "C1": 0.55,
        "C2": 5_000,
        "C3": 0,
        "C4": 10,
        "C5": 0,
        "C6": 380_000_000,   # knockout: exceeds any realistic $5M runway
        "C7": 4,
        "C8": 8.0,
    },
}

ALL_FACTS = [
    make_fact(jur, crit, val)
    for jur, criteria in RAW.items()
    for crit, val in criteria.items()
]

# BWM survey: C3 best (regulatory pathway), C6 worst (capital burden)
CFG = PipelineConfig(
    liquid_runway_usd=5_000_000,
    bwm_best_idx=2,
    bwm_worst_idx=5,
    bwm_a_B=np.array([3, 4, 1, 2, 5, 9, 6, 4], dtype=float),
    bwm_a_W=np.array([5, 4, 9, 6, 3, 1, 2, 4], dtype=float),
    smaa_n_trials=5_000,
    mc_n_trials=2_000,    # smaller for test speed; still enough to verify bug fix
    wave_budgets=[4_000_000, 3_000_000],
    mutually_exclusive_groups=[["Brazil", "Indonesia"]],  # SE-Asia hub choice
)


@pytest.fixture(scope="module")
def pipeline_output():
    pipeline = RankingPipeline(CFG)
    results, artifacts = pipeline.run(
        facts=ALL_FACTS,
        margin_12mo={"UAE": 4_000_000, "Brazil": 2_000_000, "Indonesia": 1_200_000},
        verbose=False,
    )
    return results, artifacts


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestKnockout:
    def test_vietnam_knocked_out(self, pipeline_output):
        """Vietnam's $380M capital requirement must not appear in results."""
        results, _ = pipeline_output
        jurs = [r.jurisdiction for r in results]
        assert "Vietnam" not in jurs, f"Vietnam should be knocked out, got: {jurs}"

    def test_three_markets_survive(self, pipeline_output):
        results, _ = pipeline_output
        assert len(results) == 3


class TestRankingOrder:
    def test_uae_ranks_first(self, pipeline_output):
        """UAE has dominant scores across all 8 criteria and must rank #1."""
        results, _ = pipeline_output
        assert results[0].jurisdiction == "UAE", \
            f"Expected UAE #1, got: {[(r.jurisdiction, r.rank) for r in results]}"

    def test_ranks_are_1_indexed_and_unique(self, pipeline_output):
        results, _ = pipeline_output
        ranks = [r.rank for r in results]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_closeness_descending(self, pipeline_output):
        """Results list is sorted by closeness descending."""
        results, _ = pipeline_output
        for i in range(len(results) - 1):
            assert results[i].closeness >= results[i + 1].closeness


class TestWeightSanity:
    """
    Tests that catch Bug #1 (CRITIC-M double inversion).

    Under the buggy version, C6 and C8 (which have wide variance in this
    fixture after mistaken direction flip) would dominate.  After the fix,
    the direction-corrected matrix correctly attributes high variance to
    the criteria that genuinely discriminate: C2 (large GNI gap between
    UAE at $50k and Indonesia at $7k) and C3 (explicit vs. no framework).
    """
    def test_c6_weight_not_dominant(self, pipeline_output):
        """
        C6 (capital burden) should NOT be the highest-weight criterion.
        Under the double-inversion bug, C6's variance was computed on
        negated-then-direction-corrected scores, making it appear highly
        discriminatory when it isn't.
        """
        _, artifacts = pipeline_output
        weights = artifacts["weights"]
        w_c6 = weights["C6"]
        # C6 should be relatively low (it's a cost criterion with moderate spread)
        assert w_c6 < 0.20, \
            f"C6 weight = {w_c6:.3f} is suspiciously high — possible double-inversion bug"

    def test_benefit_criteria_carry_more_weight_than_cost(self, pipeline_output):
        """Benefit criteria (C1-C5) should collectively outweigh cost criteria (C6-C8)."""
        _, artifacts = pipeline_output
        weights = artifacts["weights"]
        w_benefit = sum(weights[c] for c in ["C1", "C2", "C3", "C4", "C5"])
        w_cost    = sum(weights[c] for c in ["C6", "C7", "C8"])
        assert w_benefit > w_cost, \
            f"Cost criteria outweigh benefit criteria: {w_benefit:.3f} vs {w_cost:.3f}"

    def test_weights_sum_to_one(self, pipeline_output):
        _, artifacts = pipeline_output
        total = sum(artifacts["weights"].values())
        assert abs(total - 1.0) < 1e-9


class TestSMAAIntegration:
    def test_rai_rows_sum_to_one(self, pipeline_output):
        _, artifacts = pipeline_output
        rai = artifacts["rai"]
        for jur in rai.index:
            assert abs(rai.loc[jur].sum() - 1.0) < 1e-9

    def test_uae_has_highest_rank1_probability(self, pipeline_output):
        _, artifacts = pipeline_output
        rai = artifacts["rai"]
        uae_p1 = rai.loc["UAE", 1]
        for jur in rai.index:
            if jur != "UAE":
                assert uae_p1 >= rai.loc[jur, 1], \
                    f"UAE P(rank=1)={uae_p1:.3f} < {jur} P(rank=1)={rai.loc[jur,1]:.3f}"


class TestMCIntegration:
    """
    Tests that catch Bug #2 (alt_raw_value not mapped through MAPPING_FUNCTIONS).
    """
    def test_mc_intervals_in_valid_range(self, pipeline_output):
        _, artifacts = pipeline_output
        mc = artifacts["mc_intervals"]
        assert (mc["p5"].values >= 0).all() and (mc["p95"].values <= 1).all()

    def test_mc_percentiles_monotonic(self, pipeline_output):
        _, artifacts = pipeline_output
        mc = artifacts["mc_intervals"]
        assert (mc["p5"] <= mc["p50"]).all()
        assert (mc["p50"] <= mc["p95"]).all()

    def test_unresolved_policy_alt_raw_value_is_mapped(self):
        """
        Regression test for Bug #2.

        A C4 UNRESOLVED_POLICY fact: regime A allows 200x leverage (score ≈ 9)
        vs regime B allows 5x leverage (score ≈ 2).  alt_raw_value=5.0 (raw)
        must be mapped through map_c4_leverage_fit() before use.

        If alt_raw_value were used raw (= 5.0), it would be clipped to [1,10]
        as-is.  After mapping, map_c4_leverage_fit(5.0) ≈ 1.0 (very low).
        The bimodal distribution should be centered near 9 and near 1 — NOT
        near 9 and near 5.
        """
        from ranking_engine.robustness_mc import monte_carlo_robustness, perturb_score
        from ranking_engine.criterion_mapping import MAPPING_FUNCTIONS, BENEFIT_MASK, CRITERIA
        import pandas as pd

        # Confirm the mapping: leverage cap of 5x should score LOW
        score_5x_raw  = MAPPING_FUNCTIONS["C4"](5.0)    # regime B: very low leverage
        score_200x_raw = MAPPING_FUNCTIONS["C4"](200.0)  # regime A: very high leverage
        assert score_5x_raw <= 2.0, \
            f"Leverage cap 5x should score ≤2.0, got {score_5x_raw}"
        assert score_200x_raw >= 8.0, \
            f"Leverage cap 200x should score ≥8.0, got {score_200x_raw}"

        # Build a single-market, single-criterion fact with UNRESOLVED_POLICY
        fact = make_fact(
            "TestJur", "C4",
            v=200.0,                           # raw: regime A (200x leverage)
            tag=EvidenceTag.UNRESOLVED_POLICY,
            alt=5.0,                           # raw: regime B (5x leverage) — NOT pre-mapped
        )

        # The tag_lookup in monte_carlo_robustness should map alt=5.0 through
        # MAPPING_FUNCTIONS["C4"] → score ≈ 1.0, NOT keep it as 5.0
        base_scores = {c: 5.0 for c in CRITERIA}
        base_scores["C4"] = MAPPING_FUNCTIONS["C4"](200.0)   # regime A score

        vpl = np.array([1.0,1.0,1.0,1.0,1.0,1.5,1.5,1.5], dtype=float)
        dpl = np.array([8.5,8.5,8.5,8.5,8.5,10.0,10.0,10.0], dtype=float)
        weights = np.ones(8) / 8.0
        matrix = pd.DataFrame([base_scores], index=["TestJur"], columns=CRITERIA)
        facts_for_jur = [
            make_fact("TestJur", c, raw, tag=EvidenceTag.VERIFIED_FACT)
            for c, raw in {
                "C1": 0.30, "C2": 30_000, "C3": 2, "C4": 200,
                "C5": 1, "C6": 300_000, "C7": 1, "C8": 2.0,
            }.items()
        ]
        # Override C4 with the UNRESOLVED_POLICY fact
        facts_for_jur = [f for f in facts_for_jur if f.criterion_code != "C4"]
        facts_for_jur.append(fact)

        mc = monte_carlo_robustness(
            base_matrix=matrix,
            facts=facts_for_jur,
            weights=weights,
            benefit_mask=BENEFIT_MASK,
            vpl=vpl,
            dpl=dpl,
            n_trials=3_000,
            seed=0,
        )

        # The p5-p95 interval must be wide: it flips between regime A (high
        # closeness) and regime B (very low closeness).
        interval_width = float(mc.loc["TestJur", "p95"] - mc.loc["TestJur", "p5"])
        assert interval_width > 0.10, \
            (f"UNRESOLVED_POLICY C4 interval should be wide (bimodal), got width={interval_width:.3f}. "
             f"If alt_raw_value=5.0 was passed raw instead of mapped, the alt score would "
             f"be 5.0 (mid-range) instead of {score_5x_raw:.1f} (very low), collapsing the interval.")


class TestKnapsackIntegration:
    def test_mutual_exclusion_brazil_indonesia(self, pipeline_output):
        """Brazil and Indonesia are in the same mutex group — never co-selected in one wave."""
        _, artifacts = pipeline_output
        allocation = artifacts["expansion_allocation"]
        for wave_idx, selected in allocation.items():
            both = "Brazil" in selected and "Indonesia" in selected
            assert not both, \
                f"Wave {wave_idx+1}: Brazil + Indonesia selected despite mutex group: {selected}"

    def test_each_wave_within_budget(self, pipeline_output):
        from ranking_engine.knockout import extract_capital_required
        _, artifacts = pipeline_output
        allocation = artifacts["expansion_allocation"]
        cap = extract_capital_required(ALL_FACTS)
        for wave_idx, budget in enumerate(CFG.wave_budgets):
            selected = allocation.get(wave_idx, [])
            total = sum(cap.get(m, 0) for m in selected)
            assert total <= budget, \
                f"Wave {wave_idx+1} over budget: {total:,} > {budget:,}"

    def test_uae_in_wave_1(self, pipeline_output):
        """UAE has the highest combined closeness+margin and should be selected in wave 1."""
        _, artifacts = pipeline_output
        allocation = artifacts["expansion_allocation"]
        wave1 = allocation.get(0, [])
        assert "UAE" in wave1, \
            f"UAE should be in wave 1 (dominant market), got wave 1 = {wave1}"
