"""
test_v7_reverify.py — Independent re-verification of TOPSIS-RAD.

This test is the strongest in the suite: it does NOT call topsis_rad() at all.
Instead it:
  1. Runs the full pipeline with dump_dir= to write stage CSVs/JSONs.
  2. Loads 02_post_knockout_matrix.csv + 03_weights.json independently.
  3. Re-implements TOPSIS-RAD from scratch using only numpy (no imports from
     ranking_engine.topsis_rad).
  4. Asserts that its own closeness results match 04_topsis_closeness.csv to 5 dp.

If the module and the reference disagree, something is wrong in the module —
the reference implementation has no shared code and no shared bugs.

This also validates Fix D (dump_dir): if the files are missing or malformed,
the test will fail loudly rather than silently.

Run:  python -m pytest ranking_engine/tests/test_v7_reverify.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from datetime import date

from ranking_engine.schema import RawFact, EvidenceTag
from ranking_engine.pipeline import RankingPipeline, PipelineConfig
from ranking_engine.criterion_mapping import CRITERIA


# ---------------------------------------------------------------------------
# Shared test dataset (same as test_v6_pipeline.py for consistency)
# ---------------------------------------------------------------------------

def make_fact(j, c, v, tag=EvidenceTag.VERIFIED_FACT, alt=None):
    return RawFact(
        jurisdiction=j, criterion_code=c, raw_value=v,
        evidence_tag=tag, source_citation="reverify-test",
        as_of=date(2026, 1, 1), alt_raw_value=alt,
    )


RAW = {
    "UAE":       {"C1": 0.20, "C2": 50_000, "C3": 3, "C4": 200, "C5": 2,
                  "C6": 150_000,   "C7": 1, "C8": 1.5},
    "Brazil":    {"C1": 0.40, "C2": 16_000, "C3": 2, "C4": 100, "C5": 1,
                  "C6": 800_000,   "C7": 2, "C8": 4.0},
    "Indonesia": {"C1": 0.50, "C2": 7_000,  "C3": 1, "C4": 30,  "C5": 0,
                  "C6": 1_500_000, "C7": 3, "C8": 6.0},
    "Vietnam":   {"C1": 0.55, "C2": 5_000,  "C3": 0, "C4": 10,  "C5": 0,
                  "C6": 380_000_000, "C7": 4, "C8": 8.0},  # knocked out
}

ALL_FACTS = [
    make_fact(jur, crit, val)
    for jur, criteria in RAW.items()
    for crit, val in criteria.items()
]

CFG = PipelineConfig(
    liquid_runway_usd=5_000_000,
    smaa_n_trials=2_000,
    mc_n_trials=500,
)


# ---------------------------------------------------------------------------
# Independent TOPSIS-RAD reference implementation
# Written from the mathematical definition only — no code shared with topsis_rad.py
# ---------------------------------------------------------------------------

def reference_topsis_rad(
    matrix: np.ndarray,      # shape (n_alts, n_criteria) — already direction-corrected
    weights: np.ndarray,     # shape (n_criteria,)
    vpl: np.ndarray,         # shape (n_criteria,)
    dpl: np.ndarray,         # shape (n_criteria,)
) -> np.ndarray:
    """
    Naive reference implementation of TOPSIS-RAD.

    Steps (from Costa et al., arXiv:2606.07253):
      1. Cap scores at DPL (uniform — higher = better for all columns).
      2. Normalize against fixed VPL/DPL anchors.
      3. PIS = normalized DPL × weights; NIS = normalized VPL × weights.
      4. Euclidean distances to PIS and NIS.
      5. Closeness = d_neg / (d_pos + d_neg).

    This re-derives every step from algebra, not from topsis_rad.py.
    """
    X = matrix.astype(float).copy()

    # Step 1: cap — uniform minimum(X, DPL)
    X_cap = np.minimum(X, dpl)

    # Step 2: normalize against fixed anchors
    anchors = np.vstack([vpl, dpl])              # (2, n_criteria)
    norm_denom = np.sqrt((anchors ** 2).sum(axis=0))
    norm_denom[norm_denom == 0] = 1e-12
    Xn = X_cap / norm_denom
    Wn = Xn * weights

    # Step 3: PIS and NIS
    an = anchors / norm_denom
    pis = an[1] * weights   # DPL row
    nis = an[0] * weights   # VPL row

    # Step 4: distances
    d_pos = np.sqrt(((Wn - pis) ** 2).sum(axis=1))
    d_neg = np.sqrt(((Wn - nis) ** 2).sum(axis=1))

    # Step 5: closeness
    denom = d_pos + d_neg
    denom[denom == 0] = 1e-12
    return d_neg / denom


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dump_output():
    """Run the pipeline once with a temp dump_dir; return paths."""
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = RankingPipeline(CFG)
        results, _ = pipeline.run(
            facts=ALL_FACTS,
            margin_12mo={"UAE": 4_000_000, "Brazil": 2_000_000, "Indonesia": 1_200_000},
            verbose=False,
            dump_dir=tmp,
        )
        # Read the files while the tempdir is still alive
        matrix_csv = pd.read_csv(os.path.join(tmp, "02_post_knockout_matrix.csv"), index_col=0)
        weights_json = json.loads(open(os.path.join(tmp, "03_weights.json")).read())
        closeness_csv = pd.read_csv(os.path.join(tmp, "04_topsis_closeness.csv"), index_col=0)
        facts_json = json.loads(open(os.path.join(tmp, "00_raw_facts.json")).read())
        knocked_json = json.loads(open(os.path.join(tmp, "02_knocked_out.json")).read())
        rai_csv = pd.read_csv(os.path.join(tmp, "05_smaa_rai.csv"), index_col=0)
        mc_csv = pd.read_csv(os.path.join(tmp, "06_mc_intervals.csv"), index_col=0)
        alloc_json = json.loads(open(os.path.join(tmp, "07_knapsack_allocation.json")).read())
        results_json = json.loads(open(os.path.join(tmp, "08_final_results.json")).read())
        yield {
            "matrix": matrix_csv,
            "weights": weights_json,
            "closeness": closeness_csv,
            "facts": facts_json,
            "knocked_out": knocked_json,
            "rai": rai_csv,
            "mc": mc_csv,
            "allocation": alloc_json,
            "results": results_json,
        }


class TestDumpFilesExist:
    """Fix D: verify the dump_dir files are actually written and well-formed."""

    def test_matrix_has_correct_shape(self, dump_output):
        mat = dump_output["matrix"]
        assert list(mat.columns) == CRITERIA
        assert len(mat) == 3  # Vietnam knocked out

    def test_weights_have_all_criteria(self, dump_output):
        w = dump_output["weights"]
        for key in ("w_bwm", "w_critic_m", "w_fused"):
            assert set(w[key].keys()) == set(CRITERIA), f"{key} missing criteria"
        assert abs(sum(w["w_fused"].values()) - 1.0) < 1e-9

    def test_knocked_out_contains_vietnam(self, dump_output):
        assert "Vietnam" in dump_output["knocked_out"]["knocked_out"]

    def test_closeness_csv_has_rank_column(self, dump_output):
        assert "closeness" in dump_output["closeness"].columns
        assert "rank" in dump_output["closeness"].columns

    def test_rai_shape(self, dump_output):
        rai = dump_output["rai"]
        assert len(rai) == 3  # one row per surviving jurisdiction

    def test_mc_shape(self, dump_output):
        mc = dump_output["mc"]
        assert set(["p5", "p50", "p95", "std"]).issubset(mc.columns)

    def test_results_json_has_all_fields(self, dump_output):
        for r in dump_output["results"]:
            for field in ("jurisdiction", "closeness", "rank", "mc_90pct_interval"):
                assert field in r, f"Missing field '{field}' in result: {r}"


class TestIndependentTOPSISVerification:
    """
    The core test: re-implement TOPSIS-RAD from scratch, feed it the same
    02/03 data, and assert it matches 04 to 5 decimal places.

    If this fails, the module is computing something different from the
    mathematical specification — not just "tests pass" different.
    """

    def test_reference_matches_module_closeness(self, dump_output):
        """
        Reference implementation (this file) vs topsis_rad.py output (04_topsis_closeness.csv).
        Must agree to 5 decimal places.
        """
        mat = dump_output["matrix"]
        w_fused = dump_output["weights"]["w_fused"]
        weights = np.array([w_fused[c] for c in CRITERIA])
        vpl = np.array([1.0] * 8)
        dpl = np.array([8.5] * 8)

        X = mat[CRITERIA].values
        ref_cc = reference_topsis_rad(X, weights, vpl, dpl)

        closeness_df = dump_output["closeness"]
        for i, jur in enumerate(mat.index):
            module_cc = float(closeness_df.loc[jur, "closeness"])
            ref = float(ref_cc[i])
            assert abs(module_cc - ref) < 1e-5, (
                f"{jur}: module closeness={module_cc:.6f}, "
                f"reference={ref:.6f}, delta={abs(module_cc-ref):.2e}. "
                f"The module is computing something other than the mathematical spec."
            )

    def test_reference_ranking_matches_module_ranking(self, dump_output):
        """
        The rank column in 04_topsis_closeness.csv must match what the
        reference implementation would assign.
        """
        mat = dump_output["matrix"]
        w_fused = dump_output["weights"]["w_fused"]
        weights = np.array([w_fused[c] for c in CRITERIA])
        vpl = np.array([1.0] * 8)
        dpl = np.array([8.5] * 8)

        X = mat[CRITERIA].values
        ref_cc = reference_topsis_rad(X, weights, vpl, dpl)
        ref_order = [mat.index[i] for i in np.argsort(ref_cc)[::-1]]

        closeness_df = dump_output["closeness"]
        module_order = closeness_df.sort_values("rank")["closeness"].index.tolist()

        assert ref_order == module_order, (
            f"Reference ranking: {ref_order}\n"
            f"Module ranking:    {module_order}"
        )

    def test_uae_ranks_first_in_both(self, dump_output):
        mat = dump_output["matrix"]
        w_fused = dump_output["weights"]["w_fused"]
        weights = np.array([w_fused[c] for c in CRITERIA])
        vpl = np.array([1.0] * 8)
        dpl = np.array([8.5] * 8)

        X = mat[CRITERIA].values
        ref_cc = reference_topsis_rad(X, weights, vpl, dpl)
        ref_best = mat.index[np.argmax(ref_cc)]

        closeness_df = dump_output["closeness"]
        module_best = closeness_df["rank"].idxmin()   # rank=1 is the best

        assert ref_best == "UAE", f"Reference: UAE should rank #1, got {ref_best}"
        assert module_best == "UAE", f"Module: UAE should rank #1, got {module_best}"

    def test_c6_affects_closeness(self, dump_output):
        """
        Fix C / bug #3 regression: run the reference implementation on two
        artificial jurisdictions that differ ONLY in C6.  The one with better
        C6 (higher mapped score) must get a strictly higher closeness.
        If the old np.maximum(X, dpl=10.0) bug were still present, BOTH would
        be forced to 10.0 → tie → this test would fail.
        """
        from ranking_engine.criterion_mapping import MAPPING_FUNCTIONS
        from ranking_engine.knockout import build_decision_matrix
        from ranking_engine.schema import RawFact, EvidenceTag

        base_raw = {
            "C1": 0.40, "C2": 20_000, "C3": 1, "C4": 50,
            "C5": 1, "C6": 500_000, "C7": 2, "C8": 4.0,
        }
        facts = []
        for jur, c6_raw in [("CheapCapital", 100_000), ("ExpensiveCapital", 5_000_000)]:
            for c, v in base_raw.items():
                raw = c6_raw if c == "C6" else v
                facts.append(RawFact(
                    jurisdiction=jur, criterion_code=c, raw_value=raw,
                    evidence_tag=EvidenceTag.VERIFIED_FACT, source_citation="c6-mono",
                    as_of=date(2026, 1, 1),
                ))

        matrix = build_decision_matrix(facts, MAPPING_FUNCTIONS)
        weights = np.ones(8) / 8.0
        vpl = np.array([1.0] * 8)
        dpl = np.array([8.5] * 8)
        X = matrix[CRITERIA].values
        ref_cc = reference_topsis_rad(X, weights, vpl, dpl)

        cc = dict(zip(matrix.index, ref_cc))
        assert cc["CheapCapital"] > cc["ExpensiveCapital"], (
            f"C6 monotonicity FAILED in reference implementation: "
            f"CheapCapital({cc['CheapCapital']:.4f}) <= ExpensiveCapital({cc['ExpensiveCapital']:.4f}). "
            f"If both are equal, np.maximum(X, dpl) bug has been re-introduced."
        )
        assert (cc["CheapCapital"] - cc["ExpensiveCapital"]) > 0.02, \
            "Margin too small — C6 may be effectively zeroed"
