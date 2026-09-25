"""
test_v8_persistence.py — Tests for ResultsStore and staleness detection.

What's verified here (per the review's standing rule):
  - test_save_and_retrieve: weights table has NON-NULL w_bwm / w_critic_m  (Fix #3 check)
  - test_vpl_change_triggers_rerun: a VPL-only config change must cause needs_rerun=True (Fix #2 check)
  - test_dpl_change_triggers_rerun: same for DPL (Fix #2 check)
  - test_surveys_change_triggers_rerun: adding a survey must register as config change (Fix #2 check)
  - test_same_config_no_rerun: identical run → no re-run
  - test_newer_facts_trigger_rerun: new as_of date → re-run
  - test_force_always_reruns: force=True → always re-runs
  - test_jurisdiction_history_oldest_first: two runs → history sorted oldest-first
  - test_latest_results_current: latest_results() reflects the most recent run

Run:  python -m pytest ranking_engine/tests/test_v8_persistence.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pytest
import numpy as np
from datetime import date, datetime

from ranking_engine.schema import RawFact, EvidenceTag, StakeholderSurvey
from ranking_engine.pipeline import RankingPipeline, PipelineConfig
from ranking_engine.results_store import ResultsStore
from ranking_engine.staleness import needs_rerun, config_hash


# ---------------------------------------------------------------------------
# Shared fixture — minimal dataset (same jurisdictions as other test files)
# ---------------------------------------------------------------------------

def _fact(j, c, v, tag=EvidenceTag.VERIFIED_FACT, as_of=date(2026, 1, 1)):
    return RawFact(
        jurisdiction=j, criterion_code=c, raw_value=v,
        evidence_tag=tag, source_citation="t-v8",
        as_of=as_of,
    )


_RAW = {
    "UAE":       {"C1": 0.20, "C2": 50_000, "C3": 3, "C4": 200, "C5": 2,
                  "C6": 150_000,   "C7": 1, "C8": 1.5},
    "Brazil":    {"C1": 0.40, "C2": 16_000, "C3": 2, "C4": 100, "C5": 1,
                  "C6": 800_000,   "C7": 2, "C8": 4.0},
    "Indonesia": {"C1": 0.50, "C2": 7_000,  "C3": 1, "C4": 30,  "C5": 0,
                  "C6": 1_500_000, "C7": 3, "C8": 6.0},
}

ALL_FACTS = [
    _fact(jur, c, v)
    for jur, crit in _RAW.items()
    for c, v in crit.items()
]

_CFG_FAST = PipelineConfig(
    liquid_runway_usd=5_000_000,
    smaa_n_trials=500,
    mc_n_trials=200,
)


def _run_pipeline(facts=None, cfg=None):
    f = facts or ALL_FACTS
    c = cfg   or _CFG_FAST
    return RankingPipeline(c).run(f, verbose=False)


# ---------------------------------------------------------------------------
# ResultsStore tests
# ---------------------------------------------------------------------------

class TestResultsStore:

    def test_save_and_retrieve_returns_correct_jurisdiction_count(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        results, artifacts = _run_pipeline()
        store.save_run(ALL_FACTS, results, artifacts, _CFG_FAST)
        latest = store.latest_results()
        assert len(latest) == 3
        store.close()

    def test_weights_table_has_non_null_w_bwm_and_w_cm(self, tmp_path):
        """
        Fix #3 regression check: if w_bwm/w_cm keys are missing from artifacts,
        every row in the weights table would have NULL — this test will catch it.
        """
        store = ResultsStore(str(tmp_path / "r.db"))
        results, artifacts = _run_pipeline()

        # Confirm the keys actually exist in artifacts (they must after Fix #3)
        assert "w_bwm" in artifacts, "artifacts missing 'w_bwm' key — Fix #3 not applied"
        assert "w_cm"  in artifacts, "artifacts missing 'w_cm' key — Fix #3 not applied"

        run_id = store.save_run(ALL_FACTS, results, artifacts, _CFG_FAST)
        weights = store.weights_for_run(run_id)
        assert len(weights) == 8, f"Expected 8 weight rows, got {len(weights)}"
        for row in weights:
            assert row["w_bwm"]     is not None, f"{row['criterion']}: w_bwm is NULL"
            assert row["w_critic_m"] is not None, f"{row['criterion']}: w_critic_m is NULL"
            assert row["w_fused"]   is not None, f"{row['criterion']}: w_fused is NULL"
        store.close()

    def test_jurisdiction_history_oldest_first(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))

        # Run 1
        results1, arts1 = _run_pipeline()
        store.save_run(ALL_FACTS, results1, arts1, _CFG_FAST)

        # Run 2 with different config (lambda shifted) so config hash differs
        cfg2 = PipelineConfig(liquid_runway_usd=5_000_000, smaa_n_trials=500,
                              mc_n_trials=200, weight_lambda=0.7)
        results2, arts2 = _run_pipeline(cfg=cfg2)
        store.save_run(ALL_FACTS, results2, arts2, cfg2)

        history = store.jurisdiction_history("UAE")
        assert len(history) == 2
        assert history[0]["run_at"] <= history[1]["run_at"], "History not sorted oldest-first"
        store.close()

    def test_latest_results_returns_most_recent_run(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))

        results1, arts1 = _run_pipeline()
        store.save_run(ALL_FACTS, results1, arts1, _CFG_FAST)

        # Second run with shifted lambda
        cfg2 = PipelineConfig(liquid_runway_usd=5_000_000, smaa_n_trials=500,
                              mc_n_trials=200, weight_lambda=0.3)
        results2, arts2 = _run_pipeline(cfg=cfg2)
        store.save_run(ALL_FACTS, results2, arts2, cfg2)

        latest = store.latest_results()
        # All latest results should belong to the same (second) run_id
        run_ids = {r["run_id"] for r in latest}
        assert len(run_ids) == 1
        store.close()

    def test_empty_store_returns_none_for_latest_run_id(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        assert store.latest_run_id() is None
        assert store.latest_results() == []
        store.close()


# ---------------------------------------------------------------------------
# config_hash tests — verifying Fix #2 specifically
# ---------------------------------------------------------------------------

class TestConfigHash:

    def test_identical_config_same_hash(self):
        cfg_a = PipelineConfig(liquid_runway_usd=5_000_000)
        cfg_b = PipelineConfig(liquid_runway_usd=5_000_000)
        assert config_hash(cfg_a) == config_hash(cfg_b)

    def test_scalar_change_changes_hash(self):
        cfg_a = PipelineConfig(weight_lambda=0.6)
        cfg_b = PipelineConfig(weight_lambda=0.7)
        assert config_hash(cfg_a) != config_hash(cfg_b)

    def test_vpl_change_changes_hash(self):
        """
        Fix #2 regression: changing VPL (ndarray) must produce a different hash.
        The original scalar-only filter would have dropped vpl entirely,
        making cfg_a and cfg_b hash-equal despite different VPL values.
        """
        cfg_a = PipelineConfig(vpl=np.array([1.0]*8))
        cfg_b = PipelineConfig(vpl=np.array([2.0]*8))
        assert config_hash(cfg_a) != config_hash(cfg_b), (
            "VPL change did NOT change config hash — Fix #2 is not applied correctly. "
            "config_hash() is silently dropping the vpl ndarray field."
        )

    def test_dpl_change_changes_hash(self):
        """Fix #2 regression: DPL change must change the hash."""
        cfg_a = PipelineConfig(dpl=np.array([8.5]*8))
        cfg_b = PipelineConfig(dpl=np.array([7.0]*8))
        assert config_hash(cfg_a) != config_hash(cfg_b), (
            "DPL change did NOT change config hash — Fix #2 not applied for dpl."
        )

    def test_survey_addition_changes_hash(self):
        """Fix #2 regression: adding a StakeholderSurvey must change the hash."""
        cfg_no_surveys = PipelineConfig()
        survey = StakeholderSurvey(
            best_idx=2, worst_idx=5,
            a_B=[3,4,1,2,5,9,6,4], a_W=[5,4,9,6,3,1,2,4],
            dm_id="founder_1", submitted=datetime(2026, 9, 25),
        )
        cfg_with_survey = PipelineConfig(surveys=[survey])
        assert config_hash(cfg_no_surveys) != config_hash(cfg_with_survey), (
            "Adding a survey did NOT change config hash — Fix #2 not applied for surveys."
        )


# ---------------------------------------------------------------------------
# needs_rerun tests
# ---------------------------------------------------------------------------

class TestStalenessChecker:

    def test_empty_store_always_needs_rerun(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        should, reason = needs_rerun(store, ALL_FACTS, config_hash(_CFG_FAST))
        assert should is True
        assert "no previous run" in reason
        store.close()

    def test_force_always_needs_rerun(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        results, arts = _run_pipeline()
        store.save_run(ALL_FACTS, results, arts, _CFG_FAST)
        should, reason = needs_rerun(store, ALL_FACTS, config_hash(_CFG_FAST), force=True)
        assert should is True
        assert "forced" in reason
        store.close()

    def test_same_facts_and_config_no_rerun(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        results, arts = _run_pipeline()
        store.save_run(ALL_FACTS, results, arts, _CFG_FAST)
        should, reason = needs_rerun(store, ALL_FACTS, config_hash(_CFG_FAST))
        assert should is False, f"Unexpected re-run triggered: {reason}"
        store.close()

    def test_newer_facts_trigger_rerun(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        results, arts = _run_pipeline()
        store.save_run(ALL_FACTS, results, arts, _CFG_FAST)

        # Create facts with a future as_of date
        future_fact = _fact("UAE", "C1", 0.20, as_of=date(2030, 1, 1))
        updated_facts = ALL_FACTS + [future_fact]

        should, reason = needs_rerun(store, updated_facts, config_hash(_CFG_FAST))
        assert should is True
        assert "new facts dated" in reason
        store.close()

    def test_new_jurisdiction_triggers_rerun(self, tmp_path):
        store = ResultsStore(str(tmp_path / "r.db"))
        results, arts = _run_pipeline()
        store.save_run(ALL_FACTS, results, arts, _CFG_FAST)

        # Add a new jurisdiction to the facts set
        new_facts = ALL_FACTS + [
            _fact("Singapore", c, v)
            for c, v in {"C1": 0.30, "C2": 60_000, "C3": 3, "C4": 100, "C5": 2,
                         "C6": 200_000, "C7": 2, "C8": 2.0}.items()
        ]
        should, reason = needs_rerun(store, new_facts, config_hash(_CFG_FAST))
        assert should is True
        assert "Singapore" in reason
        store.close()

    def test_vpl_change_triggers_rerun(self, tmp_path):
        """
        Fix #2 integration test: changing VPL must cause needs_rerun=True.
        If config_hash() was still using the scalar-only filter, this would
        pass (False) because the changed vpl ndarray would be silently dropped.
        """
        store = ResultsStore(str(tmp_path / "r.db"))
        results, arts = _run_pipeline()
        store.save_run(ALL_FACTS, results, arts, _CFG_FAST)

        # Same facts, same scalars, different VPL
        cfg_new_vpl = PipelineConfig(
            liquid_runway_usd=5_000_000,
            smaa_n_trials=500, mc_n_trials=200,
            vpl=np.array([2.0]*8),   # changed from [1.0]*8
        )
        should, reason = needs_rerun(store, ALL_FACTS, config_hash(cfg_new_vpl))
        assert should is True, (
            "VPL change did NOT trigger re-run — Fix #2 in config_hash() is not working. "
            "The staleness checker is silently ignoring VPL changes."
        )
        store.close()

    def test_dpl_change_triggers_rerun(self, tmp_path):
        """Fix #2 integration test: changing DPL must cause needs_rerun=True."""
        store = ResultsStore(str(tmp_path / "r.db"))
        results, arts = _run_pipeline()
        store.save_run(ALL_FACTS, results, arts, _CFG_FAST)

        cfg_new_dpl = PipelineConfig(
            liquid_runway_usd=5_000_000,
            smaa_n_trials=500, mc_n_trials=200,
            dpl=np.array([7.0]*8),   # changed from [8.5]*8
        )
        should, reason = needs_rerun(store, ALL_FACTS, config_hash(cfg_new_dpl))
        assert should is True, (
            "DPL change did NOT trigger re-run — Fix #2 in config_hash() is not working."
        )
        store.close()
