"""
test_v0_knockout.py — Tests for V0: Decision matrix assembly + hard knockout screen.

Run:  python -m pytest ranking_engine/tests/test_v0_knockout.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import numpy as np
import pandas as pd
from datetime import date

from ranking_engine.schema import RawFact, EvidenceTag
from ranking_engine.criterion_mapping import CRITERIA, MAPPING_FUNCTIONS, map_c4_leverage_fit
from ranking_engine.knockout import build_decision_matrix, knockout_screen, extract_capital_required


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_fact(jurisdiction, criterion, value, tag=EvidenceTag.VERIFIED_FACT, alt=None):
    return RawFact(
        jurisdiction=jurisdiction,
        criterion_code=criterion,
        raw_value=value,
        evidence_tag=tag,
        source_citation="test",
        as_of=date(2026, 1, 1),
        alt_raw_value=alt,
    )


UAE_FACTS = [
    make_fact("UAE", "C1", 0.3),      # HHI → score ≈ 7
    make_fact("UAE", "C2", 45_000),   # GNI PPP → score = 8
    make_fact("UAE", "C3", 3),        # Explicit framework → score = 10
    make_fact("UAE", "C4", 200),      # 200x leverage cap → score = 9
    make_fact("UAE", "C5", 2),        # High rail openness → score = 10
    make_fact("UAE", "C6", 250_000),  # $250k capital → score = 8 (cost)
    make_fact("UAE", "C7", 1),        # 1 presence mandate → score = 8
    make_fact("UAE", "C8", 2.0),      # volatility index 2 → score = 8
]

VIETNAM_FACTS = [
    make_fact("Vietnam", "C1", 0.5),
    make_fact("Vietnam", "C2", 4_000),
    make_fact("Vietnam", "C3", 0),
    make_fact("Vietnam", "C4", 5),
    make_fact("Vietnam", "C5", 0),
    make_fact("Vietnam", "C6", 380_000_000),  # ~$380M — must be knocked out
    make_fact("Vietnam", "C7", 4),
    make_fact("Vietnam", "C8", 8.0),
]

ALL_FACTS = UAE_FACTS + VIETNAM_FACTS


# ---------------------------------------------------------------------------
# Tests: criterion mapping functions
# ---------------------------------------------------------------------------

class TestCriterionMapping:

    def test_leverage_low_cap_scores_low(self):
        """A 1:5 cap should score low (≤ 2)."""
        assert map_c4_leverage_fit(5.0) <= 2.0

    def test_leverage_high_cap_scores_high(self):
        """A 1:300 cap should score maximum (10)."""
        assert map_c4_leverage_fit(300.0) == 10.0

    def test_leverage_mid_cap_scores_mid(self):
        """A 1:100 cap is clearly mid-range."""
        score = map_c4_leverage_fit(100.0)
        assert 7.0 < score < 10.0

    def test_all_mapping_functions_return_valid_range(self):
        """Every mapping function should return a value in [1.0, 10.0]."""
        sample_inputs = {
            "C1": 0.5, "C2": 30_000, "C3": 2, "C4": 50,
            "C5": 1, "C6": 500_000, "C7": 2, "C8": 5.0,
        }
        for code, val in sample_inputs.items():
            score = MAPPING_FUNCTIONS[code](val)
            assert 1.0 <= score <= 10.0, f"{code}({val}) = {score} out of [1, 10]"


# ---------------------------------------------------------------------------
# Tests: build_decision_matrix
# ---------------------------------------------------------------------------

class TestBuildDecisionMatrix:

    def test_shape(self):
        """Matrix should have one row per jurisdiction and 8 criterion columns."""
        mat = build_decision_matrix(ALL_FACTS)
        assert mat.shape == (2, 8)

    def test_column_order(self):
        """Columns must follow canonical CRITERIA order C1..C8."""
        mat = build_decision_matrix(ALL_FACTS)
        assert list(mat.columns) == CRITERIA

    def test_all_scores_in_range(self):
        """All score values should be in [1.0, 10.0]."""
        mat = build_decision_matrix(ALL_FACTS)
        assert (mat.values >= 1.0).all() and (mat.values <= 10.0).all()

    def test_no_nan_when_all_facts_present(self):
        """With a complete fact list, no NaN cells should appear."""
        mat = build_decision_matrix(ALL_FACTS)
        assert not mat.isnull().any().any()


# ---------------------------------------------------------------------------
# Tests: knockout_screen
# ---------------------------------------------------------------------------

class TestKnockoutScreen:

    def test_vietnam_knocked_out(self):
        """Vietnam's $380M requirement must be removed for any realistic runway."""
        mat = build_decision_matrix(ALL_FACTS)
        cap = extract_capital_required(ALL_FACTS)
        filtered = knockout_screen(mat, cap, liquid_runway_usd=5_000_000)
        assert "Vietnam" not in filtered.index

    def test_uae_survives(self):
        """UAE's $250k requirement must survive under a $5M runway."""
        mat = build_decision_matrix(ALL_FACTS)
        cap = extract_capital_required(ALL_FACTS)
        filtered = knockout_screen(mat, cap, liquid_runway_usd=5_000_000)
        assert "UAE" in filtered.index

    def test_boundary_exactly_at_runway(self):
        """A market with capital requirement exactly equal to runway must survive."""
        facts = UAE_FACTS + [make_fact("Edge", c, 5.0) for c in CRITERIA if c != "C6"]
        facts.append(make_fact("Edge", "C6", 5_000_000))  # exactly at limit
        mat = build_decision_matrix(facts)
        cap = extract_capital_required(facts)
        filtered = knockout_screen(mat, cap, liquid_runway_usd=5_000_000)
        assert "Edge" in filtered.index

    def test_one_above_runway_dropped(self):
        """A market requiring $1 more than the runway must be dropped."""
        facts = UAE_FACTS + [make_fact("OverEdge", c, 5.0) for c in CRITERIA if c != "C6"]
        facts.append(make_fact("OverEdge", "C6", 5_000_001))
        mat = build_decision_matrix(facts)
        cap = extract_capital_required(facts)
        filtered = knockout_screen(mat, cap, liquid_runway_usd=5_000_000)
        assert "OverEdge" not in filtered.index
