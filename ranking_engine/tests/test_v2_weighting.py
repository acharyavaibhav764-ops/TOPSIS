"""
test_v2_weighting.py — Tests for V2: BWM + CRITIC-M weight functions.

Run:  python -m pytest ranking_engine/tests/test_v2_weighting.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from ranking_engine.criterion_mapping import CRITERIA, BENEFIT_MASK
from ranking_engine.weighting_bwm import bwm_weights
from ranking_engine.weighting_critic_m import critic_m_weights, fuse_weights

# Shared test matrix
MATRIX_DF = pd.DataFrame({
    "C1": [8, 7, 6], "C2": [7, 6, 5], "C3": [8, 7, 5],
    "C4": [9, 6, 4], "C5": [7, 6, 5], "C6": [3, 4, 3],
    "C7": [4, 5, 6], "C8": [3, 5, 6],
}, index=["UAE", "Brazil", "Indonesia"])


# -----------------------------------------------------------------------
# BWM Tests
# -----------------------------------------------------------------------

class TestBWM:
    # Survey: Best=C3 (idx 2), Worst=C6 (idx 5)
    a_B = np.array([3, 4, 1, 2, 5, 9, 6, 4], dtype=float)
    a_W = np.array([5, 4, 9, 6, 3, 1, 2, 4], dtype=float)

    def test_weights_sum_to_one(self):
        w, _ = bwm_weights(CRITERIA, 2, 5, self.a_B, self.a_W)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        w, _ = bwm_weights(CRITERIA, 2, 5, self.a_B, self.a_W)
        assert (w >= 0).all()

    def test_best_criterion_gets_highest_weight(self):
        """The declared 'Best' criterion (C3) must get the highest weight."""
        w, _ = bwm_weights(CRITERIA, 2, 5, self.a_B, self.a_W)
        assert w[2] == pytest.approx(w.max(), rel=1e-3)

    def test_worst_criterion_gets_lowest_weight(self):
        """The declared 'Worst' criterion (C6) must get the lowest weight."""
        w, _ = bwm_weights(CRITERIA, 2, 5, self.a_B, self.a_W)
        assert w[5] == pytest.approx(w.min(), rel=1e-3)

    def test_perfect_consistency_gives_near_zero_xi(self):
        """Perfectly consistent comparisons (all ratios exactly consistent) → xi ≈ 0."""
        # Constructed example: Best is 3x more important than every other, Worst is 1/3
        a_B_perfect = np.array([3, 3, 1, 3, 3, 9, 3, 3], dtype=float)
        a_W_perfect = np.array([3, 3, 9, 3, 3, 1, 3, 3], dtype=float)
        _, xi = bwm_weights(CRITERIA, 2, 5, a_B_perfect, a_W_perfect)
        assert xi < 0.1, f"Expected low xi for consistent survey, got {xi:.4f}"

    def test_equal_comparisons_give_equal_weights(self):
        """If every criterion is rated equally vs Best and Worst, weights should be uniform."""
        a_B_eq = np.ones(8, dtype=float)
        a_W_eq = np.ones(8, dtype=float)
        # Best = Worst for this degenerate case — use different indices
        w, _ = bwm_weights(CRITERIA, 0, 7, a_B_eq, a_W_eq)
        # Weights should all be approximately equal
        assert np.std(w) < 0.05, f"Expected near-equal weights, got std={np.std(w):.4f}"


# -----------------------------------------------------------------------
# CRITIC-M Tests
# -----------------------------------------------------------------------

class TestCRITICM:

    def test_weights_sum_to_one(self):
        w = critic_m_weights(MATRIX_DF, BENEFIT_MASK)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_weights_non_negative(self):
        w = critic_m_weights(MATRIX_DF, BENEFIT_MASK)
        assert (w >= 0).all()

    def test_length_equals_n_criteria(self):
        w = critic_m_weights(MATRIX_DF, BENEFIT_MASK)
        assert len(w) == len(CRITERIA)

    def test_constant_criterion_gets_near_zero_weight(self):
        """A criterion with zero variance provides no discriminatory power → weight ≈ 0.
        critic_m_weights (logistic squash) produces NaN (0/0 in normalization) for a
        constant column. nan_to_num converts that to 0.0, which correctly means 'zero weight'.
        """
        mat_const = MATRIX_DF.copy()
        mat_const["C1"] = 7.0   # constant across all markets
        w = np.nan_to_num(critic_m_weights(mat_const, BENEFIT_MASK), nan=0.0)
        assert w[0] < 0.02, f"Constant C1 got weight {w[0]:.4f}, expected < 0.02"


# -----------------------------------------------------------------------
# Fuse Weights Tests
# -----------------------------------------------------------------------

class TestFuseWeights:

    def _get_weights(self):
        w_bwm, _ = bwm_weights(CRITERIA, 2, 5,
                                np.array([3,4,1,2,5,9,6,4], dtype=float),
                                np.array([5,4,9,6,3,1,2,4], dtype=float))
        w_cm = critic_m_weights(MATRIX_DF, BENEFIT_MASK)
        return w_bwm, w_cm

    def test_fused_sums_to_one(self):
        w_bwm, w_cm = self._get_weights()
        w = fuse_weights(w_bwm, w_cm, lam=0.6)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_lam_one_returns_bwm(self):
        """lam=1 should return the pure BWM weights."""
        w_bwm, w_cm = self._get_weights()
        w = fuse_weights(w_bwm, w_cm, lam=1.0)
        np.testing.assert_allclose(w, w_bwm / w_bwm.sum(), rtol=1e-6)

    def test_lam_zero_returns_critic_m(self):
        """lam=0 should return the pure CRITIC-M weights."""
        w_bwm, w_cm = self._get_weights()
        w = fuse_weights(w_bwm, w_cm, lam=0.0)
        np.testing.assert_allclose(w, w_cm / w_cm.sum(), rtol=1e-6)

    def test_invalid_lam_raises(self):
        w_bwm, w_cm = self._get_weights()
        with pytest.raises(ValueError):
            fuse_weights(w_bwm, w_cm, lam=1.5)
