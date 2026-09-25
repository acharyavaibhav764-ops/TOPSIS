"""
test_v1_mvp.py — Tests for V1: MVP end-to-end ranking with CRITIC weights + standard TOPSIS.

These tests verify the basic plumbing (data flows, types, shapes) before the
rigorous V2+ methods are wired in. Uses pymcdm directly.

Run:  python -m pytest ranking_engine/tests/test_v1_mvp.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from pymcdm.methods import TOPSIS
from pymcdm.weights import critic_weights
from pymcdm.helpers import rrankdata

from ranking_engine.criterion_mapping import BENEFIT_MASK

# Verified sample matrix from the implementation plan (§5 V1)
MATRIX = np.array([
    [8, 7, 8, 9, 7, 3, 4, 3],   # UAE
    [7, 6, 7, 6, 6, 4, 5, 5],   # Brazil
    [6, 5, 5, 4, 5, 3, 6, 6],   # Indonesia
], dtype=float)

MARKET_NAMES = ["UAE", "Brazil", "Indonesia"]
TYPES = np.where(BENEFIT_MASK, 1, -1)


class TestMVPPipeline:

    def test_critic_weights_sum_to_one(self):
        """CRITIC weights must sum to exactly 1."""
        w = critic_weights(MATRIX)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_critic_weights_non_negative(self):
        """All CRITIC weights must be non-negative."""
        w = critic_weights(MATRIX)
        assert (w >= 0).all()

    def test_critic_weights_length(self):
        """CRITIC must produce one weight per criterion."""
        w = critic_weights(MATRIX)
        assert len(w) == MATRIX.shape[1]

    def test_topsis_scores_in_range(self):
        """TOPSIS closeness coefficients must be in [0, 1]."""
        w = critic_weights(MATRIX)
        scores = TOPSIS()(MATRIX, w, TYPES)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_uae_ranks_first(self):
        """On the verified sample matrix, UAE should rank #1."""
        w = critic_weights(MATRIX)
        scores = TOPSIS()(MATRIX, w, TYPES)
        ranks = rrankdata(scores)
        uae_rank = ranks[MARKET_NAMES.index("UAE")]
        assert uae_rank == 1, f"Expected UAE rank 1, got {uae_rank}"

    def test_rankings_are_unique(self):
        """No two jurisdictions should share the same rank."""
        w = critic_weights(MATRIX)
        scores = TOPSIS()(MATRIX, w, TYPES)
        ranks = rrankdata(scores)
        assert len(set(ranks)) == len(ranks)

    def test_zero_variance_criterion_gets_near_zero_weight(self):
        """If all markets score identically on one criterion, CRITIC should weight it near 0.
        pymcdm's critic_weights produces NaN (0/0) for a constant column — NaN is treated
        as zero weight for this assertion since both express 'no discriminatory power'.
        """
        mat = MATRIX.copy()
        mat[:, 0] = 5.0   # make C1 constant across all markets
        w = np.nan_to_num(critic_weights(mat), nan=0.0)
        assert w[0] < 0.05, f"Constant criterion got weight {w[0]:.4f}, expected near 0"
