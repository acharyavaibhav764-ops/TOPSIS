"""
test_v5_knapsack.py — Tests for V5: Multiple-Choice Multi-Period Knapsack.

Run:  python -m pytest ranking_engine/tests/test_v5_knapsack.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import pandas as pd

from ranking_engine.knapsack import solve_wave, solve_multi_period

# Verified example from the implementation plan (§10)
MARKETS = ["UAE", "Brazil", "Indonesia", "Vietnam", "Philippines"]

CLOSENESS = pd.Series({
    "UAE":         0.726,
    "Brazil":      0.619,
    "Indonesia":   0.440,
    "Vietnam":     0.710,   # high closeness but absurd capital requirement
    "Philippines": 0.500,
})

CAPITAL = {
    "UAE":         4_000_000,
    "Brazil":      2_500_000,
    "Indonesia":   1_500_000,
    "Vietnam":   380_000_000,   # makes it self-selecting out
    "Philippines": 2_000_000,
}

MARGIN = {
    "UAE":         3_000_000,
    "Brazil":      2_000_000,
    "Indonesia":   1_200_000,
    "Vietnam":    50_000_000,
    "Philippines": 1_400_000,
}


class TestSolveWave:

    def test_budget_never_exceeded(self):
        """Total capital of selected markets must not exceed the wave budget."""
        selected, _ = solve_wave(CLOSENESS, CAPITAL, MARGIN, budget_usd=8_000_000)
        total_capital = sum(CAPITAL[m] for m in selected)
        assert total_capital <= 8_000_000, \
            f"Budget exceeded: selected {selected}, total capital ${total_capital:,}"

    def test_vietnam_excluded_by_budget(self):
        """Vietnam's $380M requirement makes it impossible to select under any realistic budget."""
        selected, _ = solve_wave(CLOSENESS, CAPITAL, MARGIN, budget_usd=8_000_000)
        assert "Vietnam" not in selected, \
            f"Vietnam should not fit in an $8M wave but was selected: {selected}"

    def test_mutual_exclusion_respected(self):
        """Indonesia and Philippines must never both appear when they are mutually exclusive."""
        selected, _ = solve_wave(
            CLOSENESS, CAPITAL, MARGIN,
            budget_usd=10_000_000,
            mutually_exclusive_groups=[["Indonesia", "Philippines"]]
        )
        both_present = "Indonesia" in selected and "Philippines" in selected
        assert not both_present, \
            f"Mutual exclusion violated: both Indonesia and Philippines were selected: {selected}"

    def test_verified_example_from_plan(self):
        """
        The verified example from §10 of the plan:
        Budget=$8M, Indonesia/Philippines mutually exclusive.
        Expected: UAE + Brazil + Indonesia (or UAE + Brazil + Philippines — solver may pick either).
        UAE must be selected (highest combined value and fits in budget).
        """
        selected, obj = solve_wave(
            CLOSENESS, CAPITAL, MARGIN,
            budget_usd=8_000_000,
            mutually_exclusive_groups=[["Indonesia", "Philippines"]]
        )
        assert "UAE" in selected, f"UAE should be selected; got {selected}"
        assert "Brazil" in selected, f"Brazil should be selected; got {selected}"
        assert len(selected) == 3, f"Expected 3 markets; got {len(selected)}: {selected}"

    def test_empty_budget_returns_empty(self):
        """A budget of $0 should result in no markets selected."""
        selected, obj = solve_wave(CLOSENESS, CAPITAL, MARGIN, budget_usd=0)
        assert selected == [], f"Expected empty selection with $0 budget, got {selected}"

    def test_objective_is_non_negative(self):
        """Objective value from a feasible solution must be ≥ 0."""
        _, obj = solve_wave(CLOSENESS, CAPITAL, MARGIN, budget_usd=8_000_000)
        assert obj >= 0.0, f"Objective is negative: {obj}"


class TestSolveMultiPeriod:

    def test_no_market_selected_twice(self):
        """Once a market is chosen in Wave 1, it must not appear in Wave 2."""
        wave_budgets = [8_000_000, 5_000_000]
        allocation = solve_multi_period(CLOSENESS, CAPITAL, MARGIN, wave_budgets)

        wave1 = set(allocation.get(0, []))
        wave2 = set(allocation.get(1, []))
        overlap = wave1 & wave2
        assert not overlap, f"Markets appear in multiple waves: {overlap}"

    def test_each_wave_respects_its_budget(self):
        """Budget constraint applies independently to each wave."""
        wave_budgets = [8_000_000, 3_000_000]
        allocation = solve_multi_period(CLOSENESS, CAPITAL, MARGIN, wave_budgets)

        for wave_idx, budget in enumerate(wave_budgets):
            selected = allocation.get(wave_idx, [])
            total = sum(CAPITAL[m] for m in selected)
            assert total <= budget, \
                f"Wave {wave_idx+1} budget ${budget:,} exceeded: selected {selected} = ${total:,}"

    def test_mutual_exclusion_across_waves(self):
        """The mutual exclusion constraint should hold within each wave, not across waves."""
        wave_budgets = [5_000_000, 5_000_000]
        allocation = solve_multi_period(
            CLOSENESS, CAPITAL, MARGIN, wave_budgets,
            mutually_exclusive_groups=[["Indonesia", "Philippines"]]
        )
        # Within any single wave, the exclusive pair must not co-occur
        for wave_idx, selected in allocation.items():
            both = "Indonesia" in selected and "Philippines" in selected
            assert not both, \
                f"Wave {wave_idx+1}: mutual exclusion violated for Indonesia/Philippines"
