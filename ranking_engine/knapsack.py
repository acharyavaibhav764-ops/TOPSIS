"""
knapsack.py — V5: Multiple-Choice Multi-Period Knapsack (MCMPKP).

Concept
-------
A plain ordinal ranking (UAE #1, Brazil #2, ...) tempts you to expand in order.
That is wrong when capital is the binding constraint.

Counter-example:
  UAE ranks #1 but costs $4M in licensing/setup.
  Brazil ranks #2 at $2.5M.  Indonesia ranks #3 at $1.5M.
  Total wave-1 budget: $5M.

  Naive strategy: open UAE only → total TOPSIS value = 0.726
  Knapsack solution: open Brazil + Indonesia → total value = 0.619 + 0.440 = 1.059
  (you get more combined value AND have $1M left over for operations)

The knapsack maximizes the *total* strategic value across all markets selected
in one expansion wave, subject to:

  1. Budget constraint: Σ capital_burden[selected] ≤ wave_budget
  2. One-market-per-row: each market is either in or out (0/1 binary)
  3. Multiple-choice (mutual exclusion): you cannot pick two markets from
     a "competing hub" group (e.g., Indonesia AND Philippines both serve as
     a Southeast Asia regional HQ — pick exactly one).

Multi-period: call solve_wave() once per wave, removing chosen markets
from the pool before solving the next wave.

Dependency: pip install pulp==3.3.2  (PIN THIS VERSION — see plan §3)
"""

import pulp
import pandas as pd
import numpy as np


def solve_wave(
    closeness: pd.Series,
    capital_burden: dict[str, float],
    margin_12mo: dict[str, float],
    budget_usd: float,
    mutually_exclusive_groups: list[list[str]] | None = None,
    alpha: float = 0.6,
) -> tuple[list[str], float]:
    """
    Solve one capital-constrained expansion wave as a 0/1 MILP.

    Objective:  maximize Σ  (α·closeness_norm[m]  +  (1-α)·margin_norm[m]) · x[m]
    Subject to: Σ capital_burden[m] · x[m] ≤ budget_usd       (budget)
                Σ x[m] ≤ 1   for each mutual-exclusion group   (multiple-choice)
                x[m] ∈ {0, 1}

    The value function blends the TOPSIS closeness coefficient (strategic fit)
    with the 12-month expected margin (financial reality).  alpha controls the
    balance: 0.6 = 60% strategic weight, 40% financial.

    Parameters
    ----------
    closeness   : pd.Series, index = market names, values = TOPSIS-RAD closeness [0,1]
    capital_burden : dict market → capital required (USD, matching budget_usd units)
    margin_12mo    : dict market → expected 12-month contribution margin (same currency)
    budget_usd     : capital available for this wave
    mutually_exclusive_groups : e.g. [["Indonesia", "Philippines"]] means pick at most one
    alpha          : weight on TOPSIS closeness vs. 12-month margin

    Returns
    -------
    selected : list of market names chosen for this wave
    obj_value: float — the achieved objective (for logging / comparison)
    """
    markets = closeness.index.tolist()

    # Normalize values to [0,1] per dimension so alpha blends apples-to-apples
    max_c = max(closeness.values)
    max_g = max(margin_12mo[m] for m in markets) if margin_12mo else 1.0

    c_norm = {m: closeness[m] / max_c  if max_c > 0 else 0.0 for m in markets}
    g_norm = {m: margin_12mo[m] / max_g if max_g > 0 else 0.0 for m in markets}
    value  = {m: alpha * c_norm[m] + (1.0 - alpha) * g_norm[m] for m in markets}

    # --- PuLP model ---
    prob = pulp.LpProblem("expansion_wave", pulp.LpMaximize)
    x = {m: pulp.LpVariable(f"x_{m}", cat="Binary") for m in markets}

    # Objective
    prob += pulp.lpSum(value[m] * x[m] for m in markets)

    # Budget constraint
    prob += pulp.lpSum(capital_burden[m] * x[m] for m in markets) <= budget_usd, "budget"

    # Mutual-exclusion constraints (Multiple-Choice)
    for i, group in enumerate(mutually_exclusive_groups or []):
        prob += pulp.lpSum(x[m] for m in group if m in x) <= 1, f"mutex_{i}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if prob.status != 1:
        raise RuntimeError(f"Knapsack MILP did not find an optimal solution (status={prob.status})")

    selected = [m for m in markets if pulp.value(x[m]) == 1.0]
    return selected, float(pulp.value(prob.objective))


def solve_multi_period(
    closeness: pd.Series,
    capital_burden: dict[str, float],
    margin_12mo: dict[str, float],
    wave_budgets: list[float],
    mutually_exclusive_groups: list[list[str]] | None = None,
    alpha: float = 0.6,
) -> dict[int, list[str]]:
    """
    Solve multiple sequential expansion waves.

    Markets selected in wave W are removed from the pool before solving wave W+1.
    Budget does NOT carry over (each wave has its own capital allocation).

    Parameters
    ----------
    wave_budgets : list of per-wave budgets, e.g. [8_000_000, 5_000_000]
                   (Wave 1 = 8M, Wave 2 = 5M)

    Returns
    -------
    dict: {wave_index (0-based) → list of selected market names}
    """
    remaining_closeness = closeness.copy()
    allocation: dict[int, list[str]] = {}

    for wave_idx, budget in enumerate(wave_budgets):
        if remaining_closeness.empty:
            allocation[wave_idx] = []
            continue

        selected, obj = solve_wave(
            closeness=remaining_closeness,
            capital_burden={m: capital_burden[m] for m in remaining_closeness.index},
            margin_12mo={m: margin_12mo[m] for m in remaining_closeness.index},
            budget_usd=budget,
            mutually_exclusive_groups=mutually_exclusive_groups,
            alpha=alpha,
        )

        allocation[wave_idx] = selected
        print(
            f"[WAVE {wave_idx + 1}] budget=${budget:,.0f}  "
            f"selected={selected}  obj={obj:.4f}"
        )

        # Remove selected markets from the pool for subsequent waves
        remaining_closeness = remaining_closeness.drop(index=selected, errors="ignore")

    return allocation
