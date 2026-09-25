#!/usr/bin/env python
"""
tools/test_mcmpkp_and_dejure.py — Comprehensive Integration Runner & Validator

1. MCMPKP Optimizer: Tests multi-period wave allocation on top jurisdictions (UAE, Brazil, Japan, SGP, etc.)
2. De Jure LLM Pipeline: Tests JSON schema extraction & LLM-as-a-judge evidence tagging.
"""

import sys
from pathlib import Path
import pandas as pd

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking_engine.knapsack import solve_wave, solve_multi_period
from ranking_engine.dejure_pipeline import DeJureLLMPipeline, RegulatoryRuleUnit
from ranking_engine.schema import EvidenceTag, RawFact


def test_mcmpkp_optimizer():
    print("=" * 80)
    print("  STAGE 1: TESTING MULTIPLE-CHOICE MULTI-PERIOD KNAPSACK (MCMPKP)")
    print("=" * 80)

    # Candidates & TOPSIS Closeness Scores
    closeness = pd.Series({
        "United Arab Emirates": 0.842,
        "Brazil":               0.791,
        "Singapore":            0.764,
        "United Kingdom":       0.750,
        "Switzerland":          0.730,
        "United States":        0.710,
        "Japan":                0.680,
        "Australia":            0.650,
        "Indonesia":            0.520,
        "Vietnam":              0.210  # Hard capital knockout ($380M)
    })

    # Statutory Capital Burdens ($USD)
    capital_burden = {
        "United Arab Emirates": 1_200_000,
        "Brazil":                 850_000,
        "Singapore":            2_100_000,
        "United Kingdom":       1_800_000,
        "Switzerland":          2_500_000,
        "United States":        4_500_000,
        "Japan":                3_500_000,
        "Australia":            1_500_000,
        "Indonesia":            6_200_000,
        "Vietnam":            380_000_000
    }

    # Expected 12-Month Contribution Margin ($USD)
    margin_12mo = {
        "United Arab Emirates": 4_000_000,
        "Brazil":               3_500_000,
        "Singapore":            3_200_000,
        "United Kingdom":       3_000_000,
        "Switzerland":          2_200_000,
        "United States":        6_000_000,
        "Japan":                2_800_000,
        "Australia":            2_000_000,
        "Indonesia":            1_500_000,
        "Vietnam":              5_000_000
    }

    # Multiple-Choice Constraints (Mutual Exclusion)
    # E.g., Singapore vs Indonesia both serve as SE Asia Regional HQ — choose at most 1
    mutually_exclusive_groups = [
        ["Singapore", "Indonesia"]
    ]

    # Per-Wave Capital Allocations
    wave_budgets = [
        5_000_000,   # Wave 1 Budget = $5.0M
        6_000_000,   # Wave 2 Budget = $6.0M
        10_000_000   # Wave 3 Budget = $10.0M
    ]

    print(f"\n[PARAMS] Candidate Pool: {len(closeness)} markets")
    print(f"[PARAMS] Wave Budgets: {[f'${b:,.0f}' for b in wave_budgets]}")
    print(f"[PARAMS] Mutual Exclusion Groups: {mutually_exclusive_groups}\n")

    # Run Multi-Period Solver
    allocation = solve_multi_period(
        closeness=closeness,
        capital_burden=capital_burden,
        margin_12mo=margin_12mo,
        wave_budgets=wave_budgets,
        mutually_exclusive_groups=mutually_exclusive_groups,
        alpha=0.6
    )

    print("\n[RESULT] Sequential Expansion Wave Schedule:")
    total_spent = 0
    total_markets = 0

    for wave_idx, markets in allocation.items():
        wave_capital = sum(capital_burden[m] for m in markets)
        total_spent += wave_capital
        total_markets += len(markets)
        print(f"  ► WAVE {wave_idx + 1} (Budget: ${wave_budgets[wave_idx]:,.0f}):")
        print(f"      Selected Markets: {markets}")
        print(f"      Capital Spent:    ${wave_capital:,.0f} / ${wave_budgets[wave_idx]:,.0f} ({wave_capital/wave_budgets[wave_idx]*100:.1f}% efficiency)")
        for m in markets:
            print(f"        - {m:<22}: Closeness={closeness[m]:.3f}, Capital=${capital_burden[m]:,.0f}, 12mo Margin=${margin_12mo[m]:,.0f}")
        print()

    print(f"[TOTAL] Total Markets Scheduled Across Waves: {total_markets} | Total Capital Deployed: ${total_spent:,.0f}\n")

    # Assertions
    assert "Vietnam" not in [m for wave in allocation.values() for m in wave], "Vietnam should be knocked out!"
    # Ensure mutual exclusion rule holds
    for wave in allocation.values():
        assert not ("Singapore" in wave and "Indonesia" in wave), "Mutual exclusion group violated!"

    print("✓ MCMPKP Optimizer Verification Passed!\n")


def test_dejure_llm_pipeline():
    print("=" * 80)
    print("  STAGE 2: TESTING DE JURE LLM EXTRACTION & VERIFICATION PIPELINE")
    print("=" * 80)

    pipeline = DeJureLLMPipeline()

    # Sample Legal Text 1: VARA (UAE) Gazette
    legal_text_uae = """
    VARA Virtual Assets Regulatory Authority Rulebook 2024.
    Article 14.2: Any legal entity operating a Full Derivatives VASP or Proprietary Trading desk
    must maintain minimum paid-up statutory capital of AED 4,400,000.
    Retail leverage cap is fixed at 1:100 for licensed synthetic perps.
    Effective Date: 2026-01-01.
    """

    # Sample Legal Text 2: OJK Indonesia (Draft Policy / Unresolved)
    legal_text_idn_draft = """
    OJK Consultative Paper No. 12/2026 (PRAFT / DRAFT REGULATION).
    Proposed capital requirement for digital asset futures exchanges is set at IDR 100,000,000,000.
    Under alternative regime scenario B, this requirement may be waived down to IDR 40,000,000,000.
    Pending final parliamentary approval.
    """

    print("\n--- Processing Legal Document 1: UAE (VARA) ---")
    facts_uae = pipeline.extract_raw_facts(legal_text_uae, "United Arab Emirates")
    for f in facts_uae:
        print(f"  • [{f.criterion_code}] Raw={f.raw_value:<12} Tag={f.evidence_tag.value:<18} Citation: '{f.source_citation[:75]}...'")

    print("\n--- Processing Legal Document 2: Indonesia (OJK Draft) ---")
    facts_idn = pipeline.extract_raw_facts(legal_text_idn_draft, "Indonesia")
    for f in facts_idn:
        print(f"  • [{f.criterion_code}] Raw={f.raw_value:<12} Tag={f.evidence_tag.value:<18} AltRaw={f.alt_raw_value} Citation: '{f.source_citation[:75]}...'")

    # Verification assertions
    c6_uae = next(f for f in facts_uae if f.criterion_code == "C6")
    assert c6_uae.evidence_tag == EvidenceTag.VERIFIED_FACT
    assert abs(c6_uae.raw_value - (4400000 * 0.2723)) < 100  # AED -> USD check

    c6_idn = next(f for f in facts_idn if f.criterion_code == "C6")
    assert c6_idn.evidence_tag == EvidenceTag.UNRESOLVED_POLICY
    assert c6_idn.alt_raw_value is not None, "Bimodal alt_raw_value must be populated for UNRESOLVED_POLICY"

    print("\n✓ De Jure LLM Pipeline Extraction & Evidence Tagging Verification Passed!\n")


if __name__ == "__main__":
    test_mcmpkp_optimizer()
    test_dejure_llm_pipeline()
    print("=" * 80)
    print("  ALL INTEGRATION TESTS PASSED CLEANLY!")
    print("=" * 80)
