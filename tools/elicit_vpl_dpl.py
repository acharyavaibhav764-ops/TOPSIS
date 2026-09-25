#!/usr/bin/env python
"""
tools/elicit_vpl_dpl.py — Interactive CLI to elicit VPL and DPL from the management team.

Run:
    python tools/elicit_vpl_dpl.py
    python tools/elicit_vpl_dpl.py --output ranking_engine/config/vpl_dpl.json

For each of the 8 criteria the script displays:
  - The criterion name, domain, and what a low/high score means in plain language
  - The current placeholder value  (VPL=1.0, DPL=8.5)
  - Two prompts: minimum tolerable score (VPL) and aspirational cap (DPL)

Writes responses to a JSON file that PipelineConfig.from_files() can load directly.
Includes range-checking: VPL in [1, 5], DPL in (VPL, 10].
"""

import argparse
import json
from pathlib import Path

CRITERIA = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

CRITERION_DESCRIPTIONS = {
    "C1": {
        "name": "Demand Scale (HHI-adjusted exchange fragmentation)",
        "domain": "Commercial",
        "scale_lo": "score=1.0  →  HHI > 0.7: one exchange monopolises derivatives volume",
        "scale_hi": "score=10.0 →  HHI < 0.15: highly fragmented, contestable market",
        "vpl_q": "Below what score is a market commercially irrelevant for derivatives entry?",
        "dpl_q": "Above what score does extra fragmentation add no strategic value?",
    },
    "C2": {
        "name": "Revenue Depth (GNI per capita PPP)",
        "domain": "Commercial",
        "scale_lo": "score=1.0  →  GNI per capita < $1,000 (very low user spending power)",
        "scale_hi": "score=10.0 →  GNI per capita > $60,000 (UAE / Singapore tier)",
        "vpl_q": "Below what score is user spending power too low to support your product?",
        "dpl_q": "Above what score does extra wealth add no incremental value for your platform?",
    },
    "C3": {
        "name": "Pathway Clarity (explicit regulatory framework for retail derivatives)",
        "domain": "Regulatory",
        "scale_lo": "score=1.0  →  total ban or no legal framework whatsoever",
        "scale_hi": "score=10.0 →  explicit, in-force framework (e.g. VARA Part V in UAE)",
        "vpl_q": "What is the minimum regulatory clarity you require to consider a market at all?",
        "dpl_q": "Above what score is regulatory clarity no longer a differentiating factor?",
    },
    "C4": {
        "name": "Leverage Fit (statutory permitted retail leverage cap)",
        "domain": "Regulatory",
        "scale_lo": "score=1.0  →  1:5 cap or lower (incompatible with your product)",
        "scale_hi": "score=10.0 →  1:300 cap or higher (full product fit)",
        "vpl_q": "Below what score is the leverage cap incompatible with your core product?",
        "dpl_q": "Above what score does extra leverage headroom provide no additional value?",
    },
    "C5": {
        "name": "Local Rail Fit (fiat payment rail openness to VASPs)",
        "domain": "Friction",
        "scale_lo": "score=1.0  →  no open fiat rails; VASP banking widely refused",
        "scale_hi": "score=10.0 →  multiple instant fiat rails fully open to VASPs",
        "vpl_q": "Below what score is the payment environment too closed to operate?",
        "dpl_q": "Above what score is payment rail access no longer a bottleneck?",
    },
    "C6": {
        "name": "Capital Burden (inverted: higher score = lower required capital)",
        "domain": "Friction",
        "scale_lo": "score=1.0  →  high statutory paid-up capital (mapped from raw USD)",
        "scale_hi": "score=10.0 →  very low or zero mandatory capital requirement",
        "vpl_q": "Below what score (i.e. above what capital burden) is a market unaffordable?",
        "dpl_q": "Above what score does capital efficiency become irrelevant to the decision?",
    },
    "C7": {
        "name": "Presence Friction (inverted: higher score = fewer local presence mandates)",
        "domain": "Friction",
        "scale_lo": "score=1.0  →  full local office, resident directors, data localisation required",
        "scale_hi": "score=10.0 →  minimal or zero local presence requirements",
        "vpl_q": "Below what score are local presence requirements operationally prohibitive?",
        "dpl_q": "Above what score does extra presence flexibility add no value?",
    },
    "C8": {
        "name": "Regulatory Volatility (inverted: higher score = lower volatility)",
        "domain": "Sovereign Stability",
        "scale_lo": "score=1.0  →  frequent policy reversals, active draft laws (high uncertainty)",
        "scale_hi": "score=10.0 →  highly stable, predictable governance (WGI top tier)",
        "vpl_q": "Below what score is regulatory instability too high to commit capital?",
        "dpl_q": "Above what score does extra stability provide no additional confidence?",
    },
}


def _prompt_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        try:
            val = float(input(f"  {prompt} [{lo}–{hi}]: ").strip())
            if lo <= val <= hi:
                return val
            print(f"  → Must be between {lo} and {hi}. Try again.")
        except ValueError:
            print("  → Please enter a number.")


def elicit(output_path: str) -> None:
    results = {}
    print("\n" + "="*70)
    print("  VPL / DPL Elicitation Survey")
    print("  For each criterion:")
    print("    VPL (Vetoed Performance Level)  = minimum tolerable score [1–5]")
    print("    DPL (Desired Performance Level) = aspirational cap        (VPL–10]")
    print("="*70)

    for code in CRITERIA:
        desc = CRITERION_DESCRIPTIONS[code]
        print(f"\n{'─'*70}")
        print(f"  {code}: {desc['name']}")
        print(f"  Domain: {desc['domain']}")
        print(f"  Scale low:  {desc['scale_lo']}")
        print(f"  Scale high: {desc['scale_hi']}")
        print()
        print(f"  VPL question: {desc['vpl_q']}")
        vpl = _prompt_float("  VPL (min tolerable, placeholder=1.0)", 1.0, 5.0)

        print(f"\n  DPL question: {desc['dpl_q']}")
        dpl = _prompt_float(f"  DPL (aspirational cap, placeholder=8.5)", vpl + 0.1, 10.0)

        results[code] = {"vpl": vpl, "dpl": dpl}
        print(f"  ✓  {code}: VPL={vpl}, DPL={dpl}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ✓ Saved to {output_path}")
    print()
    print("  Load in pipeline:")
    print(f"    cfg = PipelineConfig.from_files(vpl_dpl='{output_path}')")
    print()
    print("  To re-run the survey: delete the file and run this script again.")
    print("  The file should NOT be committed to git (see ranking_engine/config/.gitignore).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elicit VPL/DPL values from the management team"
    )
    parser.add_argument(
        "--output",
        default="ranking_engine/config/vpl_dpl.json",
        help="Path to write the vpl_dpl.json config file",
    )
    args = parser.parse_args()
    elicit(args.output)
