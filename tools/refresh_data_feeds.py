#!/usr/bin/env python
"""
tools/refresh_data_feeds.py — Tier-4 data feed refresh + pipeline re-run trigger.

Run on a schedule (cron, Airflow, Windows Task Scheduler, etc.).
Fetches all Tier-4 feeds, writes/merges RawFacts into the facts JSON file,
then calls run_pipeline.py's staleness check so the pipeline automatically
re-runs when the data has actually changed.

How the staleness hook works (vs the plan's mark_dirty() that doesn't exist)
-----------------------------------------------------------------------------
staleness.py's needs_rerun() already checks whether any RawFact has an as_of
date newer than the last pipeline run (Check 1). Since this script produces
facts with today's date (or the indicator's publication year) as as_of,
any genuinely new data will automatically trigger a re-run when run_pipeline.py
is next invoked. There is no separate mark_dirty() call needed — the
fact-set change IS the dirty signal.

Operational model:
  1. Cron calls: python tools/refresh_data_feeds.py --facts data/facts.json
  2. Script fetches, scores, merges facts into data/facts.json
  3. Cron (or a second step) calls: python tools/run_pipeline.py --facts data/facts.json --db ...
     → staleness check detects newer facts → re-runs pipeline automatically

Usage:
  python tools/refresh_data_feeds.py --facts data/facts.json
  python tools/refresh_data_feeds.py --facts data/facts.json --dry-run
  python tools/refresh_data_feeds.py --facts data/facts.json --skip-hhi   # if CC is down

Environment:
  CRYPTOCOMPARE_API_KEY — optional, for higher CC rate limits
  MARKETSPHERE_CACHE_DIR — override cache directory (default: .cache/data_feeds/)
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_feeds import worldbank_client, exchange_hhi_client, governance_index_client, scoring
from data_feeds.raw_fact_adapter import (
    c1_from_hhi, c2_from_worldbank, c5_base_signal, c8_from_wgi,
)
from ranking_engine.schema import RawFact, EvidenceTag


# ---------------------------------------------------------------------------
# Jurisdiction registry
# (jurisdiction name → ISO-3 for World Bank, ISO-2 for CryptoCompare)
# ---------------------------------------------------------------------------

JURISDICTIONS: list[dict] = [
    {"name": "UAE",       "iso3": "ARE", "cc2": "AE"},
    {"name": "Brazil",    "iso3": "BRA", "cc2": "BR"},
    {"name": "Indonesia", "iso3": "IDN", "cc2": "ID"},
    # Add more as the target list expands:
    # {"name": "Vietnam",   "iso3": "VNM", "cc2": "VN"},
    # {"name": "Singapore", "iso3": "SGP", "cc2": "SG"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rawfact_to_dict(f: RawFact) -> dict:
    """Serialize a RawFact to a JSON-compatible dict (for facts.json)."""
    return {
        "jurisdiction":    f.jurisdiction,
        "criterion_code":  f.criterion_code,
        "raw_value":       f.raw_value,
        "evidence_tag":    f.evidence_tag.value,
        "source_citation": f.source_citation,
        "as_of":           f.as_of.isoformat(),
        "alt_raw_value":   f.alt_raw_value,
    }


def _load_existing_facts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _merge_facts(existing: list[dict], new_facts: list[RawFact]) -> list[dict]:
    """
    Merge new Tier-4 facts into the existing fact list.
    Strategy: new facts for (jurisdiction, criterion_code) replace old ones.
    Manual facts for C3/C4/C6/C7 (Tier 3 criteria) are preserved unchanged.
    """
    TIER4_CRITERIA = {"C1", "C2", "C5", "C8"}

    # Index existing facts that are NOT Tier-4 (keep all C3/C4/C6/C7 intact)
    preserved = [
        f for f in existing
        if f.get("criterion_code") not in TIER4_CRITERIA
    ]

    # Add all new Tier-4 facts
    new_dicts = [_rawfact_to_dict(f) for f in new_facts if f is not None]

    return preserved + new_dicts


# ---------------------------------------------------------------------------
# Per-criterion refresh functions
# ---------------------------------------------------------------------------

def refresh_c2_c5(jur: dict, verbose: bool) -> tuple[list[RawFact], list[str]]:
    """Fetch C2 (Revenue Depth) and C5 base signal from World Bank."""
    facts, warnings = [], []
    iso3 = jur["iso3"]
    name = jur["name"]

    try:
        inputs = worldbank_client.get_revenue_depth_inputs(iso3)
    except RuntimeError as e:
        warnings.append(f"[{name}] World Bank C2 fetch failed: {e}")
        return facts, warnings

    c2_score = scoring.score_revenue_depth(
        inputs["gni_ppp"], inputs["account_ownership_pct"]
    )
    if c2_score is not None:
        facts.append(c2_from_worldbank(
            name, c2_score, inputs["gni_year"], inputs["findex_year"]
        ))
        if verbose:
            print(f"  [{name}] C2 = {c2_score:.2f}  "
                  f"(GNI={inputs['gni_ppp']}, year={inputs['gni_year']})")
    else:
        warnings.append(f"[{name}] C2: no World Bank data found")

    # C5 base signal reuses the Findex value already fetched
    c5_score = scoring.score_c5_base_signal(inputs["account_ownership_pct"])
    if c5_score is not None:
        facts.append(c5_base_signal(name, inputs["account_ownership_pct"], inputs["findex_year"]))
        if verbose:
            print(f"  [{name}] C5 (base) = {c5_score:.2f}  "
                  f"(Findex={inputs['account_ownership_pct']}%, year={inputs['findex_year']})")
    else:
        warnings.append(f"[{name}] C5: no Findex data found (VASP-openness still manual)")

    return facts, warnings


def refresh_c1(jur: dict, verbose: bool, api_key: str | None) -> tuple[list[RawFact], list[str]]:
    """Compute HHI from CryptoCompare exchange volume for C1."""
    facts, warnings = [], []
    name = jur["name"]
    cc2  = jur["cc2"]

    try:
        hhi_result = exchange_hhi_client.compute_hhi(jur["iso3"], api_key=api_key)
    except RuntimeError as e:
        warnings.append(f"[{name}] CryptoCompare C1 fetch failed: {e}")
        return facts, warnings

    if hhi_result["hhi"] is None:
        warnings.append(f"[{name}] C1: no exchanges found for country code '{cc2}'")
        return facts, warnings

    if hhi_result["low_coverage"]:
        warnings.append(
            f"[{name}] C1: low exchange coverage (n={hhi_result['n_exchanges']} < 3). "
            f"HHI={hhi_result['hhi']:.0f} — consider manual review before trusting this score."
        )

    fact = c1_from_hhi(name, hhi_result)
    if fact:
        facts.append(fact)
        if verbose:
            print(f"  [{name}] C1 = {fact.raw_value:.2f}  "
                  f"(HHI={hhi_result['hhi']:.0f}, n={hhi_result['n_exchanges']})")

    return facts, warnings


def refresh_c8(jur: dict, verbose: bool) -> tuple[list[RawFact], list[str]]:
    """Fetch WGI Regulatory Quality for C8."""
    facts, warnings = [], []
    iso3 = jur["iso3"]
    name = jur["name"]

    try:
        estimate, year = governance_index_client.get_wgi_regulatory_quality(iso3)
    except RuntimeError as e:
        warnings.append(f"[{name}] WGI C8 fetch failed: {e}")
        return facts, warnings

    if estimate is not None:
        facts.append(c8_from_wgi(name, estimate, year))
        score = scoring.score_wgi_regulatory_quality(estimate)
        if verbose:
            print(f"  [{name}] C8 = {score:.2f}  "
                  f"(WGI RQ.EST={estimate:.3f}, year={year})")
    else:
        warnings.append(f"[{name}] C8: no WGI data found")

    return facts, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    facts_path:  str,
    dry_run:     bool = False,
    skip_hhi:    bool = False,
    verbose:     bool = True,
) -> list[RawFact]:
    """
    Run all Tier-4 feed refreshes and merge results into facts_path.

    Returns the full list of new RawFacts produced (even in dry_run mode).
    """
    import os
    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY")

    new_facts: list[RawFact] = []
    all_warnings: list[str] = []

    for jur in JURISDICTIONS:
        if verbose:
            print(f"\n→ {jur['name']} ({jur['iso3']})")

        c2c5_facts, c2c5_warn = refresh_c2_c5(jur, verbose)
        new_facts.extend(c2c5_facts)
        all_warnings.extend(c2c5_warn)

        if not skip_hhi:
            c1_facts, c1_warn = refresh_c1(jur, verbose, api_key)
            new_facts.extend(c1_facts)
            all_warnings.extend(c1_warn)

        c8_facts, c8_warn = refresh_c8(jur, verbose)
        new_facts.extend(c8_facts)
        all_warnings.extend(c8_warn)

    if verbose:
        print(f"\n✓ {len(new_facts)} facts produced from {len(JURISDICTIONS)} jurisdictions")

    if all_warnings:
        print(f"\n⚠  {len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"   {w}")

    if dry_run:
        print("\n[dry-run] Not writing to disk. Facts that would be written:")
        for f in new_facts:
            print(f"  {f.jurisdiction} {f.criterion_code} = {f.raw_value:.2f} "
                  f"[{f.evidence_tag.value}] (as_of={f.as_of})")
        return new_facts

    # Merge into existing facts file
    path = Path(facts_path)
    existing = _load_existing_facts(path)
    merged   = _merge_facts(existing, new_facts)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))

    if verbose:
        print(f"\n✓ Written {len(merged)} total facts to {facts_path}")
        print(f"  ({len(existing)} existing preserved, {len(new_facts)} Tier-4 updated)")
        print("\n  Next step: run tools/run_pipeline.py --facts", facts_path,
              "--db ranking_engine/data/results.db")
        print("  The staleness checker will detect the new as_of dates and re-run automatically.")

    return new_facts


def main():
    parser = argparse.ArgumentParser(
        description="Refresh Tier-4 data feeds (C1, C2, C5, C8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--facts", required=True,
        help="Path to facts JSON file to update (created if it doesn't exist)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and score but do not write to disk"
    )
    parser.add_argument(
        "--skip-hhi", action="store_true",
        help="Skip CryptoCompare HHI fetch (use if API is down or key is missing)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-jurisdiction output; only show summary and warnings"
    )
    args = parser.parse_args()

    run(
        facts_path=args.facts,
        dry_run=args.dry_run,
        skip_hhi=args.skip_hhi,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
