#!/usr/bin/env python
"""
tools/run_full_pipeline.py — One-shot: refresh live data feeds → run ranking pipeline.

This is the "real data" entry point. It:
  1. Fetches live C1/C2/C5/C8 facts from World Bank + CryptoCompare
  2. Merges them with any existing C3/C4/C6/C7 manual facts
  3. Checks staleness (skip if data unchanged and pipeline already ran)
  4. Runs the full ranking pipeline with real data
  5. Prints results + persists to SQLite

Usage:
  python tools/run_full_pipeline.py --db data/results.db
  python tools/run_full_pipeline.py --db data/results.db --force
  python tools/run_full_pipeline.py --db data/results.db --skip-hhi    # if CC API is down
  python tools/run_full_pipeline.py --db data/results.db --dry-run     # fetch only, don't rank

Environment:
  CRYPTOCOMPARE_API_KEY   — required for C1 (HHI)
  MARKETSPHERE_CACHE_DIR  — override cache location (default: .cache/data_feeds/)
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── imports (data_feeds don't need numpy/ranking_engine) ──────────────────
from data_feeds import worldbank_client, exchange_hhi_client, governance_index_client, scoring
from data_feeds.raw_fact_adapter import (
    c1_from_hhi, c2_from_worldbank, c8_from_wgi,
)
from data_feeds.seed_c5_estimates import C5_SEED

# ranking_engine imports (require numpy — use sys Python with full deps)
from ranking_engine.pipeline   import RankingPipeline, PipelineConfig
from ranking_engine.schema     import RawFact, EvidenceTag
from ranking_engine.results_store import ResultsStore
from ranking_engine.staleness  import needs_rerun, config_hash


# ---------------------------------------------------------------------------
# Jurisdiction table (matches live_endpoint_check.py)
# ---------------------------------------------------------------------------
JURISDICTIONS = [
    {"name": "United States",        "iso3": "USA"},
    {"name": "United Kingdom",       "iso3": "GBR"},
    {"name": "Singapore",            "iso3": "SGP"},
    {"name": "Australia",            "iso3": "AUS"},
    {"name": "Switzerland",          "iso3": "CHE"},
    {"name": "Hong Kong",            "iso3": "HKG"},
    {"name": "Japan",                "iso3": "JPN"},
    {"name": "India",                "iso3": "IND"},
    {"name": "United Arab Emirates", "iso3": "ARE"},
    {"name": "Canada",               "iso3": "CAN"},
    {"name": "France",               "iso3": "FRA"},
    {"name": "Germany",              "iso3": "DEU"},
    {"name": "Netherlands",          "iso3": "NLD"},
    {"name": "South Korea",          "iso3": "KOR"},
    {"name": "Brazil",               "iso3": "BRA"},
]

FACTS_PATH = Path("data/facts_live.json")
MANUAL_CRITERIA = {"C3", "C4", "C6", "C7"}   # Tier-3: not yet automated


# ---------------------------------------------------------------------------
# Step 1: Fetch all Tier-4 facts
# ---------------------------------------------------------------------------

def fetch_all_facts(api_key: str | None, skip_hhi: bool, verbose: bool) -> list[RawFact]:
    new_facts: list[RawFact] = []
    warnings:  list[str]    = []

    for jur in JURISDICTIONS:
        name = jur["name"]
        iso3 = jur["iso3"]

        if verbose:
            print(f"  [{iso3}] {name}")

        # C2 + C5: World Bank
        try:
            inputs = worldbank_client.get_revenue_depth_inputs(iso3)
            c2s = scoring.score_revenue_depth(inputs["gni_ppp"], inputs["account_ownership_pct"])
            if c2s is not None:
                new_facts.append(c2_from_worldbank(
                    name, c2s, inputs["gni_year"], inputs["findex_year"]
                ))
                if verbose:
                    print(f"       C2={c2s:.2f} (GNI={inputs['gni_ppp']:,}, Findex={inputs['account_ownership_pct']}%)")
            # C5 is now appended via C5_SEED (see below)
        except Exception as e:
            warnings.append(f"[{iso3}] C2/C5 fetch failed: {e}")

        # C8: WGI
        try:
            est, yr = governance_index_client.get_wgi_regulatory_quality(iso3)
            if est is not None:
                c8s = scoring.score_wgi_regulatory_quality(est)
                new_facts.append(c8_from_wgi(name, est, yr))
                if verbose:
                    print(f"       C8={c8s:.2f} (WGI RQ={est:.3f}, year={yr})")
        except Exception as e:
            warnings.append(f"[{iso3}] C8 fetch failed: {e}")

        # C1: CryptoCompare HHI
        if not skip_hhi:
            try:
                hhi_res = exchange_hhi_client.compute_hhi(iso3, api_key=api_key)
                if hhi_res["hhi"] is not None:
                    if hhi_res["low_coverage"]:
                        warnings.append(
                            f"[{iso3}] C1: low exchange coverage "
                            f"(n={hhi_res['n_exchanges']}). Fact included but marked ANALYST_ESTIMATE."
                        )
                    fact = c1_from_hhi(name, hhi_res)
                    if fact:
                        new_facts.append(fact)
                        if verbose:
                            c1s = scoring.score_hhi(hhi_res["hhi"])
                            print(f"       C1={c1s:.2f} (HHI={hhi_res['hhi']:.0f}, n={hhi_res['n_exchanges']})")
                else:
                    warnings.append(f"[{iso3}] C1: no exchanges found in CC data")
            except Exception as e:
                warnings.append(f"[{iso3}] C1 fetch failed: {e}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    {w}")

    # Append C5 interim seed data
    new_facts.extend(C5_SEED)

    return new_facts


# ---------------------------------------------------------------------------
# Step 2: Merge with existing manual facts (C3/C4/C6/C7)
# ---------------------------------------------------------------------------

def merge_with_manual(new_facts: list[RawFact]) -> list[RawFact]:
    """Keep manual facts (C3/C4/C6/C7) from existing file; replace Tier-4 criteria."""
    if not FACTS_PATH.exists():
        return new_facts

    existing_raw = json.loads(FACTS_PATH.read_text())
    manual_dicts = [f for f in existing_raw if f.get("criterion_code") in MANUAL_CRITERIA]

    manual_facts = []
    for r in manual_dicts:
        try:
            manual_facts.append(RawFact(
                jurisdiction    = r["jurisdiction"],
                criterion_code  = r["criterion_code"],
                raw_value       = float(r["raw_value"]),
                evidence_tag    = EvidenceTag(r["evidence_tag"]),
                source_citation = r.get("source_citation", ""),
                as_of           = date.fromisoformat(r["as_of"]),
                alt_raw_value   = float(r["alt_raw_value"]) if r.get("alt_raw_value") else None,
            ))
        except Exception:
            pass

    return manual_facts + new_facts


# ---------------------------------------------------------------------------
# Step 3: Serialize facts to JSON file
# ---------------------------------------------------------------------------

def serialize_facts(facts: list[RawFact]) -> list[dict]:
    out = []
    for f in facts:
        out.append({
            "jurisdiction":    f.jurisdiction,
            "criterion_code":  f.criterion_code,
            "raw_value":       f.raw_value,
            "evidence_tag":    f.evidence_tag.value,
            "source_citation": f.source_citation,
            "as_of":           f.as_of.isoformat(),
            "alt_raw_value":   f.alt_raw_value,
        })
    return out


# ---------------------------------------------------------------------------
# Step 4: Run the ranking pipeline
# ---------------------------------------------------------------------------

def print_results(results) -> None:
    print("\n" + "=" * 68)
    print(f"  {'Rank':<5} {'Jurisdiction':<26} {'Closeness':>10} {'Wave':>6}")
    print("  " + "-" * 64)
    for r in results:
        wave = str(r.expansion_wave) if r.expansion_wave is not None else "—"
        interval = ""
        if r.mc_90pct_interval:
            lo, hi = r.mc_90pct_interval
            interval = f"  [p5={lo:.3f}, p95={hi:.3f}]"
        print(f"  #{r.rank:<4} {r.jurisdiction:<26} {r.closeness:>10.4f} {wave:>6}{interval}")
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MarketSphere: refresh live data feeds + run ranking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db",        default="data/results.db",
                        help="SQLite results database (default: data/results.db)")
    parser.add_argument("--vpl-dpl",   default=None,
                        help="Path to vpl_dpl.json (optional — Phase 2B inputs)")
    parser.add_argument("--surveys",   default=None,
                        help="Path to surveys.json (optional — stakeholder weights)")
    parser.add_argument("--margins",   default=None,
                        help="Path to margins.json (optional — jurisdiction USD margins)")
    parser.add_argument("--skip-hhi",  action="store_true",
                        help="Skip CryptoCompare C1 fetch (use if API is down)")
    parser.add_argument("--force",     action="store_true",
                        help="Re-run pipeline even if data hasn't changed")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Fetch and write facts.json but do not run the ranking")
    parser.add_argument("--quiet",     action="store_true",
                        help="Suppress per-jurisdiction fetch output")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Bust all data-feed cache entries before fetching")
    args = parser.parse_args()

    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY")
    if not api_key and not args.skip_hhi:
        print("WARNING: CRYPTOCOMPARE_API_KEY not set. C1 (HHI) will likely fail.\n"
              "         Set the env var or use --skip-hhi to suppress this.\n")

    if args.no_cache:
        import shutil
        cache_dir = Path(os.environ.get("MARKETSPHERE_CACHE_DIR", ".cache/data_feeds"))
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cache cleared: {cache_dir}\n")

    # ── Step 1: Fetch live data ───────────────────────────────────────────
    print("\n-- Step 1: Fetching Tier-4 data feeds --")
    new_facts = fetch_all_facts(api_key, skip_hhi=args.skip_hhi, verbose=not args.quiet)
    print(f"\n  Fetched {len(new_facts)} live facts across {len(JURISDICTIONS)} jurisdictions")

    # ── Step 2: Merge with manual C3/C4/C6/C7 facts ──────────────────────
    all_facts = merge_with_manual(new_facts)
    print(f"  Total facts (incl. manual): {len(all_facts)}")

    # ── Step 3: Write facts.json ──────────────────────────────────────────
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FACTS_PATH.write_text(json.dumps(serialize_facts(all_facts), indent=2))
    print(f"  Written to: {FACTS_PATH}")

    if args.dry_run:
        print("\n[dry-run] Stopping before pipeline run.")
        return

    # ── Step 4: Run pipeline ──────────────────────────────────────────────
    print("\n-- Step 2: Running ranking pipeline --")

    cfg   = PipelineConfig.from_files(vpl_dpl=args.vpl_dpl, surveys=args.surveys)
    store = ResultsStore(args.db) if args.db else None
    Path(args.db).parent.mkdir(parents=True, exist_ok=True) if args.db else None

    if store and not args.force:
        should_run, reason = needs_rerun(
            store=store,
            current_facts=all_facts,
            current_cfg_hash=config_hash(cfg),
            force=False,
        )
        if not should_run:
            print(f"  Skipping: {reason}")
            store.close()
            return
        print(f"  Running: {reason}")

    margin_12mo = None
    if args.margins:
        margin_12mo = json.loads(Path(args.margins).read_text())

    pipeline = RankingPipeline(cfg)
    try:
        results, artifacts = pipeline.run(
            facts        = all_facts,
            margin_12mo  = margin_12mo,
            verbose      = not args.quiet,
            store        = store,
        )
    except (ValueError, RuntimeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if store:
            store.close()
        sys.exit(1)

    print_results(results)

    if store:
        run_id = artifacts.get("run_id")
        if run_id:
            print(f"Persisted: run_id={run_id}, db={args.db}")
        store.close()


if __name__ == "__main__":
    main()
