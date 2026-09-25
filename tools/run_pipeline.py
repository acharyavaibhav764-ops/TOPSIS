#!/usr/bin/env python
"""
tools/run_pipeline.py — Production CLI entry point for the ranking pipeline.

Usage examples:

  # Basic run (no persistence):
  python tools/run_pipeline.py --facts data/facts.json

  # Run with staleness check (skips if data is current):
  python tools/run_pipeline.py --facts data/facts.json --db ranking_engine/data/results.db

  # Force re-run regardless of staleness:
  python tools/run_pipeline.py --facts data/facts.json --db ranking_engine/data/results.db --force

  # Full production run with custom config files and stage dump:
  python tools/run_pipeline.py \\
      --facts data/facts.json \\
      --vpl-dpl ranking_engine/config/vpl_dpl.json \\
      --surveys ranking_engine/config/surveys.json \\
      --margins data/margins.json \\
      --dump-dir runs/2026-09-25 \\
      --db ranking_engine/data/results.db

Exit codes:
  0 = success or skipped (data current)
  1 = pipeline error (all markets knocked out, invalid config, etc.)

data/facts.json format:
  A JSON array of RawFact-serializable dicts:
  [{"jurisdiction": "UAE", "criterion_code": "C1", "raw_value": 0.2,
    "evidence_tag": "verified_fact", "source_citation": "VARA 2024",
    "as_of": "2026-01-01"}, ...]

data/margins.json format:
  {"UAE": 4000000, "Brazil": 2000000}
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Allow running from repo root or tools/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking_engine.pipeline import RankingPipeline, PipelineConfig
from ranking_engine.schema import RawFact, EvidenceTag
from ranking_engine.results_store import ResultsStore
from ranking_engine.staleness import needs_rerun, config_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_facts(path: str) -> list[RawFact]:
    raw = json.loads(Path(path).read_text())
    facts = []
    for r in raw:
        facts.append(RawFact(
            jurisdiction=r["jurisdiction"],
            criterion_code=r["criterion_code"],
            raw_value=float(r["raw_value"]),
            evidence_tag=EvidenceTag(r["evidence_tag"]),
            source_citation=r.get("source_citation", ""),
            as_of=date.fromisoformat(r["as_of"]),
            alt_raw_value=(float(r["alt_raw_value"]) if r.get("alt_raw_value") is not None else None),
        ))
    return facts


def print_table(results) -> None:
    print("\n" + "="*64)
    print(f"  {'Rank':<5} {'Jurisdiction':<22} {'Closeness':>10} {'Wave':>6}")
    print("  " + "-"*60)
    for r in results:
        wave = str(r.expansion_wave) if r.expansion_wave is not None else "—"
        interval = ""
        if r.mc_90pct_interval:
            interval = f"  [p5={r.mc_90pct_interval[0]:.3f}, p95={r.mc_90pct_interval[1]:.3f}]"
        print(f"  #{r.rank:<4} {r.jurisdiction:<22} {r.closeness:>10.4f} {wave:>6}{interval}")
    print("="*64 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MarketSphere ranking pipeline — production CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--facts",      required=True, help="Path to facts JSON")
    parser.add_argument("--vpl-dpl",    default=None,  help="Path to vpl_dpl.json (Phase 2B)")
    parser.add_argument("--surveys",    default=None,  help="Path to surveys.json  (Phase 2A)")
    parser.add_argument("--margins",    default=None,  help="Path to margins.json  (jur → USD 12mo)")
    parser.add_argument("--dump-dir",   default=None,  help="Directory to write numbered stage artifacts")
    parser.add_argument("--db",         default=None,  help="SQLite results database path")
    parser.add_argument("--force",      action="store_true", help="Re-run even if data is current")
    parser.add_argument("--quiet",      action="store_true", help="Suppress verbose pipeline output")
    args = parser.parse_args()

    # --- Load inputs ---
    facts = load_facts(args.facts)
    cfg   = PipelineConfig.from_files(
        vpl_dpl=args.vpl_dpl,
        surveys=args.surveys,
    )

    # --- Staleness check (only if a DB is provided) ---
    store = ResultsStore(args.db) if args.db else None
    if store:
        should_run, reason = needs_rerun(
            store=store,
            current_facts=facts,
            current_cfg_hash=config_hash(cfg),
            force=args.force,
        )
        if not should_run:
            print(f"✓ Skipping run: {reason}")
            store.close()
            sys.exit(0)
        if not args.quiet:
            print(f"→ Running pipeline: {reason}")

    # --- Load optional margins ---
    margin_12mo = None
    if args.margins:
        margin_12mo = json.loads(Path(args.margins).read_text())

    # --- Run pipeline ---
    pipeline = RankingPipeline(cfg)
    try:
        results, artifacts = pipeline.run(
            facts=facts,
            margin_12mo=margin_12mo,
            verbose=not args.quiet,
            dump_dir=args.dump_dir,
            store=store,
        )
    except (ValueError, ImportError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if store:
            store.close()
        sys.exit(1)

    print_table(results)

    if store:
        run_id = artifacts.get("run_id")
        if run_id:
            print(f"Persisted to DB: run_id={run_id}")
        store.close()


if __name__ == "__main__":
    main()
