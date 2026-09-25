#!/usr/bin/env python
"""
tools/live_endpoint_check.py

Hits real World Bank and CryptoCompare endpoints for 15 target jurisdictions
and prints a formatted table of every raw value fetched + derived score.
No mocking. Uses the actual data_feeds/ clients with a fresh cache.

Run:
    python tools/live_endpoint_check.py
    python tools/live_endpoint_check.py --no-cache   # bust existing cache first
"""

import sys
import os
import io
import argparse
from pathlib import Path
from datetime import date

# Force UTF-8 stdout on Windows (default is cp1252 in some terminals)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force the cache to a sub-dir of this run so it's clean and inspectable
os.environ.setdefault("MARKETSPHERE_CACHE_DIR", ".cache/live_check")

from data_feeds import worldbank_client, exchange_hhi_client, governance_index_client, scoring


# ---------------------------------------------------------------------------
# Target jurisdictions
# ---------------------------------------------------------------------------
TARGETS = [
    # (display name,          iso3,  cc2  )
    ("United States",         "USA", "US"),
    ("United Kingdom",        "GBR", "GB"),
    ("Singapore",             "SGP", "SG"),
    ("Australia",             "AUS", "AU"),
    ("Switzerland",           "CHE", "CH"),
    ("Hong Kong",             "HKG", "HK"),
    ("Japan",                 "JPN", "JP"),
    ("India",                 "IND", "IN"),
    ("United Arab Emirates",  "ARE", "AE"),
    ("Canada",                "CAN", "CA"),
    ("France",                "FRA", "FR"),
    ("Germany",               "DEU", "DE"),
    ("Netherlands",           "NLD", "NL"),
    ("South Korea",           "KOR", "KR"),
    ("Brazil",                "BRA", "BR"),
]


def bust_all():
    """Delete all cached files so we get fresh data."""
    import shutil
    cache_dir = Path(os.environ["MARKETSPHERE_CACHE_DIR"])
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"  Cache cleared: {cache_dir}\n")


def fmt(v, decimals=2):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:,.{decimals}f}"
    return str(v)


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Bust cache before fetching")
    args = parser.parse_args()

    if args.no_cache:
        bust_all()

    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY")

    # -------------------------------------------------------------------------
    # SECTION 1 — C2: Revenue Depth (GNI PPP + Findex blend)
    # -------------------------------------------------------------------------
    section("C2 — Revenue Depth  [World Bank: NY.GNP.PCAP.PP.CD + FX.OWN.TOTL.ZS]")
    print(f"  {'Jurisdiction':<26} {'GNI/cap PPP':>13} {'Year':>6}  "
          f"{'Findex%':>8} {'FY':>6}  {'C2 Score':>9}")
    print(f"  {'-'*76}")

    c2_results = {}
    for name, iso3, cc2 in TARGETS:
        try:
            inputs = worldbank_client.get_revenue_depth_inputs(iso3)
            score  = scoring.score_revenue_depth(
                inputs["gni_ppp"], inputs["account_ownership_pct"]
            )
            c2_results[name] = (inputs, score)
            print(f"  {name:<26} {fmt(inputs['gni_ppp'], 0):>13} {inputs['gni_year'] or 'N/A':>6}  "
                  f"{fmt(inputs['account_ownership_pct']):>8} {inputs['findex_year'] or 'N/A':>6}  "
                  f"{fmt(score):>9}")
        except Exception as e:
            print(f"  {name:<26}  ERROR: {e}")

    # -------------------------------------------------------------------------
    # SECTION 2 — C8: Regulatory Volatility (WGI Regulatory Quality)
    # -------------------------------------------------------------------------
    section("C8 — Regulatory Volatility  [World Bank WGI: RQ.EST]  (-2.5 weak → +2.5 strong)")
    print(f"  {'Jurisdiction':<26} {'WGI RQ':>10} {'Year':>6}  {'C8 Score':>9}")
    print(f"  {'-'*56}")

    c8_results = {}
    for name, iso3, cc2 in TARGETS:
        try:
            estimate, year = governance_index_client.get_wgi_regulatory_quality(iso3)
            score = scoring.score_wgi_regulatory_quality(estimate)
            c8_results[name] = (estimate, year, score)
            print(f"  {name:<26} {fmt(estimate, 3):>10} {year or 'N/A':>6}  {fmt(score):>9}")
        except Exception as e:
            print(f"  {name:<26}  ERROR: {e}")

    # -------------------------------------------------------------------------
    # SECTION 3 — C5: Local Rail Fit base signal (Findex proxy)
    # -------------------------------------------------------------------------
    section("C5 — Local Rail Fit BASE SIGNAL  [World Bank FX.OWN.TOTL.ZS proxy]")
    print(f"  ⚠  Tagged ANALYST_ESTIMATE — VASP-openness component NOT included")
    print(f"  {'Jurisdiction':<26} {'Account own%':>13} {'Year':>6}  {'C5 Score':>9}")
    print(f"  {'-'*60}")

    for name, iso3, cc2 in TARGETS:
        try:
            pct, year = worldbank_client.get_digital_payments_proxy(iso3)
            score = scoring.score_c5_base_signal(pct)
            print(f"  {name:<26} {fmt(pct, 1):>13} {year or 'N/A':>6}  {fmt(score):>9}")
        except Exception as e:
            print(f"  {name:<26}  ERROR: {e}")

    # -------------------------------------------------------------------------
    # SECTION 4 — C1: Demand Scale (HHI from CryptoCompare)
    # -------------------------------------------------------------------------
    section("C1 — Demand Scale  [CryptoCompare: HHI of exchange vol by domicile]")
    print(f"  ⚠  Tagged ANALYST_ESTIMATE — jurisdiction attribution is inferred")
    if api_key:
        print(f"  ℹ  Using API key from CRYPTOCOMPARE_API_KEY")
    else:
        print(f"  ℹ  No CRYPTOCOMPARE_API_KEY set — using free tier (rate-limited)")
    print(f"  {'Jurisdiction':<26} {'n_exch':>7} {'HHI':>10} {'Coverage':>10}  {'C1 Score':>9}")
    print(f"  {'-'*70}")

    for name, iso3, cc2 in TARGETS:
        try:
            result = exchange_hhi_client.compute_hhi(iso3, api_key=api_key)
            score  = scoring.score_hhi(result["hhi"])
            low    = "LOW ⚠" if result["low_coverage"] else "OK"
            print(f"  {name:<26} {result['n_exchanges']:>7} {fmt(result['hhi'], 0):>10} "
                  f"{low:>10}  {fmt(score):>9}")
            if result["exchanges"]:
                exch_str = ", ".join(result["exchanges"][:6])
                if len(result["exchanges"]) > 6:
                    exch_str += f" (+{len(result['exchanges'])-6} more)"
                print(f"  {'':>26}   exchanges: {exch_str}")
        except Exception as e:
            print(f"  {name:<26}  ERROR: {e}")

    # -------------------------------------------------------------------------
    # SECTION 5 — Combined summary table
    # -------------------------------------------------------------------------
    section("COMBINED SCORE SUMMARY (higher = better for all 4 criteria)")
    print(f"  {'Jurisdiction':<26} {'C1':>9} {'C2':>9} {'C5 base':>9} {'C8':>9}")
    print(f"  {'-'*65}")

    for name, iso3, cc2 in TARGETS:
        try:
            inputs = worldbank_client.get_revenue_depth_inputs(iso3)
            c2s = scoring.score_revenue_depth(inputs["gni_ppp"], inputs["account_ownership_pct"])
        except:
            c2s = None
        try:
            est, yr = governance_index_client.get_wgi_regulatory_quality(iso3)
            c8s = scoring.score_wgi_regulatory_quality(est)
        except:
            c8s = None
        try:
            pct, _ = worldbank_client.get_digital_payments_proxy(iso3)
            c5s = scoring.score_c5_base_signal(pct)
        except:
            c5s = None
        try:
            r   = exchange_hhi_client.compute_hhi(iso3, api_key=api_key)
            c1s = scoring.score_hhi(r["hhi"])
        except:
            c1s = None

        print(f"  {name:<26} {fmt(c1s):>9} {fmt(c2s):>9} {fmt(c5s):>9} {fmt(c8s):>9}")

    print(f"\n  ℹ  C5 scores above are base signal only (World Bank proxy).")
    print(f"     VASP-openness requires manual override (Tier 3 not yet built).")
    print(f"     C1 low-coverage jurisdictions (⚠) have n < 3 exchanges in CC data —")
    print(f"     scores may not reflect actual derivative market depth.\n")


if __name__ == "__main__":
    main()
