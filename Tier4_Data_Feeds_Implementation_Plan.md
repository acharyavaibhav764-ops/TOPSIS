# Tier 4 — Automated Data Feeds: Implementation Plan

Scope, per the roadmap in `ranking_engine_final_steps.md`: replace hand-entered values for **C1** (Demand Scale), **C2** (Revenue Depth), **C5** (Local Rail Fit), and **C8** (Regulatory Volatility) with automated feeds. This is a self-contained package — it does **not** touch `topsis_rad.py`, `pipeline.py`'s scoring logic, or anything from Tier 1/2. It only produces `RawFact` rows and hands them to whatever already ingests facts today.

**One assumption flagged up front:** I don't have the actual `RawFact` dataclass from `ranking_engine.zip`, so the adapter functions below emit a generic dict with the fields implied by the existing docs (`jurisdiction`, `criterion`, `raw_value`, `alt_raw_value`, `evidence_tag`, `source`, `as_of`). If the real dataclass has different field names, it's a 5-minute rename — flag it and I'll patch the adapters, no need to re-plan anything.

---

## 1. Architecture

```
data_feeds/
  __init__.py
  cache.py                 # TTL disk cache, shared by all clients
  worldbank_client.py       # C2, part of C5
  exchange_hhi_client.py    # C1
  governance_index_client.py # C8 (WGI + B-READY)
  scoring.py                 # raw value -> 1-10 mapping functions
  raw_fact_adapter.py        # -> RawFact rows
tools/
  refresh_data_feeds.py      # CLI entry point, cron/schedule target
```

Flow: each client fetches + caches raw numbers → `scoring.py` maps them to the 1–10 construct the matrix expects (or High/Med/Low for C5) → `raw_fact_adapter.py` wraps them as `RawFact` rows with an evidence tag → `refresh_data_feeds.py` writes them into the same ingestion path `tools/run_pipeline.py` already reads from, then touches `config_hash()` so `staleness.py` (built in Tier 2) picks up the change and flags a re-run.

Nothing here calls an LLM. That's the point of Tier 4 vs Tier 3 — these four criteria have clean, structured, institutional sources, so there's no extraction/hallucination problem to solve, only fetch → normalize → map.

---

## 2. Shared cache layer

World Bank indicators update annually; WGI updates annually; exchange volume is the only fast-moving one. No reason to hit any API more than once a day, and World Bank/WGI can be cached for weeks.

```python
# data_feeds/cache.py
import json, time, hashlib
from pathlib import Path

CACHE_DIR = Path(".cache/data_feeds")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cached(ttl_seconds: int):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = hashlib.sha256(f"{fn.__name__}{args}{kwargs}".encode()).hexdigest()
            path = CACHE_DIR / f"{key}.json"
            if path.exists() and time.time() - path.stat().st_mtime < ttl_seconds:
                return json.loads(path.read_text())
            result = fn(*args, **kwargs)
            path.write_text(json.dumps(result))
            return result
        return wrapper
    return decorator
```

Use `@cached(ttl_seconds=90*86400)` for World Bank/WGI calls, `@cached(ttl_seconds=86400)` for exchange volume.

---

## 3. C2 (Revenue Depth) + partial C5 — World Bank Open Data API

No API key needed. Two indicators:

| Indicator | Code | Feeds |
|---|---|---|
| GNI per capita, PPP | `NY.GNP.PCAP.PP.CD` | C2 |
| Account ownership / digital payments (Global Findex, folded into WB API) | `FX.OWN.TOTL.ZS` | C2 (financial inclusion half) |

```python
# data_feeds/worldbank_client.py
import requests
from .cache import cached

WB_BASE = "https://api.worldbank.org/v2/country/{iso3}/indicator/{code}"

@cached(ttl_seconds=90*86400)
def get_indicator(iso3: str, code: str, year: int | None = None) -> float | None:
    params = {"format": "json", "per_page": 5}
    if year:
        params["date"] = str(year)
    resp = requests.get(WB_BASE.format(iso3=iso3, code=code), params=params, timeout=15)
    resp.raise_for_status()
    _, rows = resp.json()
    for row in rows or []:
        if row["value"] is not None:
            return row["value"], row["date"]  # most recent non-null year
    return None, None

def get_revenue_depth_inputs(iso3: str) -> dict:
    gni, gni_year = get_indicator(iso3, "NY.GNP.PCAP.PP.CD")
    findex, findex_year = get_indicator(iso3, "FX.OWN.TOTL.ZS")
    return {"gni_ppp": gni, "gni_year": gni_year,
            "account_ownership_pct": findex, "findex_year": findex_year}
```

**C5 caveat:** the strategy doc defines C5 as "availability and openness of instant fiat payment rails **to VASPs**." World Bank data can tell you whether a country has real-time payment rails at all (a reasonable quantitative floor), but *not* whether regulators let virtual-asset platforms plug into them — that's a regulatory-openness fact, which is really Tier 3 territory (it needs the same verbatim-source-span verification as C3/C4/C7). Recommendation: use World Bank digital-payments-infrastructure data as the automated **base signal** for C5 (feeds a Low/Med/High tier), and keep a small manual override table for the VASP-specific openness flag until Tier 3 exists to extract it properly. Don't silently automate the part that isn't actually automatable — that's the same class of bug this whole project just spent two tiers fixing.

---

## 4. C1 (Demand Scale) — exchange volume + HHI

**Data source:** CryptoCompare's free-tier API (`/data/exchanges/general`) gives per-exchange volume and country-of-incorporation/regulatory-domicile metadata, no contract needed to start. Kaiko is the institutional-grade upgrade path if CryptoCompare's coverage turns out too thin for a target jurisdiction — same interface shape, swap the client.

```python
# data_feeds/exchange_hhi_client.py
import requests
from .cache import cached

@cached(ttl_seconds=86400)
def get_exchange_volumes() -> list[dict]:
    resp = requests.get("https://min-api.cryptocompare.com/data/exchanges/general", timeout=15)
    resp.raise_for_status()
    return resp.json()  # {exchange_name: {Country, TOTALVOLUME24H, ...}, ...}

def compute_hhi(jurisdiction_country_code: str) -> tuple[float, int]:
    """Returns (hhi, n_exchanges) for exchanges domiciled/licensed in this jurisdiction."""
    data = get_exchange_volumes()
    volumes = {
        name: info["TOTALVOLUME24H"]
        for name, info in data.items()
        if info.get("Country") == jurisdiction_country_code and info.get("TOTALVOLUME24H")
    }
    total = sum(volumes.values())
    if total == 0:
        return None, 0
    hhi = sum((v / total) ** 2 for v in volumes.values()) * 10000
    return hhi, len(volumes)
```

**Attribution caveat, stated plainly:** filtering by exchange country-of-incorporation is a *proxy* for "who actually serves this jurisdiction's derivative demand," not a precise measure — an exchange licensed in Country X can serve global order flow. This is a known, documented limitation, not a hidden one. Tag every C1 fact produced this way `ANALYST_ESTIMATE`, not `VERIFIED_FACT` (see §6) — it's a real computed number from real trade data, but the jurisdiction attribution is inferred, so it should carry the wider Monte Carlo variance band the evidence-tag scheme already applies to estimates.

---

## 5. C8 (Regulatory Volatility) — WGI + B-READY

```python
# data_feeds/governance_index_client.py
from .worldbank_client import get_indicator

def get_regulatory_quality(iso3: str) -> tuple[float, int]:
    # WGI Regulatory Quality, estimate (-2.5 weak .. +2.5 strong)
    return get_indicator(iso3, "RQ.EST")
```

**B-READY caveat:** unlike the classic WGI indicators, the World Bank's B-READY project is not (as of this writing) exposed on the standard `api.worldbank.org/v2/indicator` endpoint the same way — it's published as downloadable Excel/CSV releases per report cycle. Treat it as a **manual/scripted bulk-ingest job**, not a live API call: download the release once per cycle, parse the relevant sub-index (regulatory framework quality) into the same cache format `worldbank_client.py` produces, and let it flow through the same scoring path. Recommend building this as a separate `governance_index_client.load_bready_release(path)` function that a human runs once per B-READY release, rather than pretending it's a polling API — same principle as the C5 caveat: don't automate a step that isn't actually automatable yet, mark it explicitly instead.

---

## 6. Scoring + evidence tagging

```python
# data_feeds/scoring.py
def score_gni(gni_ppp: float) -> float:
    # example fixed thresholds — swap for whatever thresholds pipeline.py already uses elsewhere
    breakpoints = [(2000, 1), (8000, 4), (20000, 7), (40000, 10)]
    for cap, score in breakpoints:
        if gni_ppp <= cap:
            return score
    return 10

def score_hhi(hhi: float) -> float:
    # lower HHI = more fragmented = more contestable = higher C1 score (benefit)
    if hhi is None:
        return None
    if hhi < 1500: return 9
    if hhi < 2500: return 6
    if hhi < 5000: return 3
    return 1

def score_regulatory_quality(wgi_estimate: float) -> float:
    # map -2.5..2.5 to 1..10
    return round(((wgi_estimate + 2.5) / 5.0) * 9 + 1, 2)
```

```python
# data_feeds/raw_fact_adapter.py
from datetime import date, datetime, UTC

def to_raw_fact(jurisdiction: str, criterion: str, raw_value: float,
                 evidence_tag: str, source: str, as_of: str | None = None) -> dict:
    return {
        "jurisdiction": jurisdiction,
        "criterion": criterion,          # "C1".."C8" — confirm against actual RawFact field name
        "raw_value": raw_value,
        "evidence_tag": evidence_tag,    # "VERIFIED_FACT" | "ANALYST_ESTIMATE" | "UNRESOLVED_POLICY"
        "source": source,
        "as_of": as_of or datetime.now(UTC).isoformat(),
    }
```

Tagging rule for Tier 4 specifically (this is the one addition to the evidence-tag scheme Tier 1–3 didn't need to make, since it was designed for LLM extraction):

| Criterion | Source | Tag |
|---|---|---|
| C2 (GNI, Findex) | World Bank API, direct pull | `VERIFIED_FACT` |
| C8 (WGI) | World Bank API, direct pull | `VERIFIED_FACT` |
| C8 (B-READY) | Manual bulk ingest of official release | `VERIFIED_FACT` |
| C1 (HHI) | Computed, jurisdiction-attribution is inferred | `ANALYST_ESTIMATE` |
| C5 (base signal) | World Bank digital-payments proxy | `ANALYST_ESTIMATE` |
| C5 (VASP-openness flag) | Manual override, no automated source yet | `ANALYST_ESTIMATE` until Tier 3 exists |

This slots directly into the Evidence-Calibrated Monte Carlo already built (tight variance for `VERIFIED_FACT`, wider for `ANALYST_ESTIMATE`) — no changes needed there.

---

## 7. Refresh CLI + staleness hook

```python
# tools/refresh_data_feeds.py
"""Run on a schedule (cron/Airflow/etc). Fetches all Tier-4 feeds, writes RawFacts,
then reuses the Tier-2 staleness mechanism so a data change flags a pipeline re-run."""

from data_feeds import worldbank_client, exchange_hhi_client, governance_index_client, scoring
from data_feeds.raw_fact_adapter import to_raw_fact
# from staleness import mark_dirty   # reuse the Tier-2 module — confirm exact import path

JURISDICTIONS = [
    # (jurisdiction_name, iso3, exchange_country_code)
    ("UAE", "ARE", "AE"),
    ("Brazil", "BRA", "BR"),
    # ... rest of the target list
]

def run():
    facts = []
    for name, iso3, cc in JURISDICTIONS:
        inputs = worldbank_client.get_revenue_depth_inputs(iso3)
        if inputs["gni_ppp"] is not None:
            facts.append(to_raw_fact(name, "C2", scoring.score_gni(inputs["gni_ppp"]),
                                      "VERIFIED_FACT", "World Bank NY.GNP.PCAP.PP.CD",
                                      inputs["gni_year"]))

        hhi, n = exchange_hhi_client.compute_hhi(cc)
        if hhi is not None:
            facts.append(to_raw_fact(name, "C1", scoring.score_hhi(hhi),
                                      "ANALYST_ESTIMATE", f"CryptoCompare, n={n} exchanges"))

        wgi, wgi_year = governance_index_client.get_regulatory_quality(iso3)
        if wgi is not None:
            facts.append(to_raw_fact(name, "C8", scoring.score_regulatory_quality(wgi),
                                      "VERIFIED_FACT", "World Bank WGI RQ.EST", wgi_year))

    # write `facts` into whatever ingestion path tools/run_pipeline.py already reads
    # then: mark_dirty(reason="tier4_data_feed_refresh")  # triggers staleness re-run flag

if __name__ == "__main__":
    run()
```

Where `mark_dirty` doesn't exist by that name — check `staleness.py`'s actual public function; Tier 2's log mentions `config_hash()` explicitly serializing `vpl/dpl/surveys`, so the equivalent trigger for a fact-set change is likely already there or a one-line addition next to it.

---

## 8. Test plan (same verification discipline as Tier 1–3)

| Test | Catches |
|---|---|
| `test_worldbank_client_picks_most_recent_nonnull_year` | Silent use of a stale/null year when a country has gaps |
| `test_score_gni_breakpoints` | Off-by-one at threshold boundaries |
| `test_hhi_hand_calculated_example` | HHI formula regression (3 exchanges, hand-computed expected value) |
| `test_hhi_single_exchange_jurisdiction_returns_max_concentration` | Edge case: one exchange = HHI 10000 |
| `test_missing_jurisdiction_data_does_not_crash_refresh` | A country with no WB/CryptoCompare coverage should produce no fact for that criterion, not an exception that kills the whole refresh |
| `test_evidence_tags_match_source_table` | Someone changes a source later and forgets to also change the tag in §6's table |
| `test_cache_respects_ttl` | Stale cache silently served past its TTL |
| `test_refresh_triggers_staleness_flag` | A data change that doesn't propagate to `staleness.py`, silently leaving the pipeline serving an outdated rank |

---

## 9. Suggested build order

1. **World Bank client (C2, C8-WGI)** — cleanest API, no attribution ambiguity, do this first and get the evidence-tagging/cache pattern right here.
2. **Governance/B-READY manual ingest (C8)** — small, mechanical, reuses §1's cache format.
3. **Exchange HHI (C1)** — hardest of the four because of the attribution caveat in §4; budget real time for deciding how strict the country-filter should be, and consider flagging low-`n_exchanges` jurisdictions for manual review rather than trusting a 2-exchange HHI.
4. **C5** — smallest piece of actual code, but don't skip writing down the manual-override table; it's the one place in Tier 4 where "automated" is genuinely partial, and that should be visible in the data, not buried.

Once this lands, only Tier 3 (LLM regulatory extraction — C3, C4, C6, C7) and Tier 5 (non-engineering, GIFT City/IFSCA setup) remain on the roadmap.
