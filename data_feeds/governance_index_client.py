"""
data_feeds/governance_index_client.py — WGI Regulatory Quality + B-READY ingest.

Two sources for C8 (Regulatory Volatility — inverted: high score = low volatility):

1. WGI Regulatory Quality (RQ.EST) — live World Bank API, same client as C2.
   Range: approximately -2.5 (weakest) to +2.5 (strongest).
   Update cycle: annual.

2. B-READY Index — NOT a polling API as of this writing.
   The World Bank publishes B-READY as downloadable Excel/CSV per report cycle,
   not via the standard v2/indicator endpoint. This module provides
   load_bready_release() for a human to run once per release, which writes
   the values into the same cache format so everything downstream is identical.

Attribution for C8 evidence tagging:
  WGI (API pull)      → VERIFIED_FACT   (direct pull from institutional source)
  B-READY (bulk load) → VERIFIED_FACT   (official World Bank release, just not via API)
"""

from pathlib import Path
import csv
import json

import requests as _requests
from .cache import cached as _cached, _ensure_cache_dir

# WGI Regulatory Quality: correct series ID on World Bank sources/3 endpoint
# (the old RQ.EST code was retired from the standard v2 indicator API)
_WGI_SERIES   = "GOV_WGI_RQ.EST"
_WGI_BASE     = "https://api.worldbank.org/v2/sources/3/country/{iso3}/series/{series}/time/all"
_WGI_TIMEOUT  = 20

# Cache key for manually-loaded B-READY data (namespaced to avoid collisions)
_BREADY_CACHE_KEY = "bready_manual_release.json"


@_cached(ttl_seconds=90 * 86_400)
def get_wgi_regulatory_quality(iso3: str) -> tuple[float | None, str | None]:
    """
    Fetch WGI Regulatory Quality estimate for a jurisdiction.

    Uses World Bank datasource 3 (WGI dataset) via the structured sources endpoint.
    Series: GOV_WGI_RQ.EST — "Regulatory Quality - Governance estimate (approx. -2.5 to +2.5)"

    Returns (estimate, year_str), e.g. (1.35, "2024").
    estimate is in [-2.5, +2.5] — convert to 1-10 with score_wgi_regulatory_quality().
    """
    url = _WGI_BASE.format(iso3=iso3.upper(), series=_WGI_SERIES)
    try:
        resp = _requests.get(url, params={"format": "json"}, timeout=_WGI_TIMEOUT)
        resp.raise_for_status()
    except _requests.RequestException as exc:
        raise RuntimeError(f"WGI API error for {iso3}: {exc}") from exc

    try:
        payload = resp.json()
    except Exception:
        raise RuntimeError(f"WGI response was not valid JSON for {iso3}: {resp.text[:200]}")

    # Response structure: {"source": {"data": [{"variable": [...], "value": float}, ...]}}
    rows = (
        payload.get("source", {}).get("data", [])
        if isinstance(payload.get("source"), dict)
        else []
    )

    # Rows are newest-first; find the first with a non-null value
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        # Extract year from the Time variable (e.g. "YR2024" -> "2024")
        year = None
        for var in row.get("variable", []):
            if var.get("concept") == "Time":
                raw_yr = var.get("id", "")        # "YR2024"
                year   = raw_yr.lstrip("YR") if raw_yr.startswith("YR") else raw_yr
                break
        return float(value), year

    return None, None


def load_bready_release(csv_path: str, value_column: str = "score") -> dict[str, float]:
    """
    Manually ingest a B-READY release CSV into the disk cache.

    The B-READY project publishes per-country sub-index scores in downloadable
    tables. This function reads the CSV, extracts the regulatory-framework-quality
    column, and writes it to the cache so downstream scoring sees a uniform interface.

    Parameters
    ----------
    csv_path     : path to the B-READY release CSV (e.g. "data/bready_2024.csv")
    value_column : name of the column containing the score to ingest (default: "score")
                   The CSV must also have a column named "iso3" or "country_code".

    Returns
    -------
    dict[iso3 -> score]

    Usage (run once per B-READY release, not on a cron):
        from data_feeds.governance_index_client import load_bready_release
        load_bready_release("data/bready_2025.csv", value_column="reg_quality_score")
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"B-READY CSV not found: {csv_path}")

    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        iso_col = None
        for candidate in ("iso3", "country_code", "ISO3", "CountryCode"):
            if candidate in (reader.fieldnames or []):
                iso_col = candidate
                break
        if iso_col is None:
            raise ValueError(
                f"Cannot find ISO-3 column in {csv_path}. "
                f"Expected one of: iso3, country_code, ISO3, CountryCode. "
                f"Got: {reader.fieldnames}"
            )
        if value_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{value_column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            iso3 = row[iso_col].strip().upper()
            raw  = row[value_column].strip()
            if raw:
                try:
                    data[iso3] = float(raw)
                except ValueError:
                    pass  # skip non-numeric rows (headers, notes)

    cache_path = _ensure_cache_dir() / _BREADY_CACHE_KEY
    cache_path.write_text(json.dumps(data))
    return data


def get_bready_score(iso3: str) -> float | None:
    """
    Read the B-READY score for a jurisdiction from the manually-loaded cache.

    Returns None if load_bready_release() has not been run yet, or if the
    jurisdiction is not in the release. Callers should treat None as 'no data'
    and skip producing a fact for this criterion rather than crashing.
    """
    cache_path = _ensure_cache_dir() / _BREADY_CACHE_KEY
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text())
    return data.get(iso3.upper())
