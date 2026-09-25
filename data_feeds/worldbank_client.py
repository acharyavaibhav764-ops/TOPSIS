"""
data_feeds/worldbank_client.py — World Bank Open Data API client.

Feeds
-----
  C2 (Revenue Depth):
    - NY.GNP.PCAP.PP.CD  — GNI per capita, PPP (current international $)
    - FX.OWN.TOTL.ZS     — Account ownership at financial institution or mobile
                            money (% adults, Global Findex proxy)

  C8 partial / C5 base signal:
    See governance_index_client.py for WGI (reuses get_indicator from here).

API
---
  Base URL : https://api.worldbank.org/v2/country/{iso3}/indicator/{code}
  Auth     : none — completely open
  Format   : JSON (two-element list: [metadata, [data rows]])
  TTL      : 90 days (annual update cycle)

The API returns rows newest-first, possibly with null values for recent years
(data lag). get_indicator() walks the rows and returns the most recent
non-null value together with its year string, so a 2024 API call might
return 2022 data — the year is part of the returned tuple for the caller
to surface as the as_of date.
"""

import requests
from .cache import cached

_WB_BASE = "https://api.worldbank.org/v2/country/{iso3}/indicator/{code}"
_TIMEOUT  = 15       # seconds
_PER_PAGE = 10       # fetch last 10 years; walk until non-null found


@cached(ttl_seconds=90 * 86_400)
def get_indicator(iso3: str, code: str, source: int | None = None) -> tuple[float | None, str | None]:
    """
    Fetch the most recent non-null value for a World Bank indicator.

    Parameters
    ----------
    iso3   : ISO-3166-1 alpha-3 country code (e.g. "ARE", "BRA")
    code   : World Bank indicator code (e.g. "NY.GNP.PCAP.PP.CD")
    source : World Bank datasource ID. WGI indicators require source=3.
             Standard indicators (GNI, Findex) use the default (None).

    Returns
    -------
    (value, year_str) — e.g. (54_210.0, "2023")
    (None, None)      — if no non-null value found in the last _PER_PAGE years
    """
    url    = _WB_BASE.format(iso3=iso3, code=code)
    params = {"format": "json", "per_page": _PER_PAGE, "mrv": _PER_PAGE}
    if source is not None:
        params["source"] = source

    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"World Bank API error for {iso3}/{code}: {exc}"
        ) from exc

    payload = resp.json()
    # The API returns [metadata_dict, [data_rows]] or [metadata_dict, None]
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return None, None

    for row in payload[1]:
        if row.get("value") is not None:
            return float(row["value"]), str(row["date"])

    return None, None


def get_revenue_depth_inputs(iso3: str) -> dict:
    """
    Fetch both inputs needed for C2 (Revenue Depth) scoring.

    Returns
    -------
    dict with keys:
        gni_ppp              : float | None   — GNI per capita, PPP (USD)
        gni_year             : str   | None
        account_ownership_pct: float | None   — % adults with financial account
        findex_year          : str   | None
    """
    gni,    gni_year    = get_indicator(iso3, "NY.GNP.PCAP.PP.CD")
    findex, findex_year = get_indicator(iso3, "FX.OWN.TOTL.ZS")
    return {
        "gni_ppp":               gni,
        "gni_year":              gni_year,
        "account_ownership_pct": findex,
        "findex_year":           findex_year,
    }


def get_digital_payments_proxy(iso3: str) -> tuple[float | None, str | None]:
    """
    Fetch the digital-payments infrastructure proxy for the C5 base signal.

    Uses FX.OWN.TOTL.ZS (account ownership / mobile-money penetration) as
    a structural floor indicator — tells you whether a real-time payment
    infrastructure exists at population level. Does NOT tell you whether
    regulators allow VASPs to plug into it (that's Tier 3).

    Returns (value_pct, year_str).
    """
    return get_indicator(iso3, "FX.OWN.TOTL.ZS")
