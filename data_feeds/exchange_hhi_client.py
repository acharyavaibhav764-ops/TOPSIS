"""
data_feeds/exchange_hhi_client.py — Exchange volume + HHI computation for C1.

Source: CryptoCompare API (/data/exchanges/general).
  Requires an API key (free tier available at min-api.cryptocompare.com).
  Kaiko is the institutional upgrade path — same data shape, swap the URL/key.

Important: CryptoCompare's 'Country' field is a free-text full country name
  (e.g. 'U.S.A', 'United Kingdom', 'Singapore'), NOT ISO-2 codes.
  We maintain a mapping from ISO-3 → accepted CC name variants below.
  This is an ANALYST_ESTIMATE by nature — jurisdiction attribution is inferred
  from exchange registration, not actual order-flow geography.

TOTALVOLUME24H is returned as a dict of {symbol: volume}, not a float.
  We sum all symbol volumes as a proxy for aggregate 24h activity.

HHI formula
-----------
  HHI = sum((volume_i / total_volume)^2) * 10_000
  Range: [0, 10_000]. Monopoly -> 10_000. Fragmented -> 0.
  C1 inverts HHI: low HHI -> high C1 score (see scoring.py).
"""

import requests
from .cache import cached

_CC_BASE         = "https://min-api.cryptocompare.com/data/exchanges/general"
_TIMEOUT         = 15
_LOW_N_THRESHOLD = 3   # flag jurisdictions with fewer than this many exchanges

# CryptoCompare uses free-text country names — inconsistent spelling.
# Map from ISO-3 -> set of accepted CC name variants (case-insensitive match).
# Checked against live /data/exchanges/general on 2026-09-25.
_CC_COUNTRY_NAMES: dict[str, set[str]] = {
    "USA": {"u.s.a", "united states", "usa"},
    "GBR": {"united kingdom", "uk", "great britain"},
    "SGP": {"singapore"},
    "AUS": {"australia"},
    "CHE": {"switzerland"},
    "HKG": {"hong kong"},
    "JPN": {"japan"},
    "IND": {"india"},
    "ARE": {"united arab emirates", "uae"},
    "CAN": {"canada"},
    "FRA": {"france"},
    "DEU": {"germany"},
    "NLD": {"netherlands"},
    "KOR": {"south korea", "republic of korea"},
    "BRA": {"brazil"},
    # Add more as target list expands
}

# CryptoCompare's 'Country' field is unreliable. 
# We maintain a manual exchange_name -> ISO-3 crosswalk for attribution.
EXCHANGE_JURISDICTION: dict[str, str] = {
    "Binance": "ARE",      # regulated entity used for HHI attribution
    "Coinbase": "USA",
    "Kraken": "USA",
    "OKX": "SGP",
    "Bybit": "ARE",
    "Crypto.com": "SGP",
    "Upbit": "KOR",
    "Bitflyer": "JPN",
    "Bithumb": "KOR",
    "KuCoin": "SGP",
    "Bitstamp": "GBR",
    "Luno": "GBR",
    "Mercado Bitcoin": "BRA",
    "WazirX": "IND",
    "CoinDCX": "IND",
    "CoinSwitch": "IND",
    "Bitfinex": "HKG",
    "Gemini": "USA",
    "NDAX": "CAN",
    "Bitpanda": "CHE",     # Example
    "Bitvavo": "NLD",
    "Paymium": "FRA",
    "Boerse Stuttgart Digital": "DEU",
    "Independent Reserve": "AUS",
    "BTC Markets": "AUS",
}


def _exchange_volume(info: dict) -> float:
    """
    Extract 24h volume from a CC exchange entry.
    TOTALVOLUME24H is a dict of {symbol: volume_float}, not a single number.
    Sum all symbol volumes as a proxy for total activity.
    """
    vol = info.get("TOTALVOLUME24H") or {}
    if isinstance(vol, (int, float)):
        return float(vol)
    if isinstance(vol, dict):
        return sum(float(v) for v in vol.values() if v)
    return 0.0


@cached(ttl_seconds=86_400)          # refresh daily — volume is fast-moving
def _fetch_exchange_data(api_key: str | None = None) -> dict:
    """
    Raw fetch from CryptoCompare /data/exchanges/general.
    Cached at the raw-payload level so all HHI calls share one network hit.
    api_key is included in the cache key to handle key-vs-keyless calls.
    """
    params: dict = {}
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(_CC_BASE, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"CryptoCompare API error: {exc}") from exc

    payload = resp.json()
    if payload.get("Response") == "Error":
        raise RuntimeError(f"CryptoCompare returned error: {payload.get('Message')}")

    return payload.get("Data", {})


def compute_hhi(
    iso3: str,
    api_key: str | None = None,
) -> dict:
    """
    Compute the HHI for exchanges domiciled/licensed in a jurisdiction.

    Parameters
    ----------
    iso3     : ISO-3166-1 alpha-3 code (e.g. "ARE", "BRA").
               Mapped to CryptoCompare full-name variants via _CC_COUNTRY_NAMES.
    api_key  : CryptoCompare API key (required for live data)

    Returns
    -------
    dict with keys:
        hhi           : float | None  — None if no exchanges found
        n_exchanges   : int
        total_volume  : float         — sum of 24h symbol volumes across matched exchanges
        low_coverage  : bool          — True if n_exchanges < _LOW_N_THRESHOLD
        exchanges     : list[str]     — names of exchanges matched
    """
    data = _fetch_exchange_data(api_key=api_key)

    accepted_names = _CC_COUNTRY_NAMES.get(iso3.upper(), set())
    if not accepted_names:
        # ISO-3 not in our map — fall back to exact match on the raw Country field
        accepted_names = {iso3.lower()}

    matched = {}
    for name, info in data.items():
        # First try the manual crosswalk
        assigned_iso3 = EXCHANGE_JURISDICTION.get(name)
        if assigned_iso3 == iso3.upper():
            matched_by_crosswalk = True
        else:
            # Fall back to checking the 'Country' string if no crosswalk exists
            matched_by_crosswalk = False
            raw_country = (info.get("Country") or "").lower()
            if raw_country not in accepted_names:
                continue
        
        vol = _exchange_volume(info)
        if vol > 0:
            matched[name] = vol

    n = len(matched)
    if n == 0:
        return {"hhi": None, "n_exchanges": 0, "total_volume": 0.0,
                "low_coverage": True, "exchanges": []}

    total = sum(matched.values())
    hhi   = sum((v / total) ** 2 for v in matched.values()) * 10_000

    return {
        "hhi":          round(hhi, 2),
        "n_exchanges":  n,
        "total_volume": round(total, 2),
        "low_coverage": n < _LOW_N_THRESHOLD,
        "exchanges":    sorted(matched.keys()),
    }
