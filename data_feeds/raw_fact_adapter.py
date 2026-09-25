"""
data_feeds/raw_fact_adapter.py — Convert feed outputs to real RawFact dataclass instances.

The plan flagged one assumption: "if the real dataclass has different field names,
it's a 5-minute rename." It does have different names:
  - Plan used:  criterion   → real field: criterion_code
  - Plan used:  source      → real field: source_citation
  - Plan used:  as_of str   → real field: as_of (date object, not string)

All three are corrected here. This file is the only place in the Tier-4 stack
that imports from ranking_engine — all other data_feeds/ modules are pure
fetch/score and have no dependency on the ranking engine.

Evidence-tag rules (from the plan's §6 table):
  C2 (GNI, Findex) WB API      → VERIFIED_FACT
  C8 (WGI)         WB API      → VERIFIED_FACT
  C8 (B-READY)     bulk ingest → VERIFIED_FACT
  C1 (HHI)         computed    → ANALYST_ESTIMATE  (jurisdiction attribution inferred)
  C5 (base signal) WB proxy    → ANALYST_ESTIMATE  (VASP-openness missing)
  C5 (override)    manual      → ANALYST_ESTIMATE  (until Tier 3 exists)
"""

from __future__ import annotations

import sys
import os
from datetime import date, datetime

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ranking_engine.schema import RawFact, EvidenceTag


def _parse_date(as_of: str | date | None) -> date:
    """
    Normalise as_of to a date object.

    The World Bank API returns year strings ("2023"); we convert those to
    Dec 31 of that year (last day of the data period). If as_of is already
    a date, it's returned unchanged. If None, today's date is used.
    """
    if as_of is None:
        return date.today()
    if isinstance(as_of, date):
        return as_of
    s = str(as_of).strip()
    if len(s) == 4 and s.isdigit():
        return date(int(s), 12, 31)      # annual indicator → end of year
    try:
        return date.fromisoformat(s)
    except ValueError:
        return date.today()


def make_fact(
    jurisdiction:    str,
    criterion_code:  str,
    raw_value:       float,
    evidence_tag:    EvidenceTag,
    source_citation: str,
    as_of:           str | date | None = None,
    alt_raw_value:   float | None = None,
) -> RawFact:
    """
    Construct a RawFact from feed data.

    Parameters map directly to the RawFact dataclass fields — using the
    real field names (criterion_code, source_citation) not the plan's aliases.
    """
    return RawFact(
        jurisdiction=jurisdiction,
        criterion_code=criterion_code,
        raw_value=raw_value,
        evidence_tag=evidence_tag,
        source_citation=source_citation,
        as_of=_parse_date(as_of),
        alt_raw_value=alt_raw_value,
    )


# ---------------------------------------------------------------------------
# Per-criterion factory functions
# (thin wrappers that lock in the evidence tag so callers can't forget it)
# ---------------------------------------------------------------------------

def c2_from_worldbank(
    jurisdiction: str,
    score: float,
    gni_year: str | None,
    findex_year: str | None,
) -> RawFact:
    """C2 Revenue Depth fact from World Bank GNI+Findex blend."""
    year = gni_year or findex_year
    return make_fact(
        jurisdiction=jurisdiction,
        criterion_code="C2",
        raw_value=score,
        evidence_tag=EvidenceTag.VERIFIED_FACT,
        source_citation=(
            f"World Bank: NY.GNP.PCAP.PP.CD + FX.OWN.TOTL.ZS "
            f"(year={year})"
        ),
        as_of=year,
    )


def c1_from_hhi(
    jurisdiction: str,
    hhi_result:   dict,   # output of exchange_hhi_client.compute_hhi()
    data_date:    str | date | None = None,
) -> RawFact | None:
    """
    C1 Demand Scale fact from HHI computation.

    Returns None (skip this fact) if:
      - hhi_result["hhi"] is None (no exchanges found for this jurisdiction)
      - hhi_result["low_coverage"] is True AND score would be based on < 3 exchanges
        — caller decides; if you want to skip low-coverage, check result.low_coverage
          before calling this function.
    """
    from .scoring import score_hhi
    if hhi_result["hhi"] is None:
        return None

    score = score_hhi(hhi_result["hhi"])
    n     = hhi_result["n_exchanges"]
    exch  = ", ".join(hhi_result["exchanges"][:5])  # cap citation length

    return make_fact(
        jurisdiction=jurisdiction,
        criterion_code="C1",
        raw_value=score,
        evidence_tag=EvidenceTag.ANALYST_ESTIMATE,   # inferred jurisdiction attribution
        source_citation=(
            f"CryptoCompare /data/exchanges/general — "
            f"n={n} exchanges ({exch}); "
            f"HHI={hhi_result['hhi']:.0f}"
        ),
        as_of=data_date or date.today(),
    )


def c8_from_wgi(
    jurisdiction: str,
    estimate:     float,
    year:         str | None,
) -> RawFact:
    """C8 Regulatory Volatility fact from WGI Regulatory Quality."""
    from .scoring import score_wgi_regulatory_quality
    return make_fact(
        jurisdiction=jurisdiction,
        criterion_code="C8",
        raw_value=score_wgi_regulatory_quality(estimate),
        evidence_tag=EvidenceTag.VERIFIED_FACT,
        source_citation=f"World Bank WGI RQ.EST (year={year})",
        as_of=year,
    )


def c8_from_bready(
    jurisdiction: str,
    score:        float,   # already on 0-100 or 0-10 — normalise in scoring.py
    release_year: int,
) -> RawFact:
    """C8 fact from B-READY bulk-ingest. Tagged VERIFIED_FACT (official release)."""
    return make_fact(
        jurisdiction=jurisdiction,
        criterion_code="C8",
        raw_value=score,
        evidence_tag=EvidenceTag.VERIFIED_FACT,
        source_citation=f"World Bank B-READY {release_year} release (manual ingest)",
        as_of=date(release_year, 12, 31),
    )


def c5_base_signal(
    jurisdiction:          str,
    account_ownership_pct: float,
    year:                  str | None,
) -> RawFact:
    """
    C5 Local Rail Fit — automated base signal only.

    IMPORTANT: this fact is tagged ANALYST_ESTIMATE because the VASP-openness
    component (whether regulators allow VASPs to plug into payment rails) is
    not covered by this automated feed. That component requires Tier-3 extraction.
    Do not upgrade this tag to VERIFIED_FACT until the full criterion is automated.
    """
    from .scoring import score_c5_base_signal
    return make_fact(
        jurisdiction=jurisdiction,
        criterion_code="C5",
        raw_value=score_c5_base_signal(account_ownership_pct),
        evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
        source_citation=(
            f"World Bank FX.OWN.TOTL.ZS (digital-payments proxy, year={year}); "
            f"VASP-openness component NOT included — see Tier 3"
        ),
        as_of=year,
    )


def c5_manual_override(
    jurisdiction: str,
    score:        float,
    rationale:    str,
    as_of:        date | None = None,
) -> RawFact:
    """
    C5 manual override for VASP-openness flag.

    Use this until Tier 3 can extract the VASP-openness component from
    regulatory text. Remains ANALYST_ESTIMATE.
    """
    return make_fact(
        jurisdiction=jurisdiction,
        criterion_code="C5",
        raw_value=score,
        evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
        source_citation=f"Manual override: {rationale}",
        as_of=as_of or date.today(),
    )
