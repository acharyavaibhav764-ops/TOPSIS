"""
schema.py — shared data contracts.

This is the ONLY file the LLM-extraction team and the ranking-model team
need to agree on. Everything downstream of this is owned by the ranking team.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import date, datetime


class EvidenceTag(str, Enum):
    VERIFIED_FACT    = "verified_fact"      # passed LLM-judge + multi-pass agreement
    ANALYST_ESTIMATE = "analyst_estimate"   # secondary source / proxy / human estimate
    UNRESOLVED_POLICY = "unresolved_policy" # draft law OR judge flagged a contradiction


@dataclass(frozen=True)
class RawFact:
    """
    One row the extraction pipeline produces — one per (jurisdiction, criterion).

    Fields
    ------
    jurisdiction    : e.g. "UAE", "Brazil"
    criterion_code  : "C1".."C8"
    raw_value       : the *unmapped* extracted number  (e.g. leverage cap = 5.0)
                      The ranking team owns the C1-C8 → 1-10 mapping.
    evidence_tag    : confidence level of this fact
    source_citation : verbatim source span / canonical URL — required for audit
    as_of           : date the source document was effective / scraped
    alt_raw_value   : only populated when evidence_tag == UNRESOLVED_POLICY;
                      represents the other plausible regime (bimodal simulation pair)
    """
    jurisdiction:    str
    criterion_code:  str
    raw_value:       float
    evidence_tag:    EvidenceTag
    source_citation: str
    as_of:           date
    alt_raw_value:   float | None = None


@dataclass
class StakeholderSurvey:
    """
    One decision-maker's BWM survey response.

    Fields
    ------
    best_idx  : 0-indexed criterion they chose as Most Important (a_B[best_idx] = 1)
    worst_idx : 0-indexed criterion they chose as Least Important (a_W[worst_idx] = 1)
    a_B       : comparison vector vs Best, length n_criteria — stored as list[float]
                so the whole dataclass is directly JSON-serializable.
    a_W       : comparison vector vs Worst, length n_criteria
    dm_id     : human-readable label ("founder_1", "cto") — used in trace plots + audit
    submitted : datetime of submission (for audit trail / staleness ordering)
    """
    best_idx:  int
    worst_idx: int
    a_B:       list[float]
    a_W:       list[float]
    dm_id:     str
    submitted: datetime


@dataclass
class RankingResult:
    """One output row per surviving jurisdiction after the full pipeline."""
    jurisdiction:       str
    closeness:          float                   # TOPSIS-RAD point-estimate closeness coefficient
    rank:               int
    rank_acceptability: dict[int, float]        # SMAA: {rank -> probability}, e.g. {1: 0.675}
    mc_90pct_interval:  tuple[float, float]     # evidence-calibrated MC: (p5, p95) of closeness
    expansion_wave:     int | None              # knapsack assignment; None = not selected yet
