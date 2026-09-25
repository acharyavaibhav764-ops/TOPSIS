from datetime import date
from ranking_engine.schema import RawFact, EvidenceTag
from .entity_resolution import normalize_entity_name

# C5 Local Rail Fit — manually maintained seed data
# This handles the VASP-openness component until Tier 3 extraction is ready.
_C5_SEED_RAW = [
    RawFact("United States", "C5", raw_value=0, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="Fragmented state regulations, uneven banking access for VASPs (2026 est)", as_of=date(2026, 9, 1)),
    RawFact("United Kingdom", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="FCA registered VASPs generally have access to Faster Payments (2026 est)", as_of=date(2026, 9, 1)),
    RawFact("Singapore", "C5", raw_value=2, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="MAS licensed entities strongly supported by local banking infrastructure", as_of=date(2026, 9, 1)),
    RawFact("Australia", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="De-banking remains a minor issue, but Tier 1 rails available to fully licensed AUSTRAC entities", as_of=date(2026, 9, 1)),
    RawFact("Switzerland", "C5", raw_value=2, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="Crypto Valley banks explicitly service digital assets", as_of=date(2026, 9, 1)),
    RawFact("Hong Kong", "C5", raw_value=2, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="HKMA issued strong circulars to banks supporting licensed VATPs", as_of=date(2026, 9, 1)),
    RawFact("Japan", "C5", raw_value=2, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="JFSA licensed exchanges integrated closely with domestic bank rails", as_of=date(2026, 9, 1)),
    RawFact("India", "C5", raw_value=0, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="Shadow ban history; UPI access for VASPs highly constrained/volatile", as_of=date(2026, 9, 1)),
    RawFact("UAE", "C5", raw_value=2, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="DFSA/CBUAE bank-VASP onboarding guidance, analyst read", as_of=date(2026, 9, 1)),
    RawFact("Canada", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="Restricted dealers have rails but ongoing strict federal scrutiny limits banking competition", as_of=date(2026, 9, 1)),
    RawFact("France", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="AMF PSAN registered entities have established EU banking pathways", as_of=date(2026, 9, 1)),
    RawFact("Germany", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="BaFin licensed custody providers actively banked by German institutions", as_of=date(2026, 9, 1)),
    RawFact("Netherlands", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="DNB registered entities face high compliance but have reliable Euro rail access", as_of=date(2026, 9, 1)),
    RawFact("South Korea", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="Mandatory real-name bank accounts bind exchanges tightly to commercial banks", as_of=date(2026, 9, 1)),
    RawFact("Brazil", "C5", raw_value=1, evidence_tag=EvidenceTag.ANALYST_ESTIMATE,
            source_citation="BCB Pix + VASP account-opening precedent, analyst read", as_of=date(2026, 9, 1)),
]

import dataclasses
C5_SEED = [dataclasses.replace(f, jurisdiction=normalize_entity_name(f.jurisdiction)) for f in _C5_SEED_RAW]
