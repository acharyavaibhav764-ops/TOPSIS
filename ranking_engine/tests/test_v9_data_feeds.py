"""
test_v9_data_feeds.py — Tests for the Tier-4 data feed stack.

Strategy: no real network calls. All external API responses are mocked with
unittest.mock.patch so the suite runs offline and deterministically.

Tests (all 8 from the plan's §8, plus field-name and schema validation):
  1. test_worldbank_picks_most_recent_nonnull_year
  2. test_score_gni_breakpoints
  3. test_score_revenue_depth_blends_gni_and_findex
  4. test_hhi_hand_calculated_example
  5. test_hhi_single_exchange_returns_max_concentration
  6. test_missing_jurisdiction_data_does_not_crash_refresh
  7. test_evidence_tags_match_source_table
  8. test_cache_respects_ttl
  9. test_refresh_triggers_staleness_check  (via as_of date, not mark_dirty)
 10. test_rawfact_has_correct_field_names   (schema validation)
 11. test_c5_tagged_analyst_estimate_not_verified
 12. test_low_coverage_hhi_produces_warning_not_exception
 13. test_score_wgi_range_clamp
 14. test_bready_load_and_retrieve

Run:
  python -m pytest ranking_engine/tests/test_v9_data_feeds.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pytest
import time
from datetime import date, datetime
from unittest.mock import patch, MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wb_api_response(value, year="2023"):
    """Fake World Bank API JSON: [metadata, [rows]]."""
    return [
        {"page": 1, "total": 1},
        [{"value": value, "date": year, "country": {"id": "ARE"}}],
    ]


def _wb_api_nulls_then_value(value, year="2022"):
    """World Bank response where first row is null (data lag), second has value."""
    return [
        {"page": 1, "total": 2},
        [
            {"value": None,  "date": "2024"},
            {"value": value, "date": year},
        ],
    ]


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoreGNI:

    def test_breakpoints_boundaries(self):
        from data_feeds.scoring import score_gni
        assert score_gni(2_000)  == 1.0
        assert score_gni(2_001)  == 4.0    # just over first breakpoint
        assert score_gni(8_000)  == 4.0
        assert score_gni(8_001)  == 6.0
        assert score_gni(20_000) == 6.0
        assert score_gni(20_001) == 8.0
        assert score_gni(40_000) == 8.0
        assert score_gni(40_001) == 10.0

    def test_none_returns_none(self):
        from data_feeds.scoring import score_gni
        assert score_gni(None) is None

    def test_uae_tier_scores_max(self):
        from data_feeds.scoring import score_gni
        assert score_gni(60_000) == 10.0  # UAE: ~60K GNI PPP


class TestScoreHHI:

    def test_hand_calculated_three_exchanges(self):
        """
        Three exchanges with volumes 50, 30, 20 (total 100):
        HHI = (0.5^2 + 0.3^2 + 0.2^2) * 10000 = (0.25 + 0.09 + 0.04) * 10000 = 3800
        score_hhi(3800) should return 3.0 (2500 < 3800 < 5000 bracket).
        """
        from data_feeds.scoring import score_hhi
        volumes = [50, 30, 20]
        total = sum(volumes)
        hhi = sum((v/total)**2 for v in volumes) * 10_000
        assert abs(hhi - 3800.0) < 0.01
        assert score_hhi(hhi) == 3.0

    def test_single_exchange_monopoly_hhi_10000(self):
        from data_feeds.scoring import score_hhi
        assert score_hhi(10_000.0) == 1.0     # monopoly → lowest C1 score

    def test_highly_fragmented_scores_high(self):
        from data_feeds.scoring import score_hhi
        assert score_hhi(400.0) == 10.0       # HHI < 500 → highest C1 score

    def test_none_returns_none(self):
        from data_feeds.scoring import score_hhi
        assert score_hhi(None) is None


class TestScoreRevenueDepth:

    def test_blends_gni_and_findex_70_30(self):
        """Revenue depth = 0.70 * score_gni + 0.30 * score_findex."""
        from data_feeds.scoring import score_revenue_depth, score_gni, score_findex
        gni     = 50_000          # score = 10.0
        findex  = 90              # score = 10.0
        blended = score_revenue_depth(gni, findex)
        expected = 0.70 * score_gni(gni) + 0.30 * score_findex(findex)
        assert abs(blended - expected) < 0.001

    def test_falls_back_to_gni_only_when_findex_missing(self):
        from data_feeds.scoring import score_revenue_depth, score_gni
        assert score_revenue_depth(30_000, None) == score_gni(30_000)

    def test_falls_back_to_findex_only_when_gni_missing(self):
        from data_feeds.scoring import score_revenue_depth, score_findex
        assert score_revenue_depth(None, 65) == score_findex(65)

    def test_both_none_returns_none(self):
        from data_feeds.scoring import score_revenue_depth
        assert score_revenue_depth(None, None) is None


class TestScoreWGI:

    def test_range_clamp_min(self):
        from data_feeds.scoring import score_wgi_regulatory_quality
        # -2.5 should map to exactly 1.0
        assert score_wgi_regulatory_quality(-2.5) == 1.0

    def test_range_clamp_max(self):
        from data_feeds.scoring import score_wgi_regulatory_quality
        # +2.5 should map to exactly 10.0
        assert score_wgi_regulatory_quality(2.5) == 10.0

    def test_midpoint(self):
        from data_feeds.scoring import score_wgi_regulatory_quality
        # 0.0 should map to 5.5 (midpoint of the linear scale)
        assert abs(score_wgi_regulatory_quality(0.0) - 5.5) < 0.01

    def test_none_returns_none(self):
        from data_feeds.scoring import score_wgi_regulatory_quality
        assert score_wgi_regulatory_quality(None) is None


# ---------------------------------------------------------------------------
# World Bank client tests (mocked network)
# ---------------------------------------------------------------------------

class TestWorldbankClient:

    def test_picks_most_recent_nonnull_year(self, monkeypatch, tmp_path):
        """
        If the API returns null for 2024 (data lag) and a value for 2022,
        get_indicator() must return the 2022 value, not None.
        Bug the plan explicitly called out.
        """
        import os
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.json.return_value = _wb_api_nulls_then_value(54_210.0, "2022")
        mock_resp.raise_for_status = MagicMock()

        with patch("data_feeds.worldbank_client.requests.get", return_value=mock_resp):
            # Bust cache to force a real call
            from data_feeds import worldbank_client
            worldbank_client.get_indicator.bust("ARE", "NY.GNP.PCAP.PP.CD")
            val, year = worldbank_client.get_indicator("ARE", "NY.GNP.PCAP.PP.CD")

        assert val  == 54_210.0
        assert year == "2022"

    def test_all_null_returns_none_none(self, monkeypatch, tmp_path):
        import os
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"page": 1}, [{"value": None, "date": "2023"}]]
        mock_resp.raise_for_status = MagicMock()

        with patch("data_feeds.worldbank_client.requests.get", return_value=mock_resp):
            from data_feeds import worldbank_client
            worldbank_client.get_indicator.bust("ZZZ", "NY.GNP.PCAP.PP.CD")
            val, year = worldbank_client.get_indicator("ZZZ", "NY.GNP.PCAP.PP.CD")

        assert val  is None
        assert year is None


# ---------------------------------------------------------------------------
# Cache TTL test
# ---------------------------------------------------------------------------

class TestCacheTTL:

    def test_cache_respects_ttl(self, monkeypatch, tmp_path):
        """
        After TTL expires, the function should be called again (not return stale data).
        """
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))

        call_count = 0

        from data_feeds.cache import cached

        @cached(ttl_seconds=1)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive.bust(1)          # ensure clean slate

        assert expensive(1) == 2   # miss → call
        assert call_count   == 1

        assert expensive(1) == 2   # hit → no call
        assert call_count   == 1

        time.sleep(1.1)            # wait for TTL to expire

        assert expensive(1) == 2   # miss again → call
        assert call_count   == 2


# ---------------------------------------------------------------------------
# HHI client tests (mocked)
# ---------------------------------------------------------------------------

class TestExchangeHHIClient:

    def _cc_data(self, exchanges: dict[str, tuple[str, float]]) -> dict:
        """Build a fake CryptoCompare /data/exchanges/general response."""
        return {name: {"Country": cc, "TOTALVOLUME24H": vol}
                for name, (cc, vol) in exchanges.items()}

    def test_hhi_hand_calculated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))
        fake_data = self._cc_data({
            "ExchA": ("AE", 50_000),
            "ExchB": ("AE", 30_000),
            "ExchC": ("AE", 20_000),
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Data": fake_data}
        mock_resp.raise_for_status = MagicMock()

        with patch("data_feeds.exchange_hhi_client.requests.get", return_value=mock_resp):
            from data_feeds import exchange_hhi_client
            exchange_hhi_client._fetch_exchange_data.bust()
            result = exchange_hhi_client.compute_hhi("AE")

        assert abs(result["hhi"] - 3800.0) < 1.0
        assert result["n_exchanges"] == 3
        assert result["low_coverage"] is False

    def test_single_exchange_monopoly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))
        fake_data = self._cc_data({"OnlyExch": ("AE", 100_000)})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Data": fake_data}
        mock_resp.raise_for_status = MagicMock()

        with patch("data_feeds.exchange_hhi_client.requests.get", return_value=mock_resp):
            from data_feeds import exchange_hhi_client
            exchange_hhi_client._fetch_exchange_data.bust(api_key=None)
            result = exchange_hhi_client.compute_hhi("AE")

        assert result["hhi"] == 10_000.0
        assert result["low_coverage"] is True    # n=1 < threshold

    def test_no_exchanges_returns_none_hhi(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))
        fake_data = self._cc_data({"ExchBR": ("BR", 100_000)})  # wrong country
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Data": fake_data}
        mock_resp.raise_for_status = MagicMock()

        with patch("data_feeds.exchange_hhi_client.requests.get", return_value=mock_resp):
            from data_feeds import exchange_hhi_client
            exchange_hhi_client._fetch_exchange_data.bust(api_key=None)
            result = exchange_hhi_client.compute_hhi("AE")

        assert result["hhi"] is None


# ---------------------------------------------------------------------------
# raw_fact_adapter — schema validation tests
# ---------------------------------------------------------------------------

class TestRawFactAdapter:

    def test_rawfact_has_correct_field_names(self):
        """
        Regression: the plan used 'criterion' and 'source' as field names.
        The real RawFact uses criterion_code and source_citation.
        This test will fail if the adapter reverts to the wrong names.
        """
        from data_feeds.raw_fact_adapter import make_fact
        from ranking_engine.schema import RawFact, EvidenceTag

        f = make_fact(
            jurisdiction="UAE",
            criterion_code="C2",
            raw_value=8.0,
            evidence_tag=EvidenceTag.VERIFIED_FACT,
            source_citation="World Bank NY.GNP.PCAP.PP.CD (year=2023)",
            as_of="2023",
        )
        assert isinstance(f, RawFact)
        assert f.criterion_code   == "C2"
        assert f.source_citation  == "World Bank NY.GNP.PCAP.PP.CD (year=2023)"
        assert isinstance(f.as_of, date)
        assert f.as_of == date(2023, 12, 31)   # annual year → Dec 31

    def test_year_string_parsed_to_dec_31(self):
        """World Bank returns year strings like '2022'; as_of must be Dec 31 of that year."""
        from data_feeds.raw_fact_adapter import _parse_date
        assert _parse_date("2022") == date(2022, 12, 31)

    def test_iso_date_string_parsed(self):
        from data_feeds.raw_fact_adapter import _parse_date
        assert _parse_date("2024-06-15") == date(2024, 6, 15)

    def test_none_as_of_returns_today(self):
        from data_feeds.raw_fact_adapter import _parse_date
        assert _parse_date(None) == date.today()

    def test_c5_tagged_analyst_estimate(self):
        """
        C5 base signal must be ANALYST_ESTIMATE, never VERIFIED_FACT.
        The VASP-openness component is missing; upgrading the tag silently
        would be a material misrepresentation of data quality.
        """
        from data_feeds.raw_fact_adapter import c5_base_signal
        from ranking_engine.schema import EvidenceTag
        f = c5_base_signal("UAE", 80.0, "2023")
        assert f.evidence_tag == EvidenceTag.ANALYST_ESTIMATE, (
            "C5 base signal must be ANALYST_ESTIMATE — VASP-openness is not automated"
        )

    def test_c1_tagged_analyst_estimate(self):
        """C1 HHI facts must be ANALYST_ESTIMATE (jurisdiction attribution is inferred)."""
        from data_feeds.raw_fact_adapter import c1_from_hhi
        from ranking_engine.schema import EvidenceTag
        hhi_result = {"hhi": 3800.0, "n_exchanges": 3,
                      "total_volume": 100_000, "low_coverage": False,
                      "exchanges": ["A", "B", "C"]}
        f = c1_from_hhi("UAE", hhi_result)
        assert f.evidence_tag == EvidenceTag.ANALYST_ESTIMATE

    def test_c2_tagged_verified_fact(self):
        """C2 World Bank data must be VERIFIED_FACT."""
        from data_feeds.raw_fact_adapter import c2_from_worldbank
        from ranking_engine.schema import EvidenceTag
        f = c2_from_worldbank("UAE", 8.5, "2023", "2022")
        assert f.evidence_tag == EvidenceTag.VERIFIED_FACT

    def test_c8_wgi_tagged_verified_fact(self):
        """C8 WGI data must be VERIFIED_FACT."""
        from data_feeds.raw_fact_adapter import c8_from_wgi
        from ranking_engine.schema import EvidenceTag
        f = c8_from_wgi("UAE", 0.94, "2023")
        assert f.evidence_tag == EvidenceTag.VERIFIED_FACT


# ---------------------------------------------------------------------------
# Evidence tag table conformance (catches silent tag regressions)
# ---------------------------------------------------------------------------

class TestEvidenceTagConformance:
    """
    Comprehensive check: every criterion/source pair in §6's table
    must produce the right EvidenceTag. If someone adds a new source
    and forgets to set the tag correctly, one of these fails.
    """

    def test_tag_table_completeness(self):
        from data_feeds.raw_fact_adapter import (
            c1_from_hhi, c2_from_worldbank, c5_base_signal, c8_from_wgi, c8_from_bready
        )
        from ranking_engine.schema import EvidenceTag

        hhi_res = {"hhi": 3000.0, "n_exchanges": 2, "total_volume": 1e6,
                   "low_coverage": True, "exchanges": ["A", "B"]}

        checks = [
            ("C2-WB",    c2_from_worldbank("UAE", 8.0, "2023", None),
             EvidenceTag.VERIFIED_FACT),
            ("C8-WGI",   c8_from_wgi("UAE", 0.9, "2023"),
             EvidenceTag.VERIFIED_FACT),
            ("C8-BREADY",c8_from_bready("UAE", 75.0, 2024),
             EvidenceTag.VERIFIED_FACT),
            ("C1-HHI",   c1_from_hhi("UAE", hhi_res),
             EvidenceTag.ANALYST_ESTIMATE),
            ("C5-BASE",  c5_base_signal("UAE", 70.0, "2023"),
             EvidenceTag.ANALYST_ESTIMATE),
        ]
        for label, fact, expected_tag in checks:
            assert fact.evidence_tag == expected_tag, (
                f"{label}: expected {expected_tag}, got {fact.evidence_tag}"
            )


# ---------------------------------------------------------------------------
# Refresh integration — missing data must not crash
# ---------------------------------------------------------------------------

class TestRefreshRobustness:

    def test_missing_jurisdiction_data_does_not_crash(self, monkeypatch, tmp_path):
        """
        A jurisdiction with no World Bank and no CryptoCompare coverage
        should produce zero facts and a warning — NOT an unhandled exception.
        """
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))

        # World Bank returns null for everything
        mock_wb = MagicMock()
        mock_wb.json.return_value = [{"page": 1}, [{"value": None, "date": "2023"}]]
        mock_wb.raise_for_status = MagicMock()

        # CryptoCompare returns no matching exchanges
        mock_cc = MagicMock()
        mock_cc.json.return_value = {"Data": {"SomeExch": {"Country": "SG", "TOTALVOLUME24H": 1e6}}}
        mock_cc.raise_for_status = MagicMock()

        with patch("data_feeds.worldbank_client.requests.get",    return_value=mock_wb), \
             patch("data_feeds.exchange_hhi_client.requests.get", return_value=mock_cc):

            from data_feeds import worldbank_client, exchange_hhi_client
            # Bust caches to force real calls
            worldbank_client.get_indicator.bust("ZZZ", "NY.GNP.PCAP.PP.CD")
            worldbank_client.get_indicator.bust("ZZZ", "FX.OWN.TOTL.ZS")
            worldbank_client.get_indicator.bust("ZZZ", "RQ.EST")
            exchange_hhi_client._fetch_exchange_data.bust()

            from tools.refresh_data_feeds import refresh_c2_c5, refresh_c1, refresh_c8
            jur = {"name": "Nowhere", "iso3": "ZZZ", "cc2": "ZZ"}

            c2c5_facts, c2c5_warn = refresh_c2_c5(jur, verbose=False)
            c1_facts,   c1_warn   = refresh_c1(jur, verbose=False, api_key=None)
            c8_facts,   c8_warn   = refresh_c8(jur, verbose=False)

        all_facts = c2c5_facts + c1_facts + c8_facts
        all_warns = c2c5_warn + c1_warn + c8_warn

        # Should produce no facts (all data missing)
        assert len(all_facts) == 0
        # Should produce warnings (not silence or exceptions)
        assert len(all_warns) >= 2

    def test_low_coverage_hhi_produces_warning_not_exception(self, monkeypatch, tmp_path):
        """n < 3 exchanges should produce a warning in the output, not raise."""
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))

        mock_cc = MagicMock()
        mock_cc.json.return_value = {
            "Data": {"OnlyExch": {"Country": "AE", "TOTALVOLUME24H": 50_000}}
        }
        mock_cc.raise_for_status = MagicMock()

        with patch("data_feeds.exchange_hhi_client.requests.get", return_value=mock_cc):
            from data_feeds import exchange_hhi_client
            exchange_hhi_client._fetch_exchange_data.bust(api_key=None)

            from tools.refresh_data_feeds import refresh_c1
            jur = {"name": "UAE", "iso3": "ARE", "cc2": "AE"}
            facts, warnings = refresh_c1(jur, verbose=False, api_key=None)

        # A fact may or may not be produced (caller decides whether to trust low-n)
        # but a warning MUST be raised about the low coverage
        assert any("low" in w.lower() or "coverage" in w.lower() or "exchange" in w.lower()
                   for w in warnings), (
            f"Expected a low-coverage warning but got: {warnings}"
        )


# ---------------------------------------------------------------------------
# B-READY manual ingest test
# ---------------------------------------------------------------------------

class TestBreadyIngest:

    def test_load_bready_release_and_retrieve(self, tmp_path, monkeypatch):
        """
        load_bready_release() should parse a CSV, cache it, and
        get_bready_score() should return the correct value.
        """
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))

        csv_path = tmp_path / "bready_2024.csv"
        csv_path.write_text(
            "iso3,score\n"
            "ARE,82.5\n"
            "BRA,61.0\n"
            "IDN,55.3\n"
        )

        from data_feeds.governance_index_client import load_bready_release, get_bready_score

        data = load_bready_release(str(csv_path), value_column="score")
        assert "ARE" in data
        assert abs(data["ARE"] - 82.5) < 0.001

        # get_bready_score must read from the same cache
        score = get_bready_score("ARE")
        assert score is not None
        assert abs(score - 82.5) < 0.001

    def test_bready_missing_iso_column_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MARKETSPHERE_CACHE_DIR", str(tmp_path))
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("country,score\nUAE,80.0\n")

        from data_feeds.governance_index_client import load_bready_release
        with pytest.raises(ValueError, match="Cannot find ISO-3 column"):
            load_bready_release(str(csv_path))
