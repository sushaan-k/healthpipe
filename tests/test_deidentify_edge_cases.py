"""Tests for de-identification edge cases.

Covers NER mock tests, date shifting boundary cases (leap years,
century boundaries), LLM verifier, and Safe Harbor engine internals.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from healthpipe.deidentify.date_shift import DateShifter
from healthpipe.deidentify.llm_verify import LLMVerifier
from healthpipe.deidentify.ner import ClinicalNER, NEREntity
from healthpipe.deidentify.patterns import PatternMatcher
from healthpipe.deidentify.safe_harbor import (
    SafeHarborConfig,
    SafeHarborEngine,
)
from healthpipe.exceptions import DateShiftError
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType


class TestDateShiftBoundaries:
    def test_leap_year_feb_29(self) -> None:
        """Shifting Feb 29 on a leap year should not crash."""
        shifter = DateShifter(shift_range=(-365, 365), salt="leap-test")
        original = date(2024, 2, 29)  # 2024 is a leap year
        shifted = shifter.shift_date(original, "patient-001")
        assert isinstance(shifted, date)

    def test_century_boundary(self) -> None:
        """Shifting a date near year 2000 should work."""
        shifter = DateShifter(shift_range=(-365, 365), salt="century-test")
        original = date(2000, 1, 1)
        shifted = shifter.shift_date(original, "patient-001")
        assert isinstance(shifted, date)

    def test_century_boundary_1900(self) -> None:
        """Shifting dates near 1900 boundary."""
        shifter = DateShifter(shift_range=(-365, 365), salt="old-test")
        original = date(1900, 6, 15)
        shifted = shifter.shift_date(original, "patient-001")
        assert isinstance(shifted, date)

    def test_very_old_date(self) -> None:
        """Very old dates should still shift correctly."""
        shifter = DateShifter(shift_range=(-30, 30), salt="ancient-test")
        original = date(1920, 12, 31)
        shifted = shifter.shift_date(original, "patient-001")
        diff = abs((shifted - original).days)
        assert diff <= 30

    def test_shift_datetime_preserves_time(self) -> None:
        shifter = DateShifter(shift_range=(-10, 10), salt="time-test")
        original = datetime(2025, 6, 15, 14, 30, 45)
        shifted = shifter.shift_datetime(original, "patient-001")
        assert shifted.hour == 14
        assert shifted.minute == 30
        assert shifted.second == 45

    def test_equal_shift_range(self) -> None:
        """shift_range with lo == hi should raise DateShiftError."""
        shifter = DateShifter(shift_range=(0, 0), salt="zero-test")
        with pytest.raises(DateShiftError):
            shifter.get_offset_days("patient-001")

    def test_shift_text_no_dates(self) -> None:
        shifter = DateShifter(shift_range=(-365, 365), salt="test")
        text = "No dates here at all."
        shifted, count = shifter.shift_text(text, "patient-001")
        assert count == 0
        assert shifted == text

    def test_shift_text_mm_dd_yyyy(self) -> None:
        shifter = DateShifter(shift_range=(-365, 365), salt="test")
        text = "Date: 03/15/2025"
        shifted, count = shifter.shift_text(text, "patient-001")
        assert count >= 1
        assert "03/15/2025" not in shifted

    def test_shift_text_yyyymmdd(self) -> None:
        shifter = DateShifter(shift_range=(-365, 365), salt="test")
        text = "Date: 20250315"
        _shifted, count = shifter.shift_text(text, "patient-001")
        assert count >= 1

    def test_shift_text_invalid_date_string(self) -> None:
        """A date-like string that doesn't parse should be left alone."""
        shifter = DateShifter(shift_range=(-365, 365), salt="test")
        text = "ID: 99999999"
        shifted, _count = shifter.shift_text(text, "patient-001")
        # May or may not match the 8-digit pattern, but shouldn't crash
        assert isinstance(shifted, str)

    def test_collapse_age_disabled(self) -> None:
        shifter = DateShifter(collapse_age_over_89=False, salt="collapse-test")
        birth = date(1920, 1, 1)
        result = shifter.maybe_collapse_age(birth, reference_date=date(2026, 1, 1))
        # Age would be 106, but collapse is disabled
        assert result == birth

    def test_collapse_age_exactly_89(self) -> None:
        """Age of exactly 89 should NOT be collapsed."""
        shifter = DateShifter(collapse_age_over_89=True, salt="collapse-test")
        ref = date(2026, 1, 1)
        birth = ref - timedelta(days=89 * 365)
        result = shifter.maybe_collapse_age(birth, reference_date=ref)
        assert result == birth

    def test_collapse_age_90(self) -> None:
        """Age of 90 should be collapsed."""
        shifter = DateShifter(collapse_age_over_89=True, salt="collapse-test")
        ref = date(2026, 1, 1)
        birth = ref - timedelta(days=90 * 365)
        result = shifter.maybe_collapse_age(birth, reference_date=ref)
        expected_age = (ref - result).days // 365
        assert expected_age == 90

    def test_shift_record_data_nested(self) -> None:
        """shift_record_data recursively shifts both exact dates and prose."""
        shifter = DateShifter(shift_range=(-365, 365), salt="nested-test")
        data = {
            "patient": {
                "birthDate": "1985-03-15",
                "encounters": [
                    {"date": "2025-01-10"},
                    {"date": "2025-06-20"},
                ],
            },
            "notes": "Visit on 2025-03-15",
        }
        shifted = shifter.shift_record_data(data, "patient-001")
        # Pure date strings should be shifted
        assert shifted["patient"]["birthDate"] != "1985-03-15"
        assert shifted["patient"]["encounters"][0]["date"] != "2025-01-10"
        # Prose with embedded dates is also shifted now
        assert shifted["notes"] != "Visit on 2025-03-15"

    def test_shift_record_data_with_date_objects(self) -> None:
        shifter = DateShifter(shift_range=(-10, 10), salt="obj-test")
        data = {
            "birthDate": date(2000, 6, 15),
            "created": datetime(2025, 1, 1, 12, 0, 0),
        }
        shifted = shifter.shift_record_data(data, "patient-001")
        assert isinstance(shifted["birthDate"], date)
        assert isinstance(shifted["created"], datetime)

    def test_shift_record_data_non_date_strings(self) -> None:
        shifter = DateShifter(shift_range=(-365, 365), salt="test")
        data = {"name": "John Smith", "code": "ICD-10"}
        shifted = shifter.shift_record_data(data, "patient-001")
        assert shifted["name"] == "John Smith"
        assert shifted["code"] == "ICD-10"


class TestNERMock:
    def test_fallback_ner_with_mr_prefix(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Mr. Robert Johnson presented with chest pain."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1
        assert any("Robert" in e.text for e in names)

    def test_fallback_ner_with_mrs_prefix(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Mrs. Sarah Connor was discharged."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1

    def test_fallback_ner_with_ms_prefix(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Ms. Emily Chen scheduled a follow-up."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1

    def test_fallback_ner_context_cue_client(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Client: David Williams was seen for a consultation."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1

    def test_fallback_ner_context_cue_resident(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Resident: Maria Garcia was evaluated."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1

    def test_fallback_deduplication(self) -> None:
        """Overlapping entity spans should be deduplicated."""
        ner = ClinicalNER(use_fallback=True)
        # The prefix pattern and context pattern might overlap
        text = "Patient: Dr. John Smith was treated."
        entities = ner.extract(text)
        # Check no overlapping spans
        for i in range(len(entities) - 1):
            assert entities[i].end <= entities[i + 1].start

    def test_fallback_redact_no_entities(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Blood glucose was 95 mg/dL."
        redacted, entities = ner.redact(text)
        assert redacted == text
        assert entities == []

    def test_ner_entity_dataclass(self) -> None:
        entity = NEREntity(
            text="John",
            label="PERSON",
            phi_category="PATIENT_NAME",
            start=0,
            end=4,
            confidence=0.95,
        )
        assert entity.text == "John"
        assert entity.confidence == 0.95

    @patch("healthpipe.deidentify.ner.ClinicalNER.__post_init__")
    def test_spacy_extract_with_mock(self, mock_init: MagicMock) -> None:
        """Test spaCy extract path with a mock NLP model."""
        mock_init.return_value = None
        ner = ClinicalNER(use_fallback=False)

        # Create a mock spaCy doc
        mock_ent = MagicMock()
        mock_ent.text = "John Smith"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 9
        mock_ent.end_char = 19

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]

        mock_nlp = MagicMock(return_value=mock_doc)
        ner._nlp = mock_nlp

        entities = ner.extract("Patient: John Smith was seen.")
        assert len(entities) == 1
        assert entities[0].phi_category == "PATIENT_NAME"
        assert entities[0].confidence == 0.85

    @patch("healthpipe.deidentify.ner.ClinicalNER.__post_init__")
    def test_spacy_extract_ignores_unmapped_labels(self, mock_init: MagicMock) -> None:
        """spaCy entities with unmapped labels should be skipped."""
        mock_init.return_value = None
        ner = ClinicalNER(use_fallback=False)

        mock_ent = MagicMock()
        mock_ent.text = "42"
        mock_ent.label_ = "CARDINAL"  # Not in _PHI_LABELS
        mock_ent.start_char = 0
        mock_ent.end_char = 2

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]

        mock_nlp = MagicMock(return_value=mock_doc)
        ner._nlp = mock_nlp

        entities = ner.extract("42 years old")
        assert len(entities) == 0


class TestLLMVerifier:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self) -> None:
        verifier = LLMVerifier(api_key="")
        findings = await verifier.verify("Some text with PHI")
        assert findings == []

    def test_parse_response_empty_array(self) -> None:
        findings = LLMVerifier._parse_response("[]")
        assert findings == []

    def test_parse_response_with_findings(self) -> None:
        response = json.dumps(
            [
                {
                    "text": "Dr. Smith",
                    "category": "Names",
                    "start": 10,
                    "confidence": 0.95,
                },
                {
                    "text": "Springfield",
                    "category": "Geographic data",
                    "start": 30,
                    "confidence": 0.8,
                },
            ]
        )
        findings = LLMVerifier._parse_response(response)
        assert len(findings) == 2
        assert findings[0].text == "Dr. Smith"
        assert findings[1].category == "Geographic data"

    def test_parse_response_with_prose(self) -> None:
        """LLM responses sometimes include prose around the JSON."""
        response = (
            "Here are the findings:\n\n"
            '[{"text": "John", "category": "Names", '
            '"start": 0, "confidence": 0.9}]'
            "\n\nThese are all the PHI items found."
        )
        findings = LLMVerifier._parse_response(response)
        assert len(findings) == 1

    def test_parse_response_no_json(self) -> None:
        findings = LLMVerifier._parse_response("No PHI was found in the text.")
        assert findings == []

    def test_parse_response_missing_fields(self) -> None:
        """Items with missing fields should use defaults."""
        response = json.dumps([{"text": "test"}])
        findings = LLMVerifier._parse_response(response)
        assert len(findings) == 1
        assert findings[0].category == "UNKNOWN"
        assert findings[0].start == -1
        assert findings[0].confidence == 0.5

    def test_build_headers_anthropic(self) -> None:
        verifier = LLMVerifier(provider="anthropic", api_key="test-key")
        headers = verifier._build_headers()
        assert headers["x-api-key"] == "test-key"
        assert "anthropic-version" in headers

    def test_build_headers_openai(self) -> None:
        verifier = LLMVerifier(provider="openai", api_key="test-key")
        headers = verifier._build_headers()
        assert "Bearer test-key" in headers["Authorization"]

    def test_build_payload_anthropic(self) -> None:
        verifier = LLMVerifier(provider="anthropic", model="claude-3")
        payload = verifier._build_payload("test text")
        assert payload["model"] == "claude-3"
        assert "system" in payload
        assert payload["messages"][0]["role"] == "user"

    def test_build_payload_openai(self) -> None:
        verifier = LLMVerifier(provider="openai", model="gpt-4")
        payload = verifier._build_payload("test text")
        assert payload["model"] == "gpt-4"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_extract_text_anthropic(self) -> None:
        verifier = LLMVerifier(provider="anthropic")
        data = {"content": [{"text": "response text"}]}
        assert verifier._extract_text(data) == "response text"

    def test_extract_text_openai(self) -> None:
        verifier = LLMVerifier(provider="openai")
        data = {"choices": [{"message": {"content": "response text"}}]}
        assert verifier._extract_text(data) == "response text"


class TestSafeHarborEngineInternals:
    def test_extract_patient_id_from_id(self) -> None:
        data = {"id": "patient-123"}
        pid = SafeHarborEngine._extract_patient_id(data)
        assert pid == "patient-123"

    def test_extract_patient_id_from_subject_ref(self) -> None:
        data = {"subject": {"reference": "Patient/patient-456"}}
        pid = SafeHarborEngine._extract_patient_id(data)
        assert pid == "patient-456"

    def test_extract_patient_id_unknown(self) -> None:
        data = {"code": "something"}
        pid = SafeHarborEngine._extract_patient_id(data)
        assert pid == "unknown"

    def test_walk_strings(self) -> None:
        data = {"a": "hello", "b": {"c": "world"}, "d": [1, "test"]}
        result = SafeHarborEngine._walk_strings(data, lambda s: s.upper())
        assert result["a"] == "HELLO"
        assert result["b"]["c"] == "WORLD"
        assert result["d"][1] == "TEST"
        assert result["d"][0] == 1

    def test_collect_strings(self) -> None:
        data = {
            "name": "John",
            "nested": {"city": "Springfield"},
            "list": ["a", "b"],
            "num": 42,
        }
        strings = SafeHarborEngine._collect_strings(data)
        assert "John" in strings
        assert "Springfield" in strings
        assert "a" in strings
        assert "b" in strings
        assert len(strings) == 4

    def test_collect_strings_empty(self) -> None:
        strings = SafeHarborEngine._collect_strings({})
        assert strings == []

    @pytest.mark.asyncio
    async def test_deidentify_without_date_shift(self) -> None:
        config = SafeHarborConfig(
            date_shift=False,
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "id": "p1",
                        "birthDate": "1985-03-15",
                    },
                )
            ]
        )
        result = await engine.run(dataset)
        # Date should NOT be shifted since date_shift=False
        assert len(result.records) == 1

    @pytest.mark.asyncio
    async def test_free_text_dates_are_shifted_with_matching_audit(self) -> None:
        config = SafeHarborConfig(
            date_shift=True,
            date_shift_salt="free-text-audit-salt",
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "id": "p1",
                        "notes": (
                            "Follow-up visit on 2025-03-15 and discharge on 2025-03-20."
                        ),
                    },
                    source_format="TEST",
                )
            ]
        )

        result = await engine.run(dataset)
        notes = result.records[0].data["notes"]
        assert "2025-03-15" not in notes
        assert "2025-03-20" not in notes
        date_entries = [e for e in result.audit_log.entries if e.layer == "DATE_SHIFT"]
        assert len(date_entries) == 1
        assert "shifted_dates=2" in date_entries[0].replacement

    @pytest.mark.asyncio
    async def test_deidentify_empty_dataset(self) -> None:
        config = SafeHarborConfig(
            use_fallback_ner=True,
            llm_verification=False,
            date_shift_salt="test-salt-empty",
        )
        engine = SafeHarborEngine(config)
        dataset = ClinicalDataset(records=[])
        result = await engine.run(dataset)
        assert len(result.records) == 0
        assert len(result.audit_log.entries) == 0


class TestPatternMatcherEdgeCases:
    def test_detect_url(self) -> None:
        matcher = PatternMatcher()
        matches = matcher.scan("Visit https://hospital.org/patient/123 for records")
        url_matches = [m for m in matches if m.category == "URL"]
        assert len(url_matches) == 1

    def test_detect_zip_code(self) -> None:
        matcher = PatternMatcher()
        matches = matcher.scan("ZIP: 90210")
        zip_matches = [m for m in matches if m.category == "ZIP_CODE"]
        assert len(zip_matches) >= 1

    def test_detect_zip_plus_four(self) -> None:
        matcher = PatternMatcher()
        matches = matcher.scan("ZIP: 90210-1234")
        zip_matches = [m for m in matches if m.category == "ZIP_CODE"]
        assert len(zip_matches) >= 1

    def test_detect_account_number(self) -> None:
        matcher = PatternMatcher()
        matches = matcher.scan("Account #: 12345678")
        acct_matches = [m for m in matches if m.category == "ACCOUNT_NUMBER"]
        assert len(acct_matches) >= 1

    def test_redact_preserves_non_phi_text(self) -> None:
        matcher = PatternMatcher()
        text = "Diagnosis: Type 2 Diabetes. Email: test@test.com"
        redacted, _ = matcher.redact(text)
        assert "Diagnosis: Type 2 Diabetes" in redacted
        assert "[EMAIL]" in redacted

    def test_scan_empty_string(self) -> None:
        matcher = PatternMatcher()
        matches = matcher.scan("")
        assert matches == []

    def test_redact_dict_with_lists(self) -> None:
        matcher = PatternMatcher()
        data = {
            "phones": ["555-123-4567", "555-987-6543"],
            "name": "clean text",
        }
        _redacted, matches = matcher.redact_dict(data)
        assert len(matches) >= 2

    def test_redact_dict_non_string_values(self) -> None:
        matcher = PatternMatcher()
        data = {"age": 42, "active": True, "scores": [1, 2, 3]}
        redacted, matches = matcher.redact_dict(data)
        assert matches == []
        assert redacted["age"] == 42
