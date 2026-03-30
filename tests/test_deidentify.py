"""Tests for the de-identification engine."""

from __future__ import annotations

from datetime import date

import pytest

from healthpipe.deidentify.date_shift import DateShifter
from healthpipe.deidentify.ner import ClinicalNER
from healthpipe.deidentify.patterns import PatternMatcher
from healthpipe.deidentify.safe_harbor import (
    SafeHarborConfig,
    SafeHarborEngine,
    deidentify,
)
from healthpipe.exceptions import DateShiftError
from healthpipe.ingest.schema import ClinicalDataset


class TestPatternMatcher:
    def setup_method(self) -> None:
        self.matcher = PatternMatcher()

    def test_detect_ssn(self) -> None:
        text = "SSN: 123-45-6789"
        matches = self.matcher.scan(text)
        ssn_matches = [m for m in matches if m.category == "SSN"]
        assert len(ssn_matches) >= 1
        assert "123-45-6789" in ssn_matches[0].original

    def test_detect_phone(self) -> None:
        text = "Call me at (555) 123-4567"
        matches = self.matcher.scan(text)
        phone_matches = [m for m in matches if m.category == "PHONE"]
        assert len(phone_matches) >= 1

    def test_detect_email(self) -> None:
        text = "Contact: john.doe@hospital.org"
        matches = self.matcher.scan(text)
        email_matches = [m for m in matches if m.category == "EMAIL"]
        assert len(email_matches) == 1

    def test_detect_mrn(self) -> None:
        text = "MRN-123456"
        matches = self.matcher.scan(text)
        mrn_matches = [m for m in matches if m.category == "MRN"]
        assert len(mrn_matches) >= 1

    def test_detect_ip(self) -> None:
        text = "From IP 192.168.1.100"
        matches = self.matcher.scan(text)
        ip_matches = [m for m in matches if m.category == "IP_ADDRESS"]
        assert len(ip_matches) == 1

    def test_redact(self) -> None:
        text = "SSN is 123-45-6789 and email is test@example.com"
        redacted, _matches = self.matcher.redact(text)
        assert "123-45-6789" not in redacted
        assert "test@example.com" not in redacted
        assert "[SSN]" in redacted
        assert "[EMAIL]" in redacted

    def test_redact_dict(self) -> None:
        data = {
            "name": "Patient record",
            "ssn": "123-45-6789",
            "nested": {"phone": "(555) 123-4567"},
        }
        redacted, matches = self.matcher.redact_dict(data)
        assert "123-45-6789" not in redacted["ssn"]
        assert len(matches) >= 2

    def test_no_false_positives_on_clean_text(self) -> None:
        text = "The patient was diagnosed with type 2 diabetes."
        matches = self.matcher.scan(text)
        # Should not produce SSN/phone matches on clean clinical text
        phi_matches = [
            m for m in matches if m.category in ("SSN", "PHONE", "EMAIL", "MRN")
        ]
        assert len(phi_matches) == 0

    def test_extra_patterns(self) -> None:
        matcher = PatternMatcher(extra_patterns={"CUSTOM_ID": r"CID-\d{6}"})
        matches = matcher.scan("Reference: CID-123456")
        custom = [m for m in matches if m.category == "CUSTOM_ID"]
        assert len(custom) == 1


class TestClinicalNER:
    def test_fallback_ner_detects_names(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Patient: John Smith was admitted on March 15."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1
        assert any("John" in e.text for e in names)

    def test_fallback_ner_title_prefix(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Dr. Jane Wilson ordered the lab work."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1

    def test_redact(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Patient: John Smith was seen today."
        redacted, _entities = ner.redact(text)
        assert "John Smith" not in redacted
        assert "[PATIENT_NAME]" in redacted

    def test_no_entities_in_clean_text(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Blood glucose level was 95 mg/dL."
        entities = ner.extract(text)
        # No patient names should be found
        assert len(entities) == 0


class TestDateShifter:
    def setup_method(self) -> None:
        self.shifter = DateShifter(
            shift_range=(-365, 365),
            salt="test-salt-123",
        )

    def test_consistent_offset(self) -> None:
        offset1 = self.shifter.get_offset_days("patient-001")
        offset2 = self.shifter.get_offset_days("patient-001")
        assert offset1 == offset2

    def test_different_patients_different_offsets(self) -> None:
        o1 = self.shifter.get_offset_days("patient-001")
        o2 = self.shifter.get_offset_days("patient-002")
        # Extremely unlikely to be equal with different IDs
        assert o1 != o2

    def test_offset_in_range(self) -> None:
        for i in range(100):
            offset = self.shifter.get_offset_days(f"patient-{i}")
            assert -365 <= offset <= 365

    def test_shift_date(self) -> None:
        original = date(2025, 6, 15)
        shifted = self.shifter.shift_date(original, "patient-001")
        assert shifted != original
        # Verify the offset matches
        diff = (shifted - original).days
        assert diff == self.shifter.get_offset_days("patient-001")

    def test_shift_preserves_intervals(self) -> None:
        d1 = date(2025, 1, 1)
        d2 = date(2025, 1, 15)
        s1 = self.shifter.shift_date(d1, "patient-001")
        s2 = self.shifter.shift_date(d2, "patient-001")
        assert (s2 - s1).days == (d2 - d1).days

    def test_shift_text(self) -> None:
        text = "Visit on 2025-03-15 and follow-up on 2025-04-15."
        shifted, count = self.shifter.shift_text(text, "patient-001")
        assert count == 2
        assert "2025-03-15" not in shifted

    def test_shift_record_data(self, sample_patient_data: dict) -> None:
        shifted = self.shifter.shift_record_data(sample_patient_data, "patient-001")
        assert shifted["birthDate"] != sample_patient_data["birthDate"]

    def test_collapse_age_over_89(self) -> None:
        # A 95-year-old
        birth = date(1930, 1, 1)
        collapsed = self.shifter.maybe_collapse_age(
            birth, reference_date=date(2026, 1, 1)
        )
        age = (date(2026, 1, 1) - collapsed).days // 365
        assert age == 90

    def test_no_collapse_under_89(self) -> None:
        birth = date(1960, 1, 1)
        result = self.shifter.maybe_collapse_age(birth, reference_date=date(2026, 1, 1))
        assert result == birth

    def test_invalid_shift_range(self) -> None:
        bad_shifter = DateShifter(shift_range=(365, -365), salt="test-salt")
        with pytest.raises(DateShiftError):
            bad_shifter.get_offset_days("test")


class TestSafeHarborEngine:
    @pytest.mark.asyncio
    async def test_full_deidentification(self, sample_dataset: ClinicalDataset) -> None:
        config = SafeHarborConfig(
            date_shift=True,
            date_shift_salt="test-salt-full-deid",
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        result = await engine.run(sample_dataset)

        assert len(result.records) == len(sample_dataset.records)
        assert len(result.audit_log.entries) > 0

    @pytest.mark.asyncio
    async def test_deidentify_convenience(
        self, sample_dataset: ClinicalDataset
    ) -> None:
        result = await deidentify(
            sample_dataset,
            llm_verification=False,
            date_shift_salt="test-salt-convenience",
        )
        assert isinstance(result.method, str)
        assert result.method == "safe_harbor"
        assert len(result.records) == 2

    @pytest.mark.asyncio
    async def test_phi_removed_from_patient(
        self, sample_dataset: ClinicalDataset
    ) -> None:
        config = SafeHarborConfig(
            date_shift=True,
            date_shift_salt="test-salt-phi-removed",
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        result = await engine.run(sample_dataset)

        # SSN should be redacted
        patient_data = result.records[0].data
        ssn_found = _deep_search(patient_data, "123-45-6789")
        assert not ssn_found, "SSN should have been redacted"

    @pytest.mark.asyncio
    async def test_email_removed(self, sample_dataset: ClinicalDataset) -> None:
        config = SafeHarborConfig(
            date_shift=False,
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        result = await engine.run(sample_dataset)

        patient_data = result.records[0].data
        email_found = _deep_search(patient_data, "john.smith@example.com")
        assert not email_found, "Email should have been redacted"


def _deep_search(obj: object, needle: str) -> bool:
    """Recursively search for a string in a nested structure."""
    if isinstance(obj, str):
        return needle in obj
    if isinstance(obj, dict):
        return any(_deep_search(v, needle) for v in obj.values())
    if isinstance(obj, list):
        return any(_deep_search(item, needle) for item in obj)
    return False
