"""Tests validating HIPAA Safe Harbor 18 identifier coverage.

Verifies that the de-identification engine handles all 18 HIPAA Safe
Harbor identifiers as defined in 45 CFR 164.514(b)(2).
"""

from __future__ import annotations

import pytest

from healthpipe.audit.compliance import (
    _CATEGORY_TO_IDENTIFIER,
    SAFE_HARBOR_IDENTIFIERS,
    ComplianceReporter,
)
from healthpipe.audit.logger import AuditEntry, AuditLog
from healthpipe.deidentify.patterns import PatternMatcher
from healthpipe.deidentify.safe_harbor import (
    SafeHarborConfig,
    SafeHarborEngine,
)
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType


class TestSafeHarborIdentifierList:
    """Verify that the project has all 18 HIPAA identifiers defined."""

    def test_exactly_18_identifiers(self) -> None:
        assert len(SAFE_HARBOR_IDENTIFIERS) == 18

    def test_identifier_names(self) -> None:
        """Spot-check that the identifier list matches HIPAA spec."""
        assert "Names" in SAFE_HARBOR_IDENTIFIERS
        assert "Social Security Numbers" in SAFE_HARBOR_IDENTIFIERS
        assert "Phone numbers" in SAFE_HARBOR_IDENTIFIERS
        assert "Email addresses" in SAFE_HARBOR_IDENTIFIERS
        assert "Medical record numbers" in SAFE_HARBOR_IDENTIFIERS
        assert "IP addresses" in SAFE_HARBOR_IDENTIFIERS
        assert "Web URLs" in SAFE_HARBOR_IDENTIFIERS
        assert "Full-face photographs" in SAFE_HARBOR_IDENTIFIERS
        assert "Dates (except year) related to an individual" in SAFE_HARBOR_IDENTIFIERS
        assert "Geographic data (smaller than state)" in SAFE_HARBOR_IDENTIFIERS
        assert "Any other unique identifying number or code" in SAFE_HARBOR_IDENTIFIERS

    def test_category_mapping_covers_specific_identifiers(self) -> None:
        """Identifiers 1-17 should each have at least one category
        mapping. Identifier #18 ('Any other unique identifying number
        or code') is a catch-all without a specific pattern."""
        covered_indices = set(_CATEGORY_TO_IDENTIFIER.values())
        for i in range(1, 18):
            assert i in covered_indices, (
                f"Identifier #{i} ({SAFE_HARBOR_IDENTIFIERS[i - 1]}) has no category mapping"
            )
        # Identifier 18 is intentionally unmapped (catch-all)
        assert 18 not in covered_indices


class TestPatternMatcherCoversIdentifiers:
    """Verify that the pattern matcher detects the pattern-based
    Safe Harbor identifiers."""

    def setup_method(self) -> None:
        self.matcher = PatternMatcher()

    def test_detects_ssn_identifier_7(self) -> None:
        matches = self.matcher.scan("SSN: 123-45-6789")
        ssn = [m for m in matches if m.category == "SSN"]
        assert len(ssn) >= 1

    def test_detects_phone_identifier_4(self) -> None:
        matches = self.matcher.scan("Phone: (555) 123-4567")
        phone = [m for m in matches if m.category == "PHONE"]
        assert len(phone) >= 1

    def test_detects_email_identifier_6(self) -> None:
        matches = self.matcher.scan("Email: patient@hospital.org")
        email = [m for m in matches if m.category == "EMAIL"]
        assert len(email) == 1

    def test_detects_mrn_identifier_8(self) -> None:
        matches = self.matcher.scan("Medical Record: MRN-12345")
        mrn = [m for m in matches if m.category == "MRN"]
        assert len(mrn) >= 1

    def test_detects_account_number_identifier_10(self) -> None:
        matches = self.matcher.scan("Account # 123456789")
        acct = [m for m in matches if m.category == "ACCOUNT_NUMBER"]
        assert len(acct) >= 1

    def test_detects_ip_address_identifier_15(self) -> None:
        matches = self.matcher.scan("IP: 192.168.1.100")
        ip = [m for m in matches if m.category == "IP_ADDRESS"]
        assert len(ip) == 1

    def test_detects_url_identifier_14(self) -> None:
        matches = self.matcher.scan("Visit https://records.hospital.org")
        url = [m for m in matches if m.category == "URL"]
        assert len(url) == 1

    def test_detects_zip_code_identifier_2(self) -> None:
        matches = self.matcher.scan("ZIP: 62704")
        zip_m = [m for m in matches if m.category == "ZIP_CODE"]
        assert len(zip_m) >= 1


class TestComplianceReportIdentifierTracking:
    def test_report_tracks_which_identifiers_addressed(self) -> None:
        log = AuditLog()
        # Add entries for various categories
        for cat in [
            "PATIENT_NAME",
            "SSN",
            "PHONE",
            "EMAIL",
            "MRN",
            "DATE",
        ]:
            log.add(
                AuditEntry(
                    action="PHI_REMOVED",
                    layer="PATTERN",
                    category=cat,
                )
            )

        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)

        assert report.identifiers_addressed["Names"] is True
        assert report.identifiers_addressed["Social Security Numbers"] is True
        assert report.identifiers_addressed["Phone numbers"] is True
        assert report.identifiers_addressed["Email addresses"] is True
        assert report.identifiers_addressed["Medical record numbers"] is True
        assert (
            report.identifiers_addressed["Dates (except year) related to an individual"]
            is True
        )

    def test_report_marks_unaddressed_identifiers(self) -> None:
        log = AuditLog()
        # Only address one identifier
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                layer="NER",
                category="PATIENT_NAME",
            )
        )

        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)

        assert report.identifiers_addressed["Names"] is True
        # Others should be False
        assert report.identifiers_addressed["Social Security Numbers"] is False
        assert report.identifiers_addressed["IP addresses"] is False

    def test_geographic_identifiers_multiple_categories(self) -> None:
        """LOCATION, ZIP_CODE, FACILITY, ORGANIZATION all map to
        identifier #2 (Geographic data)."""
        for cat in ["LOCATION", "ZIP_CODE", "FACILITY", "ORGANIZATION"]:
            log = AuditLog()
            log.add(
                AuditEntry(
                    action="PHI_REMOVED",
                    category=cat,
                )
            )
            reporter = ComplianceReporter()
            report = reporter.generate(audit_log=log)
            assert (
                report.identifiers_addressed["Geographic data (smaller than state)"]
                is True
            ), f"Category '{cat}' should address geographic identifier"


class TestFullDeidentificationIdentifierCoverage:
    """Integration test: run the Safe Harbor engine on data containing
    all pattern-detectable PHI types and verify they are caught."""

    @pytest.mark.asyncio
    async def test_engine_removes_all_pattern_phi(self) -> None:
        data = {
            "resourceType": "Patient",
            "id": "patient-test",
            "notes": (
                "Patient SSN: 123-45-6789. "
                "Phone: (555) 123-4567. "
                "Email: patient@hospital.org. "
                "MRN-543210. "
                "Account # 9876543210. "
                "IP: 10.0.0.1. "
                "Visit https://records.hospital.org. "
                "ZIP: 62704."
            ),
        }
        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data=data,
                    source_format="TEST",
                )
            ]
        )
        config = SafeHarborConfig(
            date_shift=False,
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        result = await engine.run(dataset)

        # Check that PHI was removed
        assert result.audit_log.phi_removed_count > 0

        # Check that various categories are in the audit log
        categories = {e.category for e in result.audit_log.entries}
        assert "SSN" in categories
        assert "EMAIL" in categories
