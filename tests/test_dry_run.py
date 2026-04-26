"""Tests for pipeline dry-run mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthpipe.deidentify.ner import NEREntity
from healthpipe.deidentify.patterns import DetectionMethod, PHIMatch
from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.schema import ClinicalRecord, ResourceType
from healthpipe.pipeline import (
    DryRunFinding,
    DryRunReport,
    Pipeline,
    PipelineConfig,
    _collect_strings_with_paths,
)


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_detects_phi_without_modifying_data(
        self, tmp_path: Path
    ) -> None:
        csv_content = (
            "patient_id,first_name,last_name,phone,ssn,email\n"
            "P001,John,Smith,555-123-4567,123-45-6789,john@example.com\n"
        )
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            dry_run=True,
            deid_config={"use_fallback_ner": True, "llm_verification": False},
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        # Data should NOT be de-identified
        assert result.deidentified is None

        # Dry-run report should be populated
        assert result.dry_run_report is not None
        report = result.dry_run_report
        assert isinstance(report, DryRunReport)
        assert report.total_records_scanned >= 1
        assert report.total_phi_found > 0

        # Should detect SSN, phone, email at minimum
        found_categories = set(report.categories_found)
        assert "SSN" in found_categories or "PHONE" in found_categories

    @pytest.mark.asyncio
    async def test_dry_run_returns_findings_with_confidence(
        self, tmp_path: Path
    ) -> None:
        csv_content = "patient_id,notes\nP001,SSN: 123-45-6789\n"
        fpath = tmp_path / "notes.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            dry_run=True,
            deid_config={"use_fallback_ner": True, "llm_verification": False},
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        assert result.dry_run_report is not None
        for finding in result.dry_run_report.findings:
            assert finding.confidence > 0
            assert finding.detection_method != ""
            assert finding.category != ""

    @pytest.mark.asyncio
    async def test_dry_run_preserves_original_data(self, tmp_path: Path) -> None:
        csv_content = "patient_id,ssn\nP001,123-45-6789\n"
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            dry_run=True,
            deid_config={"use_fallback_ner": True, "llm_verification": False},
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        # The raw dataset should still contain the original SSN
        for record in result.raw_dataset.records:
            data_str = str(record.data)
            assert "123-45-6789" in data_str

    @pytest.mark.asyncio
    async def test_dry_run_no_phi_found(self, tmp_path: Path) -> None:
        csv_content = "patient_id,diagnosis\nP001,diabetes\n"
        fpath = tmp_path / "clean.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            dry_run=True,
            deid_config={"use_fallback_ner": True, "llm_verification": False},
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        assert result.dry_run_report is not None
        assert result.dry_run_report.total_phi_found == 0
        assert result.dry_run_report.categories_found == []
        assert result.dry_run_report.findings_by_category == {}
        assert result.dry_run_report.findings_by_record == {}

    @pytest.mark.asyncio
    async def test_dry_run_skips_deidentification(self, tmp_path: Path) -> None:
        """Even with deidentify=True, dry_run should skip actual de-id."""
        csv_content = "patient_id,ssn\nP001,123-45-6789\n"
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            deidentify=True,
            dry_run=True,
            deid_config={"use_fallback_ner": True, "llm_verification": False},
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        assert result.deidentified is None
        assert result.dry_run_report is not None

    def test_scan_record_deduplicates_overlapping_findings(self) -> None:
        """Overlapping NER/pattern hits should collapse to one dry-run finding."""

        class StubNER:
            def extract(self, text: str) -> list[NEREntity]:
                return [
                    NEREntity(
                        text="123-45-6789",
                        label="DATE",
                        phi_category="SSN",
                        start=5,
                        end=16,
                        confidence=0.85,
                        detection_method=DetectionMethod.NER,
                    )
                ]

        class StubPatternMatcher:
            def scan(self, text: str) -> list[PHIMatch]:
                return [
                    PHIMatch(
                        category="SSN",
                        original="123-45-6789",
                        replacement="[SSN]",
                        start=5,
                        end=16,
                        confidence=0.95,
                        detection_method=DetectionMethod.PATTERN,
                    )
                ]

        record = ClinicalRecord(
            id="rec-1",
            resource_type=ResourceType.PATIENT,
            data={"notes": "SSN: 123-45-6789"},
        )
        findings = Pipeline._scan_record(record, StubNER(), StubPatternMatcher())

        assert len(findings) == 1
        assert findings[0].detection_method == DetectionMethod.PATTERN
        assert findings[0].original == "123-45-6789"

    @pytest.mark.asyncio
    async def test_dry_run_report_exposes_breakdowns(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,ssn,phone,email\n"
            "P001,123-45-6789,555-123-4567,john@example.com\n"
            "P002,987-65-4321,555-987-6543,jane@example.com\n"
        )
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            dry_run=True,
            deid_config={"use_fallback_ner": True, "llm_verification": False},
        )
        result = await Pipeline(config).run([CSVSource(path=str(fpath))])

        assert result.dry_run_report is not None
        report = result.dry_run_report
        assert sum(report.findings_by_record.values()) == report.total_phi_found
        assert len(report.findings_by_record) == report.total_records_scanned
        assert report.findings_by_category["SSN"] >= 1
        assert report.findings_by_category["PHONE"] >= 1
        assert report.findings_by_category["EMAIL"] >= 2
        top_category, top_count = report.top_categories(limit=1)[0]
        assert top_category in report.categories_found
        assert top_count >= 2

        payload = report.model_dump()
        assert payload["findings_by_category"] == report.findings_by_category
        assert payload["findings_by_record"] == report.findings_by_record
        assert report.high_risk_records(min_findings=2)
        assert all(count >= 2 for _record_id, count in report.high_risk_records())

    def test_dry_run_high_risk_records_boundaries(self) -> None:
        report = DryRunReport(
            findings=[
                DryRunFinding(record_id="b", category="EMAIL"),
                DryRunFinding(record_id="b", category="PHONE"),
                DryRunFinding(record_id="a", category="SSN"),
                DryRunFinding(record_id="c", category="EMAIL"),
                DryRunFinding(record_id="c", category="PHONE"),
            ]
        )

        assert report.high_risk_records(min_findings=1) == [
            ("b", 2),
            ("c", 2),
            ("a", 1),
        ]
        assert report.high_risk_records(min_findings=2) == [("b", 2), ("c", 2)]
        assert report.high_risk_records(min_findings=3) == []
        assert DryRunReport().high_risk_records() == []
        with pytest.raises(ValueError, match="min_findings must be at least 1"):
            report.high_risk_records(min_findings=0)

    def test_dry_run_baseline_is_phi_safe_and_stable(self) -> None:
        finding = DryRunFinding(
            record_id="record-1",
            field_path="patient.email",
            category="EMAIL",
            original="Patient@Example.Org",
            replacement="[EMAIL]",
            detection_method="pattern",
            confidence=0.95,
        )
        same_value_different_case = DryRunFinding(
            record_id="record-1",
            field_path="patient.email",
            category="EMAIL",
            original=" patient@example.org ",
        )

        report = DryRunReport(findings=[finding], total_records_scanned=1)
        baseline = report.to_baseline_dict()

        assert finding.fingerprint() == same_value_different_case.fingerprint()
        assert "Patient@Example.Org" not in str(baseline)
        assert baseline["findings"][0]["fingerprint"] == finding.fingerprint()
        assert "original_hash" in baseline["findings"][0]
        assert "original" not in baseline["findings"][0]

    def test_dry_run_compares_against_baseline(self) -> None:
        baseline_report = DryRunReport(
            findings=[
                DryRunFinding(
                    record_id="record-1",
                    field_path="patient.email",
                    category="EMAIL",
                    original="patient@example.org",
                ),
                DryRunFinding(
                    record_id="record-2",
                    field_path="patient.phone",
                    category="PHONE",
                    original="555-123-4567",
                ),
            ],
            total_records_scanned=2,
        )
        current_report = DryRunReport(
            findings=[
                DryRunFinding(
                    record_id="record-1",
                    field_path="patient.email",
                    category="EMAIL",
                    original="patient@example.org",
                ),
                DryRunFinding(
                    record_id="record-3",
                    field_path="patient.ssn",
                    category="SSN",
                    original="123-45-6789",
                ),
            ],
            total_records_scanned=3,
        )

        comparison = current_report.compare_to_baseline(
            baseline_report.to_baseline_dict()
        )

        assert comparison.baseline_count == 2
        assert comparison.current_count == 2
        assert comparison.new_count == 1
        assert comparison.resolved_count == 1
        assert comparison.unchanged_count == 1
        assert comparison.new_findings[0].category == "SSN"
        assert (
            current_report.with_findings(comparison.new_findings).total_phi_found == 1
        )


class TestCollectStringsWithPaths:
    def test_flat_dict(self) -> None:
        data = {"name": "John", "age": 30, "email": "j@e.com"}
        pairs = _collect_strings_with_paths(data)
        paths = [p for p, _v in pairs]
        assert "name" in paths
        assert "email" in paths

    def test_nested_dict(self) -> None:
        data = {"patient": {"name": "John", "address": {"city": "NYC"}}}
        pairs = _collect_strings_with_paths(data)
        paths = [p for p, _v in pairs]
        assert "patient.name" in paths
        assert "patient.address.city" in paths

    def test_list_indexing(self) -> None:
        data = {"names": ["Alice", "Bob"]}
        pairs = _collect_strings_with_paths(data)
        paths = [p for p, _v in pairs]
        assert "names[0]" in paths
        assert "names[1]" in paths

    def test_non_string_values_excluded(self) -> None:
        data = {"count": 5, "flag": True, "name": "test"}
        pairs = _collect_strings_with_paths(data)
        assert len(pairs) == 1
        assert pairs[0] == ("name", "test")
