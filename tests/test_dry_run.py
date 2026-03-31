"""Tests for pipeline dry-run mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthpipe.deidentify.ner import NEREntity
from healthpipe.deidentify.patterns import DetectionMethod, PHIMatch
from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.schema import ClinicalRecord, ResourceType
from healthpipe.pipeline import (
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
        csv_content = (
            "patient_id,notes\n"
            "P001,SSN: 123-45-6789\n"
        )
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
        csv_content = (
            "patient_id,ssn\n"
            "P001,123-45-6789\n"
        )
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
