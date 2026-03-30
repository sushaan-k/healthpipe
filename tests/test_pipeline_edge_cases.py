"""Tests for pipeline orchestration edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.fhir import FHIRSource
from healthpipe.pipeline import Pipeline, PipelineConfig, PipelineResult, ingest


class TestPipelineEmptyInput:
    @pytest.mark.asyncio
    async def test_pipeline_with_empty_sources(self) -> None:
        config = PipelineConfig(deidentify=False)
        pipeline = Pipeline(config)
        result = await pipeline.run([])
        assert isinstance(result, PipelineResult)
        assert len(result.raw_dataset.records) == 0
        assert result.deidentified is None

    @pytest.mark.asyncio
    async def test_pipeline_default_date_shift_salt_is_generated(self) -> None:
        config = PipelineConfig()
        pipeline = Pipeline(config)
        result = await pipeline.run([])
        assert isinstance(result, PipelineResult)
        assert result.deidentified is not None

    @pytest.mark.asyncio
    async def test_pipeline_deidentify_empty_dataset(self) -> None:
        """Pipeline should handle deidentification of empty dataset."""
        config = PipelineConfig(
            deidentify=True,
            deid_config={
                "use_fallback_ner": True,
                "llm_verification": False,
                "date_shift_salt": "test-empty-pipeline-salt",
            },
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([])
        assert result.deidentified is not None
        assert len(result.deidentified.records) == 0


class TestPipelineLineageDisabled:
    @pytest.mark.asyncio
    async def test_lineage_disabled(self, tmp_path: Path) -> None:
        csv_content = "patient_id,first_name\nP1,Alice\n"
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            track_lineage=False,
            deidentify=False,
        )
        pipeline = Pipeline(config)
        await pipeline.run([CSVSource(path=str(fpath))])

        assert pipeline.lineage is None


class TestPipelineSyntheticGeneration:
    @pytest.mark.asyncio
    async def test_pipeline_with_synthetic(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,glucose\nP001,John,110\nP002,Jane,95\nP003,Bob,130\n"
        )
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            deidentify=True,
            deid_config={
                "use_fallback_ner": True,
                "llm_verification": False,
                "date_shift_salt": "test-synthetic-pipeline-salt",
            },
            generate_synthetic=True,
            synthetic_n_patients=5,
            validate_synthetic=False,
            track_lineage=True,
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        assert result.synthetic is not None
        assert len(result.synthetic.records) > 0

    @pytest.mark.asyncio
    async def test_pipeline_synthetic_requires_deidentify(self, tmp_path: Path) -> None:
        """Synthetic generation is skipped if deidentify is off."""
        csv_content = "patient_id,first_name\nP1,Alice\n"
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            deidentify=False,
            generate_synthetic=True,
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        # generate_synthetic is True but deidentified is None, so synthetic
        # should be None
        assert result.synthetic is None


class TestPipelineMalformedFHIR:
    @pytest.mark.asyncio
    async def test_malformed_fhir_bundle(self, tmp_path: Path) -> None:
        """A FHIR Bundle with entries missing resourceType should be handled."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"id": "p1"}},
                {"resource": {"resourceType": "UnknownType", "id": "u1"}},
            ],
        }
        fpath = tmp_path / "bundle.json"
        fpath.write_text(json.dumps(bundle))

        dataset = await ingest([FHIRSource(url=str(fpath))])
        # Both entries should be skipped (no valid resourceType mapping)
        assert len(dataset.records) == 0

    @pytest.mark.asyncio
    async def test_fhir_bundle_with_mixed_resources(self, tmp_path: Path) -> None:
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "p1",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "o1",
                        "status": "final",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "c1",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Procedure",
                        "id": "pr1",
                    }
                },
            ],
        }
        fpath = tmp_path / "bundle.json"
        fpath.write_text(json.dumps(bundle))

        dataset = await ingest([FHIRSource(url=str(fpath))])
        assert dataset.patients.count() == 1
        assert dataset.observations.count() == 1
        assert dataset.conditions.count() == 1
        assert len(dataset.records) == 4


class TestPipelineConfig:
    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.deidentify is True
        assert config.generate_synthetic is False
        assert config.track_lineage is True
        assert config.synthetic_n_patients == 1000

    def test_custom_config(self) -> None:
        config = PipelineConfig(
            deidentify=False,
            generate_synthetic=True,
            synthetic_n_patients=500,
            synthetic_method="ctgan",
        )
        assert config.deidentify is False
        assert config.generate_synthetic is True
        assert config.synthetic_n_patients == 500

    def test_pipeline_result_defaults(self) -> None:
        result = PipelineResult()
        assert len(result.raw_dataset.records) == 0
        assert result.deidentified is None
        assert result.synthetic is None
        assert len(result.audit_log.entries) == 0
