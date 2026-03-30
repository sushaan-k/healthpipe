"""Tests for the pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.fhir import FHIRSource
from healthpipe.pipeline import Pipeline, PipelineConfig, PipelineResult, ingest


class TestIngest:
    @pytest.mark.asyncio
    async def test_ingest_single_source(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,last_name,dob\n"
            "P001,Alice,Smith,1990-01-15\n"
            "P002,Bob,Jones,1985-06-20\n"
        )
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        dataset = await ingest([CSVSource(path=str(fpath))])
        assert len(dataset.records) >= 2

    @pytest.mark.asyncio
    async def test_ingest_multiple_sources(self, tmp_path: Path) -> None:
        csv1 = tmp_path / "a.csv"
        csv1.write_text("patient_id,first_name\nP1,Alice\n")
        csv2 = tmp_path / "b.csv"
        csv2.write_text("patient_id,first_name\nP2,Bob\n")

        dataset = await ingest(
            [
                CSVSource(path=str(csv1)),
                CSVSource(path=str(csv2)),
            ]
        )
        assert len(dataset.records) >= 2

    @pytest.mark.asyncio
    async def test_ingest_fhir_bundle(self, tmp_path: Path) -> None:
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1"}},
            ],
        }
        fpath = tmp_path / "bundle.json"
        fpath.write_text(json.dumps(bundle))

        dataset = await ingest([FHIRSource(url=str(fpath))])
        assert dataset.patients.count() == 1

    @pytest.mark.asyncio
    async def test_ingest_empty_list(self) -> None:
        dataset = await ingest([])
        assert len(dataset.records) == 0


class TestPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,last_name,dob,phone,ssn\n"
            "P001,John,Smith,1985-03-15,555-123-4567,123-45-6789\n"
            "P002,Jane,Doe,1990-07-20,555-987-6543,987-65-4321\n"
        )
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            deidentify=True,
            deid_config={
                "use_fallback_ner": True,
                "llm_verification": False,
                "date_shift_salt": "test-pipeline-salt",
            },
            generate_synthetic=False,
            track_lineage=True,
        )
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        assert isinstance(result, PipelineResult)
        assert len(result.raw_dataset.records) >= 2
        assert result.deidentified is not None
        assert len(result.audit_log.entries) > 0

    @pytest.mark.asyncio
    async def test_pipeline_no_deidentify(self, tmp_path: Path) -> None:
        csv_content = "patient_id,first_name\nP1,Alice\n"
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(deidentify=False)
        pipeline = Pipeline(config)
        result = await pipeline.run([CSVSource(path=str(fpath))])

        assert result.deidentified is None
        assert result.synthetic is None

    @pytest.mark.asyncio
    async def test_lineage_tracking(self, tmp_path: Path) -> None:
        csv_content = "patient_id,first_name\nP1,Alice\n"
        fpath = tmp_path / "data.csv"
        fpath.write_text(csv_content)

        config = PipelineConfig(
            track_lineage=True,
            deid_config={"date_shift_salt": "test-lineage-salt"},
        )
        pipeline = Pipeline(config)
        await pipeline.run([CSVSource(path=str(fpath))])

        assert pipeline.lineage is not None
        assert len(pipeline.lineage.all_records) > 0
