"""Tests for ingest adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthpipe.exceptions import FHIRValidationError, IngestError
from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.fhir import FHIRSource
from healthpipe.ingest.hl7v2 import HL7Message, HL7v2Source


class TestFHIRSource:
    @pytest.mark.asyncio
    async def test_ingest_bundle_file(self, tmp_path: Path) -> None:
        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "p1",
                        "name": [{"family": "Test"}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "o1",
                        "status": "final",
                    }
                },
            ],
        }
        fpath = tmp_path / "bundle.json"
        fpath.write_text(json.dumps(bundle))

        source = FHIRSource(url=str(fpath))
        dataset = await source.ingest()

        assert dataset.patients.count() == 1
        assert dataset.observations.count() == 1

    @pytest.mark.asyncio
    async def test_invalid_bundle_type(self, tmp_path: Path) -> None:
        fpath = tmp_path / "bad.json"
        fpath.write_text(json.dumps({"resourceType": "Patient"}))

        source = FHIRSource(url=str(fpath))
        with pytest.raises(FHIRValidationError):
            await source.ingest()

    @pytest.mark.asyncio
    async def test_corrupt_json(self, tmp_path: Path) -> None:
        fpath = tmp_path / "corrupt.json"
        fpath.write_text("{not json")

        source = FHIRSource(url=str(fpath))
        with pytest.raises(IngestError):
            await source.ingest()


class TestHL7v2:
    def test_parse_message(self) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315120000||ADT^A01|MSG001|P|2.5\r"
            "PID|||12345^^^MRN||Smith^John||19850315|M|||"
            "123 Main St^^Springfield^IL^62704|||555-123-4567\r"
            "OBX|1|NM|2345-7^Glucose^LOINC||110|mg/dL|70-100|H||F\r"
        )
        msg = HL7Message(msg_text)
        assert msg.message_type == "ADT^A01"
        assert "PID" in msg.segments
        assert "OBX" in msg.segments

    def test_get_field(self) -> None:
        msg_text = "MSH|^~\\&|SRC|FAC\rPID|||12345||Smith^John\r"
        msg = HL7Message(msg_text)
        name = msg.get_field("PID", 5)
        assert "Smith" in name

    @pytest.mark.asyncio
    async def test_ingest_hl7_file(self, tmp_path: Path) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|1|P|2.5\r\n"
            "PID|||12345^^^MRN||Doe^Jane||19900101|F\r\n"
        )
        fpath = tmp_path / "message.hl7"
        fpath.write_text(msg_text)

        source = HL7v2Source(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() == 1


class TestCSVSource:
    @pytest.mark.asyncio
    async def test_auto_mapping(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,last_name,dob,gender\n001,John,Smith,1985-03-15,M\n"
        )
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() >= 1

    @pytest.mark.asyncio
    async def test_explicit_mapping(self, tmp_path: Path) -> None:
        csv_content = "pid,fname,lname\nP001,Alice,Wonder\n"
        fpath = tmp_path / "custom.csv"
        fpath.write_text(csv_content)

        source = CSVSource(
            path=str(fpath),
            mapping={"pid": "id", "fname": "name.given", "lname": "name.family"},
        )
        dataset = await source.ingest()
        assert dataset.patients.count() >= 1

    @pytest.mark.asyncio
    async def test_missing_file(self) -> None:
        source = CSVSource(path="/nonexistent/file.csv")
        with pytest.raises(IngestError):
            await source.ingest()

    @pytest.mark.asyncio
    async def test_empty_csv(self, tmp_path: Path) -> None:
        fpath = tmp_path / "empty.csv"
        fpath.write_text("col1,col2\n")
        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        assert len(dataset.records) == 0

    @pytest.mark.asyncio
    async def test_observation_columns(self, tmp_path: Path) -> None:
        csv_content = "patient_id,first_name,glucose,diagnosis\nP001,Jane,95.5,E11.9\n"
        fpath = tmp_path / "obs.csv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.observations.count() >= 1
        assert dataset.conditions.count() >= 1
