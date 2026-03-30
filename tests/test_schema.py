"""Tests for the unified internal schema."""

from __future__ import annotations

from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    HumanName,
    ResourceCollection,
    ResourceType,
)


class TestHumanName:
    def test_full_name(self) -> None:
        name = HumanName(family="Smith", given=["John", "Michael"])
        assert name.full_name == "John Michael Smith"

    def test_full_name_with_prefix(self) -> None:
        name = HumanName(family="Doe", given=["Jane"], prefix=["Dr"])
        assert name.full_name == "Dr Jane Doe"

    def test_empty_name(self) -> None:
        name = HumanName()
        assert name.full_name == ""


class TestClinicalRecord:
    def test_checksum_generated(self, sample_patient_data: dict) -> None:
        record = ClinicalRecord(
            resource_type=ResourceType.PATIENT,
            data=sample_patient_data,
            source_format="FHIR_R4",
        )
        assert record.checksum
        assert len(record.checksum) == 64  # SHA-256 hex digest

    def test_checksum_deterministic(self, sample_patient_data: dict) -> None:
        r1 = ClinicalRecord(
            resource_type=ResourceType.PATIENT,
            data=sample_patient_data,
            source_format="FHIR_R4",
        )
        r2 = ClinicalRecord(
            resource_type=ResourceType.PATIENT,
            data=sample_patient_data,
            source_format="FHIR_R4",
        )
        assert r1.checksum == r2.checksum

    def test_different_data_different_checksum(self) -> None:
        r1 = ClinicalRecord(
            resource_type=ResourceType.PATIENT,
            data={"name": "Alice"},
            source_format="CSV",
        )
        r2 = ClinicalRecord(
            resource_type=ResourceType.PATIENT,
            data={"name": "Bob"},
            source_format="CSV",
        )
        assert r1.checksum != r2.checksum


class TestResourceCollection:
    def test_count(self) -> None:
        records = [
            ClinicalRecord(resource_type=ResourceType.PATIENT, data={}),
            ClinicalRecord(resource_type=ResourceType.PATIENT, data={}),
        ]
        coll = ResourceCollection(records)
        assert coll.count() == 2
        assert len(coll) == 2

    def test_iteration(self) -> None:
        records = [
            ClinicalRecord(resource_type=ResourceType.PATIENT, data={"id": "1"}),
        ]
        coll = ResourceCollection(records)
        for rec in coll:
            assert rec.data["id"] == "1"

    def test_to_dicts(self) -> None:
        records = [
            ClinicalRecord(resource_type=ResourceType.PATIENT, data={"a": 1}),
        ]
        coll = ResourceCollection(records)
        dicts = coll.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["data"]["a"] == 1


class TestClinicalDataset:
    def test_typed_accessors(self, sample_dataset: ClinicalDataset) -> None:
        assert sample_dataset.patients.count() == 1
        assert sample_dataset.observations.count() == 1
        assert sample_dataset.conditions.count() == 0

    def test_add_record(self) -> None:
        ds = ClinicalDataset()
        ds.add_record(ClinicalRecord(resource_type=ResourceType.PATIENT, data={}))
        assert len(ds.records) == 1

    def test_merge(self) -> None:
        ds1 = ClinicalDataset(
            records=[ClinicalRecord(resource_type=ResourceType.PATIENT, data={})]
        )
        ds2 = ClinicalDataset(
            records=[ClinicalRecord(resource_type=ResourceType.OBSERVATION, data={})]
        )
        merged = ds1.merge(ds2)
        assert len(merged.records) == 2

    def test_filter_by_type(self, sample_dataset: ClinicalDataset) -> None:
        patients = sample_dataset.filter_by_type(ResourceType.PATIENT)
        assert len(patients) == 1
