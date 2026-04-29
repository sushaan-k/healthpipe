"""Tests for dataset integrity validation."""

from __future__ import annotations

from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType
from healthpipe.validation import DatasetValidator, validate_dataset


def test_validate_dataset_passes_linked_patient_records() -> None:
    patient = ClinicalRecord(
        id="record-patient",
        resource_type=ResourceType.PATIENT,
        data={"resourceType": "Patient", "id": "p1"},
        source_format="TEST",
    )
    observation = ClinicalRecord(
        id="record-observation",
        resource_type=ResourceType.OBSERVATION,
        data={
            "resourceType": "Observation",
            "subject": {"reference": "Patient/p1"},
            "valueString": "normal",
        },
        source_format="TEST",
    )

    report = validate_dataset(ClinicalDataset(records=[patient, observation]))

    assert report.passed
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.records_by_type == {"Observation": 1, "Patient": 1}


def test_validator_flags_duplicate_ids_mismatches_and_broken_references() -> None:
    dataset = ClinicalDataset(
        records=[
            ClinicalRecord(
                id="duplicate",
                resource_type=ResourceType.PATIENT,
                data={"resourceType": "Patient", "id": "p1"},
                source_format="TEST",
            ),
            ClinicalRecord(
                id="duplicate",
                resource_type=ResourceType.PATIENT,
                data={"resourceType": "Patient", "id": "p2"},
                source_format="TEST",
            ),
            ClinicalRecord(
                id="bad-observation",
                resource_type=ResourceType.OBSERVATION,
                data={
                    "resourceType": "Condition",
                    "subject": {"reference": "Patient/missing"},
                },
                source_format="TEST",
                checksum="0" * 64,
            ),
        ]
    )

    report = DatasetValidator().validate(dataset)

    assert not report.passed
    assert report.error_count == 4
    counts = report.finding_counts_by_code()
    assert counts["record.duplicate_id"] == 1
    assert counts["record.resource_type_mismatch"] == 1
    assert counts["record.checksum_mismatch"] == 1
    assert counts["reference.unknown_patient"] == 1


def test_validator_warns_for_empty_payload_and_missing_subject() -> None:
    dataset = ClinicalDataset(
        records=[
            ClinicalRecord(
                id="p1",
                resource_type=ResourceType.PATIENT,
                data={},
                source_format="TEST",
            ),
            ClinicalRecord(
                id="obs1",
                resource_type=ResourceType.OBSERVATION,
                data={"resourceType": "Observation", "valueString": "normal"},
                source_format="TEST",
            ),
        ]
    )

    report = DatasetValidator().validate(dataset)

    assert report.passed
    assert report.error_count == 0
    assert report.warning_count == 2
    assert report.finding_counts_by_code() == {
        "record.empty_data": 1,
        "reference.missing_patient": 1,
    }


def test_validation_report_is_phi_safe() -> None:
    record = ClinicalRecord(
        id="record-1",
        resource_type=ResourceType.PATIENT,
        data={"resourceType": "Condition", "name": "Alice Smith"},
        source_format="TEST",
    )

    report = DatasetValidator().validate(ClinicalDataset(records=[record]))
    payload = report.to_safe_dict()
    markdown = report.to_markdown()

    assert payload["error_count"] == 1
    assert "Alice Smith" not in str(payload)
    assert "Alice Smith" not in markdown
