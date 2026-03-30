"""Shared test fixtures for healthpipe test suite."""

from __future__ import annotations

import pytest

from healthpipe.audit.logger import AuditEntry, AuditLog
from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    ResourceType,
)


@pytest.fixture()
def sample_patient_data() -> dict:
    """A realistic FHIR-style patient resource dict."""
    return {
        "resourceType": "Patient",
        "id": "patient-001",
        "name": [{"family": "Smith", "given": ["John"]}],
        "birthDate": "1985-03-15",
        "gender": "male",
        "address": [
            {
                "line": ["123 Main Street"],
                "city": "Springfield",
                "state": "IL",
                "postalCode": "62704",
            }
        ],
        "telecom": [
            {"system": "phone", "value": "555-123-4567"},
            {"system": "email", "value": "john.smith@example.com"},
        ],
        "identifier": [{"system": "SSN", "value": "123-45-6789"}],
    }


@pytest.fixture()
def sample_observation_data() -> dict:
    """A realistic FHIR-style observation resource dict."""
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "system": "LOINC",
            "code": "2345-7",
            "display": "Glucose",
        },
        "subject": {"reference": "Patient/patient-001"},
        "effectiveDateTime": "2025-03-15T10:30:00",
        "valueQuantity": {"value": 110, "unit": "mg/dL"},
    }


@pytest.fixture()
def sample_clinical_record(sample_patient_data: dict) -> ClinicalRecord:
    """A ClinicalRecord wrapping a patient resource."""
    return ClinicalRecord(
        resource_type=ResourceType.PATIENT,
        data=sample_patient_data,
        source_format="FHIR_R4",
        source_uri="test://fixture",
    )


@pytest.fixture()
def sample_dataset(
    sample_patient_data: dict, sample_observation_data: dict
) -> ClinicalDataset:
    """A ClinicalDataset with one patient and one observation."""
    return ClinicalDataset(
        records=[
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                data=sample_patient_data,
                source_format="FHIR_R4",
                source_uri="test://fixture",
            ),
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data=sample_observation_data,
                source_format="FHIR_R4",
                source_uri="test://fixture",
            ),
        ]
    )


@pytest.fixture()
def sample_deidentified(sample_dataset: ClinicalDataset) -> DeidentifiedDataset:
    """A DeidentifiedDataset wrapping the sample dataset."""
    audit = AuditLog()
    audit.add(
        AuditEntry(
            record_id="test-record",
            action="PHI_REMOVED",
            layer="NER",
            original="John Smith",
            replacement="[PATIENT_NAME]",
            category="PATIENT_NAME",
            confidence=0.95,
        )
    )
    return DeidentifiedDataset(
        dataset=sample_dataset,
        audit_log=audit,
        method="safe_harbor",
    )
