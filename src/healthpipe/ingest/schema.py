"""Unified internal schema for healthpipe.

Every ingest source converts its data into ``ClinicalRecord`` objects which
are collected inside a ``ClinicalDataset``.  The canonical representation
mirrors FHIR R4 resource types but uses lightweight Pydantic models so that
the heavy ``fhir.resources`` library is optional at runtime.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResourceType(StrEnum):
    """Supported FHIR R4 resource types."""

    PATIENT = "Patient"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION_REQUEST = "MedicationRequest"
    ENCOUNTER = "Encounter"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    PROCEDURE = "Procedure"
    ALLERGY_INTOLERANCE = "AllergyIntolerance"
    IMMUNIZATION = "Immunization"
    DOCUMENT_REFERENCE = "DocumentReference"


class Coding(BaseModel):
    """A FHIR-style Coding (system + code + display)."""

    system: str = ""
    code: str = ""
    display: str = ""


class HumanName(BaseModel):
    """Patient / practitioner name."""

    family: str = ""
    given: list[str] = Field(default_factory=list)
    prefix: list[str] = Field(default_factory=list)
    suffix: list[str] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Return a single-string representation."""
        parts = [*self.prefix, *self.given, self.family, *self.suffix]
        return " ".join(p for p in parts if p)


class Address(BaseModel):
    """Postal address."""

    line: list[str] = Field(default_factory=list)
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "US"


class ContactPoint(BaseModel):
    """Phone / email / fax."""

    system: str = ""  # phone | email | fax
    value: str = ""
    use: str = ""  # home | work | mobile


class PatientResource(BaseModel):
    """Lightweight FHIR R4 Patient resource."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: str = "Patient"
    name: list[HumanName] = Field(default_factory=list)
    birth_date: date | None = None
    gender: str = ""
    address: list[Address] = Field(default_factory=list)
    telecom: list[ContactPoint] = Field(default_factory=list)
    identifier: list[dict[str, str]] = Field(default_factory=list)
    deceased: bool = False
    marital_status: Coding | None = None


class ObservationResource(BaseModel):
    """Lightweight FHIR R4 Observation resource."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: str = "Observation"
    status: str = "final"
    code: Coding = Field(default_factory=Coding)
    subject_id: str = ""
    effective_datetime: datetime | None = None
    value_quantity: float | None = None
    value_unit: str = ""
    value_string: str = ""
    interpretation: str = ""


class ConditionResource(BaseModel):
    """Lightweight FHIR R4 Condition resource."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: str = "Condition"
    code: Coding = Field(default_factory=Coding)
    subject_id: str = ""
    onset_datetime: datetime | None = None
    abatement_datetime: datetime | None = None
    clinical_status: str = "active"
    verification_status: str = "confirmed"


class ClinicalRecord(BaseModel):
    """A single clinical resource in the unified schema.

    Wraps any FHIR-like resource with provenance metadata so we can
    track where data came from and what transformations were applied.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType
    data: dict[str, Any] = Field(default_factory=dict)
    source_format: str = ""
    source_uri: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checksum: str = ""

    def model_post_init(self, _context: Any) -> None:
        """Compute checksum after initialisation if not already set."""
        if not self.checksum:
            raw = self.model_dump_json(exclude={"checksum", "ingested_at", "id"})
            self.checksum = hashlib.sha256(raw.encode()).hexdigest()


class ResourceCollection:
    """A typed view over records of a single resource type."""

    def __init__(self, records: list[ClinicalRecord]) -> None:
        self._records = records

    def count(self) -> int:
        """Return the number of records in the collection."""
        return len(self._records)

    def __iter__(self) -> Iterator[ClinicalRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialise every record to a plain dict."""
        return [r.model_dump() for r in self._records]


class ClinicalDataset(BaseModel):
    """Container for all ingested clinical records.

    Provides typed accessors for common FHIR resource types and a
    flat ``records`` list for iteration.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    records: list[ClinicalRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # --- Convenience accessors ------------------------------------------------

    @property
    def patients(self) -> ResourceCollection:
        """All Patient resources in the dataset."""
        return ResourceCollection(
            [r for r in self.records if r.resource_type == ResourceType.PATIENT]
        )

    @property
    def observations(self) -> ResourceCollection:
        """All Observation resources in the dataset."""
        return ResourceCollection(
            [r for r in self.records if r.resource_type == ResourceType.OBSERVATION]
        )

    @property
    def conditions(self) -> ResourceCollection:
        """All Condition resources in the dataset."""
        return ResourceCollection(
            [r for r in self.records if r.resource_type == ResourceType.CONDITION]
        )

    @property
    def encounters(self) -> ResourceCollection:
        """All Encounter resources in the dataset."""
        return ResourceCollection(
            [r for r in self.records if r.resource_type == ResourceType.ENCOUNTER]
        )

    def add_record(self, record: ClinicalRecord) -> None:
        """Append a record to the dataset."""
        self.records.append(record)

    def merge(self, other: ClinicalDataset) -> ClinicalDataset:
        """Return a new dataset containing records from both datasets."""
        return ClinicalDataset(
            records=[*self.records, *other.records],
            metadata={**self.metadata, **other.metadata},
        )

    def filter_by_type(self, resource_type: ResourceType) -> list[ClinicalRecord]:
        """Return records matching the given resource type."""
        return [r for r in self.records if r.resource_type == resource_type]
