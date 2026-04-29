"""Dataset integrity validation for clinical pipeline inputs.

The validator checks structural issues that can silently break downstream
de-identification, privacy scoring, and synthetic generation: duplicate record
identifiers, stale checksums, resource type mismatches, and broken patient
references.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType


class ValidationSeverity(StrEnum):
    """Severity levels for dataset validation findings."""

    ERROR = "error"
    WARNING = "warning"


class DatasetValidationFinding(BaseModel):
    """A single PHI-safe dataset validation finding."""

    severity: ValidationSeverity
    code: str
    message: str
    record_id: str = ""
    resource_type: str = ""
    field_path: str = ""


class DatasetValidationReport(BaseModel):
    """Validation report for a ``ClinicalDataset``."""

    total_records: int = 0
    records_by_type: dict[str, int] = Field(default_factory=dict)
    findings: list[DatasetValidationFinding] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Number of error-severity findings."""
        return sum(1 for finding in self.findings if finding.severity == "error")

    @property
    def warning_count(self) -> int:
        """Number of warning-severity findings."""
        return sum(1 for finding in self.findings if finding.severity == "warning")

    @property
    def passed(self) -> bool:
        """Whether the dataset has no error-severity findings."""
        return self.error_count == 0

    def finding_counts_by_code(self) -> dict[str, int]:
        """Return finding counts grouped by validation code."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.code] = counts.get(finding.code, 0) + 1
        return dict(sorted(counts.items()))

    def to_safe_dict(self) -> dict[str, Any]:
        """Serialize report data without including clinical payload values."""
        return {
            "passed": self.passed,
            "total_records": self.total_records,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "records_by_type": self.records_by_type,
            "finding_counts_by_code": self.finding_counts_by_code(),
            "findings": [finding.model_dump(mode="json") for finding in self.findings],
        }

    def to_markdown(self, *, max_findings: int = 50) -> str:
        """Render a compact Markdown validation report."""
        if max_findings < 0:
            raise ValueError("max_findings must be non-negative")

        lines = [
            "# healthpipe Dataset Validation Report",
            "",
            f"- Status: {'passed' if self.passed else 'failed'}",
            f"- Records checked: {self.total_records}",
            f"- Errors: {self.error_count}",
            f"- Warnings: {self.warning_count}",
            "",
        ]

        if self.records_by_type:
            lines.extend(["## Records by Type", ""])
            for resource_type, count in sorted(self.records_by_type.items()):
                lines.append(f"- {resource_type}: {count}")
            lines.append("")

        if self.findings:
            shown = self.findings[:max_findings]
            lines.extend(
                [
                    "## Findings",
                    "",
                    "| Severity | Code | Record | Type | Field | Message |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for finding in shown:
                lines.append(
                    "| "
                    f"{_markdown_cell(finding.severity)} | "
                    f"{_markdown_cell(finding.code)} | "
                    f"{_markdown_cell(finding.record_id or '(dataset)')} | "
                    f"{_markdown_cell(finding.resource_type or '(n/a)')} | "
                    f"{_markdown_cell(finding.field_path or '(n/a)')} | "
                    f"{_markdown_cell(finding.message)} |"
                )

            remaining = len(self.findings) - len(shown)
            if remaining > 0:
                lines.extend(["", f"_Omitted {remaining} additional finding(s)._"])

        return "\n".join(lines)


class DatasetValidator:
    """Validate ``ClinicalDataset`` objects before downstream processing."""

    SUBJECT_RESOURCE_TYPES: ClassVar[set[ResourceType]] = {
        ResourceType.OBSERVATION,
        ResourceType.CONDITION,
        ResourceType.MEDICATION_REQUEST,
        ResourceType.ENCOUNTER,
        ResourceType.DIAGNOSTIC_REPORT,
        ResourceType.PROCEDURE,
        ResourceType.ALLERGY_INTOLERANCE,
        ResourceType.IMMUNIZATION,
    }

    def __init__(
        self,
        *,
        check_checksums: bool = True,
        check_references: bool = True,
    ) -> None:
        self.check_checksums = check_checksums
        self.check_references = check_references

    def validate(self, dataset: ClinicalDataset) -> DatasetValidationReport:
        """Validate a dataset and return a PHI-safe report."""
        findings: list[DatasetValidationFinding] = []
        records_by_type = _count_records_by_type(dataset.records)

        self._check_record_ids(dataset.records, findings)
        self._check_payload_ids(dataset.records, findings)

        patient_ids = _patient_identifiers(dataset.records)
        for record in dataset.records:
            self._check_record_payload(record, findings)
            if self.check_checksums:
                self._check_checksum(record, findings)
            if self.check_references:
                self._check_patient_reference(record, patient_ids, findings)

        return DatasetValidationReport(
            total_records=len(dataset.records),
            records_by_type=records_by_type,
            findings=findings,
        )

    def _check_record_ids(
        self,
        records: list[ClinicalRecord],
        findings: list[DatasetValidationFinding],
    ) -> None:
        seen: dict[str, ClinicalRecord] = {}
        for record in records:
            if not record.id:
                _add_finding(
                    findings,
                    ValidationSeverity.ERROR,
                    "record.missing_id",
                    "Record id is empty.",
                    record=record,
                    field_path="id",
                )
                continue

            if record.id in seen:
                _add_finding(
                    findings,
                    ValidationSeverity.ERROR,
                    "record.duplicate_id",
                    "Record id is duplicated.",
                    record=record,
                    field_path="id",
                )
            else:
                seen[record.id] = record

    def _check_payload_ids(
        self,
        records: list[ClinicalRecord],
        findings: list[DatasetValidationFinding],
    ) -> None:
        seen: dict[tuple[str, str], ClinicalRecord] = {}
        for record in records:
            payload_id = (
                record.data.get("id") if isinstance(record.data, dict) else None
            )
            if not isinstance(payload_id, str) or not payload_id.strip():
                continue

            key = (record.resource_type.value, payload_id.strip())
            if key in seen:
                _add_finding(
                    findings,
                    ValidationSeverity.ERROR,
                    "record.duplicate_payload_id",
                    "FHIR payload id is duplicated within this resource type.",
                    record=record,
                    field_path="data.id",
                )
            else:
                seen[key] = record

    def _check_record_payload(
        self,
        record: ClinicalRecord,
        findings: list[DatasetValidationFinding],
    ) -> None:
        if not record.data:
            _add_finding(
                findings,
                ValidationSeverity.WARNING,
                "record.empty_data",
                "Record payload is empty.",
                record=record,
                field_path="data",
            )
            return

        payload_type = record.data.get("resourceType")
        if payload_type is None:
            _add_finding(
                findings,
                ValidationSeverity.WARNING,
                "record.missing_resource_type",
                "Payload is missing data.resourceType.",
                record=record,
                field_path="data.resourceType",
            )
        elif payload_type != record.resource_type.value:
            _add_finding(
                findings,
                ValidationSeverity.ERROR,
                "record.resource_type_mismatch",
                "ClinicalRecord.resource_type does not match data.resourceType.",
                record=record,
                field_path="data.resourceType",
            )

    def _check_checksum(
        self,
        record: ClinicalRecord,
        findings: list[DatasetValidationFinding],
    ) -> None:
        if not record.checksum:
            _add_finding(
                findings,
                ValidationSeverity.WARNING,
                "record.missing_checksum",
                "Record checksum is empty.",
                record=record,
                field_path="checksum",
            )
            return

        if not _looks_like_sha256(record.checksum):
            _add_finding(
                findings,
                ValidationSeverity.WARNING,
                "record.invalid_checksum",
                "Record checksum is not a SHA-256 hex digest.",
                record=record,
                field_path="checksum",
            )
            return

        expected = _compute_record_checksum(record)
        if record.checksum != expected:
            _add_finding(
                findings,
                ValidationSeverity.ERROR,
                "record.checksum_mismatch",
                "Record checksum does not match the current payload metadata.",
                record=record,
                field_path="checksum",
            )

    def _check_patient_reference(
        self,
        record: ClinicalRecord,
        patient_ids: set[str],
        findings: list[DatasetValidationFinding],
    ) -> None:
        if record.resource_type not in self.SUBJECT_RESOURCE_TYPES:
            return

        subject = _extract_patient_reference(record.data)
        if subject is None:
            _add_finding(
                findings,
                ValidationSeverity.WARNING,
                "reference.missing_patient",
                "Record has no patient subject reference.",
                record=record,
                field_path="data.subject",
            )
            return

        if subject not in patient_ids:
            _add_finding(
                findings,
                ValidationSeverity.ERROR,
                "reference.unknown_patient",
                "Record references a patient that is not present in the dataset.",
                record=record,
                field_path="data.subject",
            )


def validate_dataset(dataset: ClinicalDataset) -> DatasetValidationReport:
    """Validate a dataset with the default validator."""
    return DatasetValidator().validate(dataset)


def _count_records_by_type(records: list[ClinicalRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = record.resource_type.value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _patient_identifiers(records: list[ClinicalRecord]) -> set[str]:
    identifiers: set[str] = set()
    for record in records:
        if record.resource_type != ResourceType.PATIENT:
            continue
        identifiers.add(record.id)
        payload_id = record.data.get("id")
        if isinstance(payload_id, str) and payload_id.strip():
            identifiers.add(payload_id.strip())
    return identifiers


def _extract_patient_reference(data: dict[str, Any]) -> str | None:
    for key in ("subject_id", "subjectId", "patient_id", "patientId"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_patient_reference(value)

    subject = data.get("subject")
    if isinstance(subject, str) and subject.strip():
        return _normalize_patient_reference(subject)
    if isinstance(subject, dict):
        reference = subject.get("reference") or subject.get("id")
        if isinstance(reference, str) and reference.strip():
            return _normalize_patient_reference(reference)
    return None


def _normalize_patient_reference(value: str) -> str:
    value = value.strip()
    if value.startswith("Patient/"):
        return value.split("/", 1)[1]
    return value


def _compute_record_checksum(record: ClinicalRecord) -> str:
    raw = record.model_dump_json(exclude={"checksum", "ingested_at", "id"})
    return hashlib.sha256(raw.encode()).hexdigest()


def _looks_like_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _add_finding(
    findings: list[DatasetValidationFinding],
    severity: ValidationSeverity,
    code: str,
    message: str,
    *,
    record: ClinicalRecord,
    field_path: str,
) -> None:
    findings.append(
        DatasetValidationFinding(
            severity=severity,
            code=code,
            message=message,
            record_id=record.id,
            resource_type=record.resource_type.value,
            field_path=field_path,
        )
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
