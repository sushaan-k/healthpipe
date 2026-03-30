"""CSV-to-FHIR mapping adapter.

Reads CSV files and maps columns to FHIR R4 Patient/Observation resources.
Supports explicit column mappings or an ``"auto"`` mode that infers
mappings from common clinical CSV header conventions.
"""

from __future__ import annotations

import csv
import logging
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from healthpipe.exceptions import IngestError
from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    ResourceType,
)

logger = logging.getLogger(__name__)

# Common column-name patterns mapped to FHIR fields.
# Keys are lowercased, stripped header names.
_AUTO_PATIENT_MAP: dict[str, str] = {
    "patient_id": "id",
    "patientid": "id",
    "mrn": "id",
    "medical_record_number": "id",
    "first_name": "name.given",
    "firstname": "name.given",
    "given_name": "name.given",
    "last_name": "name.family",
    "lastname": "name.family",
    "family_name": "name.family",
    "dob": "birthDate",
    "date_of_birth": "birthDate",
    "birth_date": "birthDate",
    "birthdate": "birthDate",
    "gender": "gender",
    "sex": "gender",
    "address": "address.line",
    "street": "address.line",
    "city": "address.city",
    "state": "address.state",
    "zip": "address.postalCode",
    "zip_code": "address.postalCode",
    "postal_code": "address.postalCode",
    "phone": "telecom.phone",
    "email": "telecom.email",
    "ssn": "identifier.SSN",
}

_AUTO_OBSERVATION_KEYS: set[str] = {
    "observation",
    "lab_value",
    "lab_result",
    "result",
    "value",
    "test_result",
    "glucose",
    "hemoglobin",
    "blood_pressure",
    "heart_rate",
    "bmi",
    "temperature",
    "weight",
    "height",
}

_AUTO_CONDITION_KEYS: set[str] = {
    "diagnosis",
    "condition",
    "icd_code",
    "icd10",
    "diagnosis_code",
    "dx",
    "problem",
}


class CSVSource(BaseModel):
    """Ingest adapter for CSV files.

    Args:
        path: Path to the CSV file.
        mapping: Either ``"auto"`` for automatic column inference or a dict
            mapping CSV column names to FHIR field paths.
        delimiter: CSV delimiter character.
        encoding: File encoding.
    """

    path: str
    mapping: str | dict[str, str] = "auto"
    delimiter: str = ","
    encoding: str = "utf-8"

    async def ingest(self) -> ClinicalDataset:
        """Read the CSV and return a ClinicalDataset."""
        filepath = Path(self.path)
        if not filepath.exists():
            raise IngestError(f"CSV file not found: {self.path}")

        try:
            text = filepath.read_text(encoding=self.encoding)
        except OSError as exc:
            raise IngestError(f"Cannot read CSV file {self.path}") from exc

        rows = list(csv.DictReader(StringIO(text), delimiter=self.delimiter))
        if not rows:
            logger.warning("CSV file %s is empty", self.path)
            return ClinicalDataset()

        column_map = self._resolve_mapping(list(rows[0].keys()))
        dataset = ClinicalDataset()

        for row in rows:
            records = self._row_to_records(row, column_map, source_uri=self.path)
            for rec in records:
                dataset.add_record(rec)

        logger.info("Ingested %d records from CSV %s", len(dataset.records), self.path)
        return dataset

    # -- Private helpers -------------------------------------------------------

    def _resolve_mapping(self, headers: list[str]) -> dict[str, str]:
        """Return a header-to-FHIR-path mapping."""
        if isinstance(self.mapping, dict):
            return self.mapping

        # Auto-mapping
        resolved: dict[str, str] = {}
        for header in headers:
            normalised = header.strip().lower().replace(" ", "_")
            if normalised in _AUTO_PATIENT_MAP:
                resolved[header] = _AUTO_PATIENT_MAP[normalised]
            elif normalised in _AUTO_OBSERVATION_KEYS:
                resolved[header] = f"observation.{normalised}"
            elif normalised in _AUTO_CONDITION_KEYS:
                resolved[header] = f"condition.{normalised}"

        if not resolved:
            logger.warning(
                "Auto-mapping found no recognised columns in %s. "
                "Provide an explicit mapping dict for best results.",
                self.path,
            )
        return resolved

    def _row_to_records(
        self,
        row: dict[str, str],
        column_map: dict[str, str],
        source_uri: str,
    ) -> list[ClinicalRecord]:
        """Convert a single CSV row into ClinicalRecords."""
        records: list[ClinicalRecord] = []
        patient_data: dict[str, Any] = {"resourceType": "Patient"}
        observations: list[dict[str, Any]] = []
        conditions: list[dict[str, Any]] = []

        for col, fhir_path in column_map.items():
            value = row.get(col, "").strip()
            if not value:
                continue

            if fhir_path.startswith("observation."):
                obs_name = fhir_path.split(".", 1)[1]
                observations.append(
                    {
                        "resourceType": "Observation",
                        "code": {"display": obs_name},
                        "valueString": value,
                    }
                )
            elif fhir_path.startswith("condition."):
                conditions.append(
                    {
                        "resourceType": "Condition",
                        "code": {"display": value},
                    }
                )
            else:
                _set_nested(patient_data, fhir_path, value)

        # Always emit a patient record if we found any patient fields
        if len(patient_data) > 1:
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data=patient_data,
                    source_format="CSV",
                    source_uri=source_uri,
                )
            )

        patient_id = patient_data.get("id", "")
        for obs in observations:
            obs["subject"] = {"reference": f"Patient/{patient_id}"}
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.OBSERVATION,
                    data=obs,
                    source_format="CSV",
                    source_uri=source_uri,
                )
            )

        for cond in conditions:
            cond["subject"] = {"reference": f"Patient/{patient_id}"}
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.CONDITION,
                    data=cond,
                    source_format="CSV",
                    source_uri=source_uri,
                )
            )

        return records


def _set_nested(data: dict[str, Any], path: str, value: str) -> None:
    """Set a value in a dict using a dot-separated path.

    Handles simple FHIR-like paths such as ``name.given`` or
    ``address.line``.
    """
    parts = path.split(".")
    if len(parts) == 1:
        data[parts[0]] = value
        return

    # Two-level paths
    parent, child = parts[0], parts[1]
    if parent == "name":
        names = data.setdefault("name", [{}])
        if child == "given":
            names[0].setdefault("given", []).append(value)
        else:
            names[0][child] = value
    elif parent == "address":
        addrs = data.setdefault("address", [{}])
        if child == "line":
            addrs[0].setdefault("line", []).append(value)
        else:
            addrs[0][child] = value
    elif parent == "telecom":
        telecoms = data.setdefault("telecom", [])
        telecoms.append({"system": child, "value": value})
    elif parent == "identifier":
        identifiers = data.setdefault("identifier", [])
        identifiers.append({"system": child, "value": value})
    else:
        sub = data.setdefault(parent, {})
        sub[child] = value
