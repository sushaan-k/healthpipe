"""HL7v2 message parser and ingest adapter.

Parses HL7v2 pipe-delimited messages (ADT, ORU, ORM, etc.) and maps
segments to the healthpipe unified schema.  No external HL7 library is
required -- we implement a lightweight parser that handles the most
common segment types encountered in real hospital interfaces.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from healthpipe.exceptions import HL7ParseError
from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    ResourceType,
)

logger = logging.getLogger(__name__)

# HL7v2 standard separators
_FIELD_SEP = "|"
_COMPONENT_SEP = "^"
_REPEAT_SEP = "~"
_ESCAPE_CHAR = "\\"
_SUB_COMPONENT_SEP = "&"


def _parse_hl7_datetime(raw: str) -> datetime | None:
    """Parse an HL7v2 timestamp (yyyyMMddHHmmss) into a datetime."""
    raw = raw.strip()
    if not raw:
        return None
    formats = [
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw[: len(fmt.replace("%", ""))], fmt)
        except ValueError:
            continue
    return None


class HL7Message:
    """Lightweight representation of a parsed HL7v2 message."""

    def __init__(self, raw: str) -> None:
        self.raw = raw.strip()
        self.segments: dict[str, list[list[str]]] = {}
        self._parse()

    def _parse(self) -> None:
        """Split the raw message into segments and fields."""
        lines = re.split(r"\r\n|\r|\n", self.raw)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            fields = line.split(_FIELD_SEP)
            seg_id = fields[0]
            self.segments.setdefault(seg_id, []).append(fields)

    def get_field(self, segment: str, index: int, occurrence: int = 0) -> str:
        """Return a field value by segment name and 1-based index.

        Args:
            segment: Three-letter segment identifier (e.g. ``"PID"``).
            index: 1-based field index within the segment.
            occurrence: Which occurrence of the segment to read (default first).

        Returns:
            The raw string value, or ``""`` if not present.
        """
        segs = self.segments.get(segment, [])
        if occurrence >= len(segs):
            return ""
        fields = segs[occurrence]
        # MSH is special -- MSH-1 is the field separator itself
        if segment == "MSH":
            index -= 1
        if index < 0 or index >= len(fields):
            return ""
        return fields[index]

    def get_components(self, segment: str, index: int) -> list[str]:
        """Split a field value on the component separator."""
        return self.get_field(segment, index).split(_COMPONENT_SEP)

    @property
    def message_type(self) -> str:
        """E.g. ``'ADT^A01'``."""
        return self.get_field("MSH", 9)


class HL7v2Source(BaseModel):
    """Ingest adapter for HL7v2 pipe-delimited messages.

    Args:
        path: Glob pattern or path to a single ``.hl7`` file.
    """

    path: str
    encoding: str = "utf-8"
    _files: list[Path] = []

    def model_post_init(self, _ctx: Any) -> None:
        """Resolve the glob pattern to concrete file paths."""
        resolved = glob(self.path, recursive=True)
        self._files = [Path(p) for p in resolved if Path(p).is_file()]

    async def ingest(self) -> ClinicalDataset:
        """Parse all matched HL7v2 files and return a ClinicalDataset."""
        dataset = ClinicalDataset()
        for fpath in self._files:
            try:
                raw = fpath.read_text(encoding=self.encoding)
            except OSError as exc:
                raise HL7ParseError(f"Cannot read HL7 file {fpath}: {exc}") from exc
            messages = self._split_messages(raw)
            for msg_raw in messages:
                msg = HL7Message(msg_raw)
                records = self._message_to_records(msg, source_uri=str(fpath))
                for rec in records:
                    dataset.add_record(rec)
        logger.info(
            "Ingested %d records from %d HL7v2 files",
            len(dataset.records),
            len(self._files),
        )
        return dataset

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _split_messages(raw: str) -> list[str]:
        """Split a file that may contain multiple MSH-delimited messages."""
        parts = re.split(r"(?=^MSH\|)", raw, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

    def _message_to_records(
        self, msg: HL7Message, source_uri: str
    ) -> list[ClinicalRecord]:
        """Convert an HL7 message into one or more ClinicalRecords."""
        records: list[ClinicalRecord] = []

        # Extract patient from PID segment
        if "PID" in msg.segments:
            patient_data = self._extract_patient(msg)
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data=patient_data,
                    source_format="HL7v2",
                    source_uri=source_uri,
                )
            )

        # Extract observations from OBX segments
        for idx, _seg in enumerate(msg.segments.get("OBX", [])):
            obs_data = self._extract_observation(msg, idx)
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.OBSERVATION,
                    data=obs_data,
                    source_format="HL7v2",
                    source_uri=source_uri,
                )
            )

        # Extract diagnosis from DG1
        for idx, _seg in enumerate(msg.segments.get("DG1", [])):
            dx_data = self._extract_diagnosis(msg, idx)
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.CONDITION,
                    data=dx_data,
                    source_format="HL7v2",
                    source_uri=source_uri,
                )
            )

        return records

    @staticmethod
    def _extract_patient(msg: HL7Message) -> dict[str, Any]:
        """Map PID fields to a patient data dict."""
        name_parts = msg.get_components("PID", 5)
        patient_id = msg.get_field("PID", 3).split(_COMPONENT_SEP)[0]
        dob_raw = msg.get_field("PID", 7)
        gender_raw = msg.get_field("PID", 8)
        address_parts = msg.get_components("PID", 11)
        phone = msg.get_field("PID", 13)
        ssn = msg.get_field("PID", 19)
        birth_dt = _parse_hl7_datetime(dob_raw)

        return {
            "resourceType": "Patient",
            "id": patient_id or None,
            "name": [
                {
                    "family": name_parts[0] if len(name_parts) > 0 else "",
                    "given": [name_parts[1]] if len(name_parts) > 1 else [],
                }
            ],
            "birthDate": str(birth_dt.date()) if birth_dt is not None else None,
            "gender": {"M": "male", "F": "female", "O": "other"}.get(
                gender_raw, "unknown"
            ),
            "address": [
                {
                    "line": [address_parts[0]] if address_parts else [],
                    "city": address_parts[2] if len(address_parts) > 2 else "",
                    "state": address_parts[3] if len(address_parts) > 3 else "",
                    "postalCode": address_parts[4] if len(address_parts) > 4 else "",
                }
            ],
            "telecom": [{"system": "phone", "value": phone}] if phone else [],
            "identifier": [
                {"system": "SSN", "value": ssn},
            ]
            if ssn
            else [],
        }

    @staticmethod
    def _extract_observation(msg: HL7Message, idx: int) -> dict[str, Any]:
        """Map OBX fields to an observation data dict."""
        code_parts = msg.get_field("OBX", 3, occurrence=idx).split(_COMPONENT_SEP)
        value = msg.get_field("OBX", 5, occurrence=idx)
        units = msg.get_field("OBX", 6, occurrence=idx).split(_COMPONENT_SEP)[0]
        timestamp = msg.get_field("OBX", 14, occurrence=idx)

        return {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "system": code_parts[2] if len(code_parts) > 2 else "LOINC",
                "code": code_parts[0] if code_parts else "",
                "display": code_parts[1] if len(code_parts) > 1 else "",
            },
            "valueQuantity": {"value": value, "unit": units},
            "effectiveDateTime": str(_parse_hl7_datetime(timestamp))
            if _parse_hl7_datetime(timestamp)
            else None,
        }

    @staticmethod
    def _extract_diagnosis(msg: HL7Message, idx: int) -> dict[str, Any]:
        """Map DG1 fields to a condition data dict."""
        code_parts = msg.get_field("DG1", 3, occurrence=idx).split(_COMPONENT_SEP)
        description = msg.get_field("DG1", 4, occurrence=idx)
        diag_type = msg.get_field("DG1", 6, occurrence=idx)

        return {
            "resourceType": "Condition",
            "code": {
                "system": "ICD-10-CM",
                "code": code_parts[0] if code_parts else "",
                "display": description
                or (code_parts[1] if len(code_parts) > 1 else ""),
            },
            "clinicalStatus": "active",
            "verificationStatus": "confirmed",
            "category": diag_type,
        }
