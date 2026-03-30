"""Date shifting for HIPAA-compliant de-identification.

Applies a consistent random offset to every date associated with a given
patient.  This preserves temporal intervals (critical for clinical
reasoning -- e.g. days between lab tests) while making absolute dates
useless for re-identification.

Per Safe Harbor, ages over 89 are collapsed to 90.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from healthpipe.exceptions import DateShiftError

logger = logging.getLogger(__name__)

# Common date formats found in clinical data
_DATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("%Y-%m-%d", re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")),
    ("%m/%d/%Y", re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")),
    ("%m-%d-%Y", re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")),
    ("%Y%m%d", re.compile(r"\b(\d{8})\b")),
]

_MAX_AGE = 89


class DateShifter(BaseModel):
    """Patient-consistent date shifter.

    Each patient receives a deterministic random offset derived from a
    keyed hash of their identifier, so the same patient always gets the
    same shift across runs (given the same ``salt``).

    Args:
        shift_range: Tuple of (min_days, max_days) for the random offset.
        salt: Secret salt used to derive per-patient offsets.  Must be
            kept confidential to prevent reversing the shift.  A unique,
            secret salt is **required** -- no default is provided.
        collapse_age_over_89: If True, any resulting age > 89 is set to 90.

    Raises:
        ValueError: If no salt is provided or an empty salt is given.
    """

    shift_range: tuple[int, int] = (-365, 365)
    salt: str
    collapse_age_over_89: bool = True

    @classmethod
    def _validate_salt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "A unique, secret salt is required for HIPAA-compliant "
                "date shifting. Do not use a default or publicly known value."
            )
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_salt(self.salt)

    def get_offset_days(self, patient_id: str) -> int:
        """Compute a deterministic day offset for *patient_id*.

        The offset is consistent across calls for the same patient_id
        and salt combination.
        """
        digest = hashlib.sha256(f"{self.salt}:{patient_id}".encode()).hexdigest()
        # Map the first 8 hex chars to an integer in the shift range
        hash_int = int(digest[:8], 16)
        lo, hi = self.shift_range
        span = hi - lo
        if span <= 0:
            raise DateShiftError(
                f"Invalid shift_range: {self.shift_range}. "
                "The upper bound must be greater than the lower bound."
            )
        return lo + (hash_int % (span + 1))

    def shift_date(self, original: date, patient_id: str) -> date:
        """Shift a single date by the patient's offset.

        Args:
            original: The original date.
            patient_id: Identifier used to derive the offset.

        Returns:
            The shifted date.
        """
        offset = self.get_offset_days(patient_id)
        return original + timedelta(days=offset)

    def shift_datetime(self, original: datetime, patient_id: str) -> datetime:
        """Shift a datetime, preserving the time component."""
        offset = self.get_offset_days(patient_id)
        return original + timedelta(days=offset)

    def shift_text(self, text: str, patient_id: str) -> tuple[str, int]:
        """Find and shift all dates embedded in free text.

        Returns:
            Tuple of (shifted_text, count_of_dates_shifted).
        """
        count = 0
        result = text

        for fmt, pattern in _DATE_PATTERNS:

            def _replace(m: re.Match[str], _fmt: str = fmt) -> str:
                nonlocal count
                try:
                    dt = datetime.strptime(m.group(1), _fmt)
                    shifted = self.shift_datetime(dt, patient_id)
                    count += 1
                    return shifted.strftime(_fmt)
                except ValueError:
                    return m.group(0)

            result = pattern.sub(_replace, result)

        return result, count

    def shift_record_data(
        self, data: dict[str, Any], patient_id: str
    ) -> dict[str, Any]:
        """Recursively shift all date-like values in a record dict.

        Handles ISO date strings and common clinical date formats.
        Returns a new dict with shifted dates.
        """
        shifted, _count = self.shift_record_data_with_count(data, patient_id)
        return shifted

    def shift_record_data_with_count(
        self, data: dict[str, Any], patient_id: str
    ) -> tuple[dict[str, Any], int]:
        """Recursively shift dates and report how many values changed."""
        shifted, count = self._walk_with_count(data, patient_id)
        if not isinstance(shifted, dict):
            raise TypeError("Expected record data to remain a dict after shifting")
        return shifted, count

    def maybe_collapse_age(
        self, birth_date: date, reference_date: date | None = None
    ) -> date:
        """If age > 89, return a birth date that makes age = 90.

        Per HIPAA Safe Harbor, ages over 89 must be aggregated into a
        single category.
        """
        if not self.collapse_age_over_89:
            return birth_date
        ref = reference_date or date.today()
        age = (ref - birth_date).days // 365
        if age > _MAX_AGE:
            return ref - timedelta(days=90 * 365)
        return birth_date

    # -- Private helpers -------------------------------------------------------

    def _walk(self, obj: Any, patient_id: str) -> Any:
        """Recursively process a nested structure."""
        shifted, _count = self._walk_with_count(obj, patient_id)
        return shifted

    def _walk_with_count(self, obj: Any, patient_id: str) -> tuple[Any, int]:
        """Recursively process a nested structure and count shifted values."""
        if isinstance(obj, str):
            return self._try_shift_string(obj, patient_id)
        if isinstance(obj, dict):
            total = 0
            result: dict[str, Any] = {}
            for k, v in obj.items():
                shifted, count = self._walk_with_count(v, patient_id)
                result[k] = shifted
                total += count
            return result, total
        if isinstance(obj, list):
            total = 0
            result_list: list[Any] = []
            for item in obj:
                shifted, count = self._walk_with_count(item, patient_id)
                result_list.append(shifted)
                total += count
            return result_list, total
        if isinstance(obj, datetime):
            return self.shift_datetime(obj, patient_id), 1
        if isinstance(obj, date):
            return self.shift_date(obj, patient_id), 1
        return obj, 0

    def _try_shift_string(self, value: str, patient_id: str) -> tuple[str, int]:
        """Try to parse and shift a string if it looks like a date."""
        shifted_text, count = self.shift_text(value, patient_id)
        if count > 0:
            return shifted_text, count

        for fmt, _ in _DATE_PATTERNS:
            try:
                dt = datetime.strptime(value.strip(), fmt)
                shifted = self.shift_datetime(dt, patient_id)
                return shifted.strftime(fmt), 1
            except ValueError:
                continue
        return value, 0
