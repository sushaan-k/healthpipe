"""HIPAA Safe Harbor orchestration engine.

Coordinates all four de-identification layers:
1. Named Entity Recognition (NER)
2. Pattern matching (regex)
3. Date shifting
4. LLM verification (optional)

Produces a ``DeidentifiedDataset`` that wraps the cleaned data together
with a full audit log of every PHI removal.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from healthpipe.audit.logger import AuditEntry, AuditLog
from healthpipe.deidentify.date_shift import DateShifter
from healthpipe.deidentify.llm_verify import LLMVerifier
from healthpipe.deidentify.ner import ClinicalNER
from healthpipe.deidentify.patterns import PatternMatcher
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord

logger = logging.getLogger(__name__)


class DeidentifiedDataset(BaseModel):
    """A dataset that has been through the de-identification pipeline.

    Carries the cleaned records alongside a full audit trail.
    """

    dataset: ClinicalDataset
    audit_log: AuditLog = Field(default_factory=AuditLog)
    method: str = "safe_harbor"
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def records(self) -> list[ClinicalRecord]:
        """Shortcut to the underlying dataset records."""
        return self.dataset.records


class SafeHarborConfig(BaseModel):
    """Configuration for the Safe Harbor de-identification engine.

    The ``date_shift_salt`` must be explicitly provided when date shifting
    is enabled.  There is no default salt -- using a public/hardcoded
    value would allow anyone to reverse the date shifts.
    """

    date_shift: bool = True
    date_shift_range: tuple[int, int] = (-365, 365)
    date_shift_salt: str = ""
    llm_verification: bool = False
    llm_model: str = "claude-haiku-4-5"
    llm_api_key: str = ""
    llm_provider: str = "anthropic"
    ner_model: str = "en_core_web_sm"
    use_fallback_ner: bool = False
    collapse_age_over_89: bool = True


class SafeHarborEngine:
    """Multi-layer de-identification engine implementing HIPAA Safe Harbor.

    The 18 HIPAA identifiers are addressed across three mandatory layers
    (NER, pattern matching, date shifting) and one optional layer (LLM
    verification).

    Usage::

        engine = SafeHarborEngine(config)
        result = await engine.run(dataset)
    """

    def __init__(self, config: SafeHarborConfig | None = None) -> None:
        self.config = config or SafeHarborConfig()
        self._ner = ClinicalNER(
            model_name=self.config.ner_model,
            use_fallback=self.config.use_fallback_ner,
        )
        self._patterns = PatternMatcher()

        if self.config.date_shift:
            if (
                not self.config.date_shift_salt
                or not self.config.date_shift_salt.strip()
            ):
                raise ValueError(
                    "A unique, secret salt is required for HIPAA-compliant "
                    "date shifting. Set 'date_shift_salt' in SafeHarborConfig "
                    "or disable date shifting with date_shift=False."
                )
            self._date_shifter = DateShifter(
                shift_range=self.config.date_shift_range,
                salt=self.config.date_shift_salt,
                collapse_age_over_89=self.config.collapse_age_over_89,
            )
        else:
            # Create a dummy shifter that won't be used (date_shift=False)
            self._date_shifter = DateShifter(
                shift_range=self.config.date_shift_range,
                salt="unused-date-shift-disabled",
                collapse_age_over_89=self.config.collapse_age_over_89,
            )
        self._llm_verifier = (
            LLMVerifier(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                provider=self.config.llm_provider,
            )
            if self.config.llm_verification
            else None
        )

    async def run(self, dataset: ClinicalDataset) -> DeidentifiedDataset:
        """Execute the full de-identification pipeline.

        Args:
            dataset: The raw clinical dataset to de-identify.

        Returns:
            A ``DeidentifiedDataset`` with cleaned records and audit log.
        """
        audit = AuditLog()
        cleaned_records: list[ClinicalRecord] = []

        for record in dataset.records:
            cleaned_data, entries = await self._deidentify_record(record)
            cleaned = record.model_copy(update={"data": cleaned_data})
            cleaned_records.append(cleaned)
            for entry in entries:
                audit.add(entry)

        result_dataset = ClinicalDataset(records=cleaned_records)
        logger.info(
            "De-identification complete: %d records processed, %d PHI items removed",
            len(cleaned_records),
            len(audit.entries),
        )

        return DeidentifiedDataset(
            dataset=result_dataset,
            audit_log=audit,
            method="safe_harbor",
        )

    async def _deidentify_record(
        self, record: ClinicalRecord
    ) -> tuple[dict[str, Any], list[AuditEntry]]:
        """Apply all de-identification layers to a single record."""
        entries: list[AuditEntry] = []
        original_key_order = list(record.data.keys())
        data = record.data.copy()

        # Determine patient ID for consistent date shifting
        patient_id = self._extract_patient_id(data)

        # Layer 1: NER on string values
        data, ner_entries = self._apply_ner(data, record.id)
        entries.extend(ner_entries)

        # Layer 2: Pattern matching
        data, pattern_entries = self._apply_patterns(data, record.id)
        entries.extend(pattern_entries)

        # Layer 3: Date shifting
        if self.config.date_shift:
            data, date_entries = self._apply_date_shift(data, patient_id, record.id)
            entries.extend(date_entries)

        # Layer 4: LLM verification (optional)
        if self._llm_verifier:
            llm_entries = await self._apply_llm_verification(data, record.id)
            entries.extend(llm_entries)

        # Preserve original field ordering so consumers see the same
        # key sequence in the de-identified output as in the input.
        data = self._restore_key_order(data, original_key_order)

        return data, entries

    def _apply_ner(
        self, data: dict[str, Any], record_id: str
    ) -> tuple[dict[str, Any], list[AuditEntry]]:
        """Run NER across all string values in the data dict."""
        entries: list[AuditEntry] = []
        result = self._walk_strings(
            data,
            lambda text: self._ner_redact(text, record_id, entries),
        )
        return result, entries

    def _ner_redact(self, text: str, record_id: str, entries: list[AuditEntry]) -> str:
        """NER-redact a single string, logging findings."""
        redacted, ner_entities = self._ner.redact(text)
        for ent in ner_entities:
            entries.append(
                AuditEntry(
                    record_id=record_id,
                    action="PHI_REMOVED",
                    layer="NER",
                    original=ent.text,
                    replacement=f"[{ent.phi_category}]",
                    category=ent.phi_category,
                    confidence=ent.confidence,
                )
            )
        return redacted

    def _apply_patterns(
        self, data: dict[str, Any], record_id: str
    ) -> tuple[dict[str, Any], list[AuditEntry]]:
        """Run pattern matching across the data dict."""
        redacted, matches = self._patterns.redact_dict(data)
        entries = [
            AuditEntry(
                record_id=record_id,
                action="PHI_REMOVED",
                layer="PATTERN",
                original=m.original,
                replacement=m.replacement,
                category=m.category,
                confidence=m.confidence,
            )
            for m in matches
        ]
        return redacted, entries

    def _apply_date_shift(
        self,
        data: dict[str, Any],
        patient_id: str,
        record_id: str,
    ) -> tuple[dict[str, Any], list[AuditEntry]]:
        """Shift dates in the data dict."""
        shifted, shifted_count = self._date_shifter.shift_record_data_with_count(
            data, patient_id
        )
        offset = self._date_shifter.get_offset_days(patient_id)
        entries = [
            AuditEntry(
                record_id=record_id,
                action="DATE_SHIFTED",
                layer="DATE_SHIFT",
                original=f"patient={patient_id}",
                replacement=(f"offset={offset} days; shifted_dates={shifted_count}"),
                category="DATE",
                confidence=1.0,
            )
        ]
        return shifted, entries

    async def _apply_llm_verification(
        self, data: dict[str, Any], record_id: str
    ) -> list[AuditEntry]:
        """Run LLM verification on string values."""
        assert self._llm_verifier is not None
        entries: list[AuditEntry] = []

        # Collect all text for a single LLM call
        texts = self._collect_strings(data)
        combined = "\n---\n".join(texts)

        findings = await self._llm_verifier.verify(combined)
        status = self._llm_verifier.last_status

        if findings:
            for f in findings:
                entries.append(
                    AuditEntry(
                        record_id=record_id,
                        action="PHI_DETECTED_LLM",
                        layer="LLM_VERIFY",
                        original=f.text,
                        replacement=f"[{f.category}]",
                        category=f.category,
                        confidence=f.confidence,
                    )
                )
        elif status == "clean":
            entries.append(
                AuditEntry(
                    record_id=record_id,
                    action="PHI_VERIFIED_CLEAN",
                    layer="LLM_VERIFY",
                    original="",
                    replacement="",
                    category="NONE",
                    confidence=1.0,
                )
            )
        elif status == "skipped":
            entries.append(
                AuditEntry(
                    record_id=record_id,
                    action="PHI_VERIFICATION_SKIPPED",
                    layer="LLM_VERIFY",
                    original="",
                    replacement=self._llm_verifier.last_error or "no API key",
                    category="NONE",
                    confidence=0.0,
                )
            )
        elif status == "error":
            entries.append(
                AuditEntry(
                    record_id=record_id,
                    action="PHI_VERIFICATION_ERROR",
                    layer="LLM_VERIFY",
                    original="",
                    replacement=self._llm_verifier.last_error or "verification failed",
                    category="NONE",
                    confidence=0.0,
                )
            )
        return entries

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _extract_patient_id(data: dict[str, Any]) -> str:
        """Best-effort extraction of a patient identifier from the data."""
        if data.get("id"):
            return str(data["id"])
        subject = data.get("subject", {})
        if isinstance(subject, dict):
            ref = subject.get("reference", "")
            if isinstance(ref, str) and "/" in ref:
                return ref.split("/")[-1]
        return "unknown"

    @staticmethod
    def _walk_strings(
        obj: Any,
        transform: Any,
    ) -> Any:
        """Recursively apply *transform* to every string in a nested structure."""
        if isinstance(obj, str):
            return transform(obj)
        if isinstance(obj, dict):
            return {
                k: SafeHarborEngine._walk_strings(v, transform) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [SafeHarborEngine._walk_strings(item, transform) for item in obj]
        return obj

    @staticmethod
    def _restore_key_order(
        data: dict[str, Any], original_keys: list[str]
    ) -> dict[str, Any]:
        """Re-order *data* so its top-level keys match *original_keys*.

        Keys present in *data* but absent from *original_keys* are
        appended at the end.  Nested dicts are **not** reordered because
        inner structures are already handled by the walk helpers.
        """
        ordered: dict[str, Any] = {}
        for key in original_keys:
            if key in data:
                ordered[key] = data[key]
        # Append any keys that were added during processing
        for key in data:
            if key not in ordered:
                ordered[key] = data[key]
        return ordered

    @staticmethod
    def _collect_strings(obj: Any) -> list[str]:
        """Collect all string values from a nested structure."""
        strings: list[str] = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(SafeHarborEngine._collect_strings(v))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(SafeHarborEngine._collect_strings(item))
        return strings


async def deidentify(
    dataset: ClinicalDataset,
    *,
    method: str = "safe_harbor",
    date_shift: bool = True,
    date_shift_salt: str = "",
    date_shift_range: tuple[int, int] = (-365, 365),
    llm_verification: bool = False,
    llm_model: str = "claude-haiku-4-5",
    llm_api_key: str = "",
) -> DeidentifiedDataset:
    """Convenience function: de-identify a dataset.

    This is the primary public API for de-identification.  It constructs
    and runs a ``SafeHarborEngine`` with the provided configuration.

    Args:
        dataset: Raw clinical dataset.
        method: De-identification method (currently only ``"safe_harbor"``).
        date_shift: Whether to apply date shifting.
        date_shift_salt: Secret salt for date shifting.  Required when
            ``date_shift=True``.
        date_shift_range: Range of random day offsets per patient.
        llm_verification: Whether to run the LLM verification layer.
        llm_model: Which LLM model to use for verification.
        llm_api_key: API key for the LLM provider.

    Returns:
        A ``DeidentifiedDataset`` with audit trail.
    """
    config = SafeHarborConfig(
        date_shift=date_shift,
        date_shift_salt=date_shift_salt,
        date_shift_range=date_shift_range,
        llm_verification=llm_verification,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        use_fallback_ner=True,  # Default to fallback for portability
    )
    engine = SafeHarborEngine(config)
    return await engine.run(dataset)


def ensure_date_shift_salt(config: SafeHarborConfig) -> SafeHarborConfig:
    """Populate a secret salt when date shifting is enabled but unset.

    This is intended for higher-level orchestration paths such as the CLI
    and pipeline defaults.  The lower-level engine still validates that a
    salt is present before it runs.
    """
    if not config.date_shift:
        return config
    if config.date_shift_salt and config.date_shift_salt.strip():
        return config
    return config.model_copy(update={"date_shift_salt": secrets.token_urlsafe(32)})
