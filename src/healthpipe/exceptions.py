"""Custom exception hierarchy for healthpipe.

All exceptions inherit from HealthPipeError so callers can catch
the full family with a single except clause when desired.
"""

from __future__ import annotations


class HealthPipeError(Exception):
    """Base exception for all healthpipe errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(message)


# --- Ingest errors -----------------------------------------------------------


class IngestError(HealthPipeError):
    """Raised when a data source cannot be ingested."""


class UnsupportedFormatError(IngestError):
    """The provided file or stream is not a recognised clinical format."""


class FHIRValidationError(IngestError):
    """A FHIR resource failed schema validation."""


class HL7ParseError(IngestError):
    """An HL7v2 message could not be parsed."""


# --- De-identification errors -------------------------------------------------


class DeidentificationError(HealthPipeError):
    """Raised when de-identification fails or is incomplete."""


class PHIDetectionError(DeidentificationError):
    """PHI was detected but could not be safely removed."""


class DateShiftError(DeidentificationError):
    """Date-shifting encountered an unrecoverable inconsistency."""


# --- Privacy errors -----------------------------------------------------------


class PrivacyError(HealthPipeError):
    """Base for privacy-mechanism failures."""


class BudgetExhaustedError(PrivacyError):
    """The differential-privacy budget (epsilon) has been fully consumed."""

    def __init__(self, epsilon_remaining: float) -> None:
        super().__init__(
            f"Privacy budget exhausted (remaining epsilon={epsilon_remaining:.6f}). "
            "No further queries can be answered without violating the privacy guarantee.",
            detail=f"epsilon_remaining={epsilon_remaining}",
        )
        self.epsilon_remaining = epsilon_remaining


class KAnonymityError(PrivacyError):
    """The dataset does not satisfy the requested k-anonymity threshold."""


# --- Synthetic data errors ----------------------------------------------------


class SyntheticDataError(HealthPipeError):
    """Raised during synthetic data generation or validation."""


class ReidentificationRiskError(SyntheticDataError):
    """Synthetic records are too similar to real records."""


# --- Audit errors -------------------------------------------------------------


class AuditError(HealthPipeError):
    """Raised when an audit operation fails."""


class LineageError(AuditError):
    """Data lineage chain is broken or inconsistent."""


# --- Pipeline errors ----------------------------------------------------------


class PipelineError(HealthPipeError):
    """Raised when the orchestration pipeline encounters an error."""


class PipelineConfigError(PipelineError):
    """The pipeline configuration is invalid."""
