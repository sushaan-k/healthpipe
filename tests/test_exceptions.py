"""Tests for the exception hierarchy."""

from __future__ import annotations

from healthpipe.exceptions import (
    AuditError,
    BudgetExhaustedError,
    DateShiftError,
    DeidentificationError,
    FHIRValidationError,
    HealthPipeError,
    HL7ParseError,
    IngestError,
    KAnonymityError,
    LineageError,
    PHIDetectionError,
    PipelineConfigError,
    PipelineError,
    PrivacyError,
    ReidentificationRiskError,
    SyntheticDataError,
    UnsupportedFormatError,
)


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        exc = HealthPipeError("test message", detail="extra info")
        assert str(exc) == "test message"
        assert exc.detail == "extra info"

    def test_base_exception_no_detail(self) -> None:
        exc = HealthPipeError("test")
        assert exc.detail is None

    def test_ingest_errors_inherit(self) -> None:
        assert issubclass(IngestError, HealthPipeError)
        assert issubclass(UnsupportedFormatError, IngestError)
        assert issubclass(FHIRValidationError, IngestError)
        assert issubclass(HL7ParseError, IngestError)

    def test_deidentification_errors_inherit(self) -> None:
        assert issubclass(DeidentificationError, HealthPipeError)
        assert issubclass(PHIDetectionError, DeidentificationError)
        assert issubclass(DateShiftError, DeidentificationError)

    def test_privacy_errors_inherit(self) -> None:
        assert issubclass(PrivacyError, HealthPipeError)
        assert issubclass(BudgetExhaustedError, PrivacyError)
        assert issubclass(KAnonymityError, PrivacyError)

    def test_synthetic_errors_inherit(self) -> None:
        assert issubclass(SyntheticDataError, HealthPipeError)
        assert issubclass(ReidentificationRiskError, SyntheticDataError)

    def test_audit_errors_inherit(self) -> None:
        assert issubclass(AuditError, HealthPipeError)
        assert issubclass(LineageError, AuditError)

    def test_pipeline_errors_inherit(self) -> None:
        assert issubclass(PipelineError, HealthPipeError)
        assert issubclass(PipelineConfigError, PipelineError)

    def test_budget_exhausted_error(self) -> None:
        exc = BudgetExhaustedError(epsilon_remaining=0.001)
        assert "exhausted" in str(exc)
        assert exc.epsilon_remaining == 0.001
        assert exc.detail is not None
