"""healthpipe -- Privacy-preserving clinical data pipeline.

End-to-end framework for ingesting, de-identifying, transforming, and
analysing clinical/health data with built-in differential privacy,
HIPAA-compliant audit logging, and synthetic data generation.

Usage::

    import healthpipe as hp

    dataset = await hp.ingest([
        hp.CSVSource(path="patients.csv"),
        hp.FHIRSource(url="https://fhir.example.org/R4"),
    ])

    deidentified = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift_salt="your-secret-salt",
    )
    synthetic = await hp.synthesize(deidentified, n_patients=1000)
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Ingest sources -----------------------------------------------------------
from healthpipe.audit.compliance import ComplianceReport, ComplianceReporter
from healthpipe.audit.lineage import LineageNode, LineageTracker

# --- Audit & compliance -------------------------------------------------------
from healthpipe.audit.logger import AuditEntry, AuditLog

# --- De-identification --------------------------------------------------------
from healthpipe.deidentify.safe_harbor import (
    DeidentifiedDataset,
    SafeHarborConfig,
    SafeHarborEngine,
    deidentify,
)

# --- Exceptions ---------------------------------------------------------------
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
from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.fhir import FHIRAuth, FHIRSource
from healthpipe.ingest.hl7v2 import HL7v2Source
from healthpipe.ingest.pdf_ocr import PDFSource
from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    ResourceType,
)

# --- Pipeline -----------------------------------------------------------------
from healthpipe.pipeline import Pipeline, PipelineConfig, PipelineResult, ingest

# --- Privacy ------------------------------------------------------------------
from healthpipe.privacy.budget import PrivacyBudget
from healthpipe.privacy.differential import (
    Count,
    DPResult,
    GaussianMechanism,
    Histogram,
    LaplaceMechanism,
    Mean,
    private_stats,
)
from healthpipe.privacy.k_anonymity import KAnonymityChecker, LDiversityChecker

# --- Synthetic data -----------------------------------------------------------
from healthpipe.synthetic.generator import SyntheticGenerator, synthesize
from healthpipe.synthetic.utility import UtilityReport, evaluate_utility
from healthpipe.synthetic.validator import ReidentificationValidator

__all__ = [
    "AuditEntry",
    "AuditError",
    "AuditLog",
    "BudgetExhaustedError",
    "CSVSource",
    "ClinicalDataset",
    "ClinicalRecord",
    "ComplianceReport",
    "ComplianceReporter",
    "Count",
    "DPResult",
    "DateShiftError",
    "DeidentificationError",
    "DeidentifiedDataset",
    "FHIRAuth",
    "FHIRSource",
    "FHIRValidationError",
    "GaussianMechanism",
    "HL7ParseError",
    "HL7v2Source",
    "HealthPipeError",
    "Histogram",
    "IngestError",
    "KAnonymityChecker",
    "KAnonymityError",
    "LDiversityChecker",
    "LaplaceMechanism",
    "LineageError",
    "LineageNode",
    "LineageTracker",
    "Mean",
    "PDFSource",
    "PHIDetectionError",
    "Pipeline",
    "PipelineConfig",
    "PipelineConfigError",
    "PipelineError",
    "PipelineResult",
    "PrivacyBudget",
    "PrivacyError",
    "ReidentificationRiskError",
    "ReidentificationValidator",
    "ResourceType",
    "SafeHarborConfig",
    "SafeHarborEngine",
    "SyntheticDataError",
    "SyntheticGenerator",
    "UnsupportedFormatError",
    "UtilityReport",
    "__version__",
    "deidentify",
    "evaluate_utility",
    "ingest",
    "private_stats",
    "synthesize",
]
