"""Main pipeline orchestration.

Provides the high-level ``Pipeline`` class that chains ingest, de-identification,
privacy mechanisms, synthetic generation, and audit into a single configurable
workflow.  Also exposes the ``ingest()`` convenience function used in the
top-level ``healthpipe`` API.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from healthpipe.audit.lineage import LineageTracker
from healthpipe.audit.logger import AuditLog
from healthpipe.deidentify.ner import ClinicalNER
from healthpipe.deidentify.patterns import PatternMatcher
from healthpipe.deidentify.safe_harbor import (
    DeidentifiedDataset,
    SafeHarborConfig,
    SafeHarborEngine,
)
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord

logger = logging.getLogger(__name__)


class DryRunFinding(BaseModel):
    """A single PHI detection found during a dry-run scan.

    Attributes:
        record_id: Identifier of the record containing the PHI.
        category: HIPAA identifier category (e.g. ``SSN``, ``EMAIL``).
        original: The detected PHI text.
        replacement: What the placeholder would be.
        detection_method: How the PHI was detected (``pattern`` or ``ner``).
        confidence: Detection confidence score.
        field_path: Dot-delimited path to the field containing the PHI,
            or an empty string if detected in a top-level scan.
    """

    record_id: str = ""
    category: str = ""
    original: str = ""
    replacement: str = ""
    detection_method: str = ""
    confidence: float = 0.0
    field_path: str = ""


class DryRunReport(BaseModel):
    """Report produced by a dry-run pipeline execution.

    Contains all detected PHI findings without modifying the input data.

    Attributes:
        findings: List of detected PHI items.
        total_records_scanned: Number of records examined.
        total_phi_found: Total number of PHI detections.
        categories_found: Set of unique PHI categories detected.
    """

    findings: list[DryRunFinding] = Field(default_factory=list)
    total_records_scanned: int = 0
    total_phi_found: int = 0
    categories_found: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Configuration for a healthpipe pipeline run.

    Attributes:
        deidentify: Whether to run de-identification.
        deid_config: Configuration for the Safe Harbor engine.
        generate_synthetic: Whether to produce synthetic data.
        synthetic_n_patients: Number of synthetic patients to generate.
        synthetic_method: Synthetic generation method.
        validate_synthetic: Whether to validate synthetic data.
        track_lineage: Whether to record data lineage.
        dry_run: If True, detect PHI without modifying data.
    """

    deidentify: bool = True
    deid_config: SafeHarborConfig = Field(default_factory=SafeHarborConfig)
    generate_synthetic: bool = False
    synthetic_n_patients: int = 1000
    synthetic_method: str = "gaussian_copula"
    validate_synthetic: bool = True
    track_lineage: bool = True
    dry_run: bool = False


class PipelineResult(BaseModel):
    """Container for all outputs of a pipeline run.

    Attributes:
        raw_dataset: The original ingested dataset.
        deidentified: The de-identified dataset (if deidentification was run).
        synthetic: The synthetic dataset (if generation was run).
        audit_log: Combined audit log.
        dry_run_report: Report of detected PHI (only when ``dry_run=True``).
    """

    raw_dataset: ClinicalDataset = Field(default_factory=ClinicalDataset)
    deidentified: DeidentifiedDataset | None = None
    synthetic: ClinicalDataset | None = None
    audit_log: AuditLog = Field(default_factory=AuditLog)
    dry_run_report: DryRunReport | None = None

    model_config = {"arbitrary_types_allowed": True}


class Pipeline:
    """End-to-end clinical data pipeline.

    Chains together ingest sources, de-identification, optional
    synthetic data generation, and audit logging.

    Usage::

        pipeline = Pipeline(config)
        result = await pipeline.run(sources)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._lineage = LineageTracker() if self.config.track_lineage else None

    async def run(
        self,
        sources: list[Any],
    ) -> PipelineResult:
        """Execute the full pipeline.

        When ``dry_run=True`` in the config, the pipeline scans for PHI
        without modifying any data and returns a ``DryRunReport`` in the
        result.

        Args:
            sources: List of ingest source objects (``FHIRSource``,
                ``CSVSource``, ``HL7v2Source``, ``PDFSource``).

        Returns:
            A ``PipelineResult`` containing all outputs.
        """
        audit = AuditLog()

        # Step 1: Ingest
        logger.info("Pipeline: ingesting from %d source(s)", len(sources))
        dataset = await ingest(sources)

        if self._lineage:
            for record in dataset.records:
                self._lineage.record_operation(
                    record_id=record.id,
                    operation="ingest",
                    metadata={
                        "source_format": record.source_format,
                        "source_uri": record.source_uri,
                    },
                    checksum_after=record.checksum,
                )

        result = PipelineResult(raw_dataset=dataset, audit_log=audit)

        # Dry-run mode: detect PHI without modifying data
        if self.config.dry_run:
            logger.info("Pipeline: dry-run mode -- scanning for PHI only")
            report = self._scan_for_phi(dataset)
            result.dry_run_report = report
            logger.info(
                "Dry-run complete: %d records scanned, %d PHI items detected",
                report.total_records_scanned,
                report.total_phi_found,
            )
            return result

        # Step 2: De-identification
        if self.config.deidentify:
            logger.info("Pipeline: running de-identification")
            from healthpipe.deidentify.safe_harbor import ensure_date_shift_salt

            deid_config = ensure_date_shift_salt(self.config.deid_config)
            engine = SafeHarborEngine(deid_config)
            deidentified = await engine.run(dataset)
            result.deidentified = deidentified

            for entry in deidentified.audit_log.entries:
                audit.add(entry)

            if self._lineage:
                for record in deidentified.records:
                    self._lineage.record_operation(
                        record_id=record.id,
                        operation="deidentify",
                        metadata={"method": deidentified.method},
                        checksum_after=record.checksum,
                    )

        # Step 3: Synthetic data generation
        if self.config.generate_synthetic and result.deidentified:
            logger.info("Pipeline: generating synthetic data")
            from healthpipe.synthetic.generator import SyntheticGenerator

            gen = SyntheticGenerator(
                n_patients=self.config.synthetic_n_patients,
                method=self.config.synthetic_method,
            )
            synthetic = gen.generate(result.deidentified)

            if self.config.validate_synthetic:
                from healthpipe.synthetic.validator import (
                    ReidentificationValidator,
                )

                validator = ReidentificationValidator()
                validator.validate(source=result.deidentified, synthetic=synthetic)

            result.synthetic = synthetic

        logger.info(
            "Pipeline complete: %d records ingested, %d audit entries",
            len(dataset.records),
            len(audit.entries),
        )
        return result

    def _scan_for_phi(self, dataset: ClinicalDataset) -> DryRunReport:
        """Scan all records for PHI without modifying any data.

        Uses both NER and pattern matching to detect PHI, then returns
        a consolidated report.
        """
        ner = ClinicalNER(
            model_name=self.config.deid_config.ner_model,
            use_fallback=self.config.deid_config.use_fallback_ner,
        )
        pattern_matcher = PatternMatcher()

        findings: list[DryRunFinding] = []

        for record in dataset.records:
            record_findings = self._scan_record(record, ner, pattern_matcher)
            findings.extend(record_findings)

        categories = sorted({f.category for f in findings})

        return DryRunReport(
            findings=findings,
            total_records_scanned=len(dataset.records),
            total_phi_found=len(findings),
            categories_found=categories,
        )

    @staticmethod
    def _scan_record(
        record: ClinicalRecord,
        ner: ClinicalNER,
        pattern_matcher: PatternMatcher,
    ) -> list[DryRunFinding]:
        """Scan a single record's string fields for PHI."""
        findings: list[DryRunFinding] = []
        strings = _collect_strings_with_paths(record.data)

        for field_path, text in strings:
            # NER scan
            entities = ner.extract(text)
            for ent in entities:
                findings.append(
                    DryRunFinding(
                        record_id=record.id,
                        category=ent.phi_category,
                        original=ent.text,
                        replacement=f"[{ent.phi_category}]",
                        detection_method=ent.detection_method,
                        confidence=ent.confidence,
                        field_path=field_path,
                    )
                )

            # Pattern scan
            matches = pattern_matcher.scan(text)
            for m in matches:
                findings.append(
                    DryRunFinding(
                        record_id=record.id,
                        category=m.category,
                        original=m.original,
                        replacement=m.replacement,
                        detection_method=m.detection_method,
                        confidence=m.confidence,
                        field_path=field_path,
                    )
                )

        return findings

    @property
    def lineage(self) -> LineageTracker | None:
        """Access the lineage tracker (if enabled)."""
        return self._lineage


def _collect_strings_with_paths(
    obj: Any, prefix: str = ""
) -> list[tuple[str, str]]:
    """Walk a nested structure and collect ``(dotted_path, value)`` pairs
    for every string leaf."""
    pairs: list[tuple[str, str]] = []
    if isinstance(obj, str):
        pairs.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child_path = f"{prefix}.{k}" if prefix else k
            pairs.extend(_collect_strings_with_paths(v, child_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            child_path = f"{prefix}[{idx}]"
            pairs.extend(_collect_strings_with_paths(item, child_path))
    return pairs


async def ingest(
    sources: list[Any],
) -> ClinicalDataset:
    """Ingest data from multiple sources and merge into a single dataset.

    This is the primary public API for data ingestion.  Each source is
    an ingest adapter (``FHIRSource``, ``CSVSource``, ``HL7v2Source``,
    ``PDFSource``) that implements an ``async ingest()`` method.

    Args:
        sources: List of ingest source objects.

    Returns:
        A merged ``ClinicalDataset`` containing all ingested records.
    """
    combined = ClinicalDataset()

    for source in sources:
        logger.info("Ingesting from %s", type(source).__name__)
        dataset = await source.ingest()
        combined = combined.merge(dataset)

    logger.info(
        "Ingested %d total records from %d source(s)",
        len(combined.records),
        len(sources),
    )
    return combined
