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
from healthpipe.deidentify.safe_harbor import (
    DeidentifiedDataset,
    SafeHarborConfig,
    SafeHarborEngine,
)
from healthpipe.ingest.schema import ClinicalDataset

logger = logging.getLogger(__name__)


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
    """

    deidentify: bool = True
    deid_config: SafeHarborConfig = Field(default_factory=SafeHarborConfig)
    generate_synthetic: bool = False
    synthetic_n_patients: int = 1000
    synthetic_method: str = "gaussian_copula"
    validate_synthetic: bool = True
    track_lineage: bool = True


class PipelineResult(BaseModel):
    """Container for all outputs of a pipeline run.

    Attributes:
        raw_dataset: The original ingested dataset.
        deidentified: The de-identified dataset (if deidentification was run).
        synthetic: The synthetic dataset (if generation was run).
        audit_log: Combined audit log.
    """

    raw_dataset: ClinicalDataset = Field(default_factory=ClinicalDataset)
    deidentified: DeidentifiedDataset | None = None
    synthetic: ClinicalDataset | None = None
    audit_log: AuditLog = Field(default_factory=AuditLog)

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

    @property
    def lineage(self) -> LineageTracker | None:
        """Access the lineage tracker (if enabled)."""
        return self._lineage


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
