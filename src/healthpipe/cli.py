"""Command-line interface for healthpipe.

Provides a ``click``-based CLI for common pipeline operations:
- ``healthpipe ingest`` -- ingest from CSV/FHIR/HL7 sources
- ``healthpipe deidentify`` -- de-identify a dataset
- ``healthpipe synthesize`` -- generate synthetic data
- ``healthpipe audit`` -- view or export audit logs
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger("healthpipe")


def _configure_logging(verbose: bool) -> None:
    """Set up structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(version="0.1.0", prog_name="healthpipe")
def main(verbose: bool) -> None:
    """healthpipe -- Privacy-preserving clinical data pipeline."""
    _configure_logging(verbose)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "fhir", "hl7", "pdf", "auto"]),
    default="auto",
    help="Source format (default: auto-detect).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for ingested dataset JSON.",
)
def ingest(path: str, fmt: str, output: str | None) -> None:
    """Ingest clinical data from PATH."""
    from healthpipe.ingest.csv_mapper import CSVSource
    from healthpipe.ingest.fhir import FHIRSource
    from healthpipe.ingest.hl7v2 import HL7v2Source
    from healthpipe.ingest.pdf_ocr import PDFSource
    from healthpipe.pipeline import ingest as pipeline_ingest

    source: CSVSource | FHIRSource | HL7v2Source | PDFSource

    source_path = Path(path)
    suffix = source_path.suffix.lower()

    if fmt == "auto":
        if suffix == ".csv":
            fmt = "csv"
        elif suffix == ".json":
            fmt = "fhir"
        elif suffix in (".hl7", ".hl7v2"):
            fmt = "hl7"
        elif suffix == ".pdf":
            fmt = "pdf"
        else:
            click.echo(
                f"Cannot auto-detect format for {suffix}. Use --format to specify.",
                err=True,
            )
            sys.exit(1)

    if fmt == "csv":
        source = CSVSource(path=path)
    elif fmt == "fhir":
        source = FHIRSource(url=path)
    elif fmt == "hl7":
        source = HL7v2Source(path=path)
    else:
        source = PDFSource(path=path)
    dataset = asyncio.run(pipeline_ingest([source]))

    click.echo(f"Ingested {len(dataset.records)} records")
    click.echo(f"  Patients: {dataset.patients.count()}")
    click.echo(f"  Observations: {dataset.observations.count()}")
    click.echo(f"  Conditions: {dataset.conditions.count()}")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Dataset saved to {out_path}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path for de-identified dataset.",
)
@click.option(
    "--date-shift/--no-date-shift",
    default=True,
    help="Enable date shifting (default: on).",
)
@click.option(
    "--date-shift-salt",
    type=str,
    default="",
    help="Secret salt for date shifting. Generated automatically if omitted.",
)
@click.option(
    "--audit-log",
    type=click.Path(),
    default=None,
    help="Path to save the audit log.",
)
def deidentify(
    input_path: str,
    output: str,
    date_shift: bool,
    date_shift_salt: str,
    audit_log: str | None,
) -> None:
    """De-identify clinical data from INPUT_PATH."""
    from healthpipe.deidentify.safe_harbor import (
        SafeHarborConfig,
        SafeHarborEngine,
        ensure_date_shift_salt,
    )
    from healthpipe.ingest.schema import ClinicalDataset

    raw = Path(input_path).read_text(encoding="utf-8")
    dataset = ClinicalDataset.model_validate_json(raw)

    config = SafeHarborConfig(
        date_shift=date_shift,
        date_shift_salt=date_shift_salt,
        use_fallback_ner=True,
    )
    config = ensure_date_shift_salt(config)
    engine = SafeHarborEngine(config)
    result = asyncio.run(engine.run(dataset))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"De-identified {len(result.records)} records -> {out_path}")
    click.echo(f"PHI items removed: {result.audit_log.phi_removed_count}")

    if audit_log:
        result.audit_log.save(audit_log)
        click.echo(f"Audit log saved to {audit_log}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output path for synthetic dataset.",
)
@click.option(
    "--n-patients",
    type=int,
    default=1000,
    help="Number of synthetic patients to generate.",
)
@click.option(
    "--method",
    type=click.Choice(["gaussian_copula", "ctgan"]),
    default="gaussian_copula",
    help="Synthesis method.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
def synthesize(
    input_path: str,
    output: str,
    n_patients: int,
    method: str,
    seed: int | None,
) -> None:
    """Generate synthetic data from de-identified INPUT_PATH."""
    from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
    from healthpipe.synthetic.generator import SyntheticGenerator

    raw = Path(input_path).read_text(encoding="utf-8")
    deid_data = DeidentifiedDataset.model_validate_json(raw)

    gen = SyntheticGenerator(
        n_patients=n_patients,
        method=method,
        seed=seed,
    )
    synthetic = gen.generate(deid_data)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(synthetic.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Generated {len(synthetic.records)} synthetic records -> {out_path}")


@main.command()
@click.argument("audit_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "markdown", "summary"]),
    default="summary",
    help="Output format.",
)
def audit(audit_path: str, fmt: str) -> None:
    """View or export an audit log from AUDIT_PATH."""
    from healthpipe.audit.compliance import ComplianceReporter
    from healthpipe.audit.logger import AuditLog

    raw = Path(audit_path).read_text(encoding="utf-8")
    data = json.loads(raw)

    # Reconstruct audit log from entries
    log = AuditLog()
    for entry_data in data.get("entries", []):
        from healthpipe.audit.logger import AuditEntry

        log.add(AuditEntry.model_validate(entry_data))

    if fmt == "summary":
        summary = log.summary
        click.echo("Audit Log Summary")
        click.echo(f"  Total entries: {summary['total_entries']}")
        click.echo(f"  PHI removed: {summary['phi_removed']}")
        click.echo(f"  By layer: {summary['by_layer']}")
        click.echo(f"  By category: {summary['by_category']}")

    elif fmt == "json":
        click.echo(log.to_json(safe=True))

    elif fmt == "markdown":
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)
        click.echo(ComplianceReporter.to_markdown(report))


if __name__ == "__main__":
    main()
