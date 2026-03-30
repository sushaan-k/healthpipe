"""Example: CSV de-identification.

Demonstrates ingesting a CSV file, auto-mapping columns to FHIR,
running the de-identification engine, and saving the audit trail.
"""

import asyncio
import csv
import tempfile
from pathlib import Path

import healthpipe as hp

# -- Generate sample CSV data --------------------------------------------------

SAMPLE_ROWS = [
    {
        "patient_id": "P001",
        "first_name": "John",
        "last_name": "Smith",
        "dob": "1985-03-15",
        "gender": "M",
        "phone": "555-123-4567",
        "email": "john.smith@hospital.org",
        "ssn": "123-45-6789",
        "city": "Springfield",
        "state": "IL",
        "zip": "62704",
        "glucose": "110",
        "diagnosis": "E11.9",
    },
    {
        "patient_id": "P002",
        "first_name": "Jane",
        "last_name": "Doe",
        "dob": "1992-07-20",
        "gender": "F",
        "phone": "(555) 987-6543",
        "email": "jane.doe@clinic.com",
        "ssn": "987-65-4321",
        "city": "Chicago",
        "state": "IL",
        "zip": "60601",
        "glucose": "95",
        "diagnosis": "I10",
    },
    {
        "patient_id": "P003",
        "first_name": "Robert",
        "last_name": "Johnson",
        "dob": "1970-12-01",
        "gender": "M",
        "phone": "555.456.7890",
        "email": "rjohnson@mail.com",
        "ssn": "456-78-9012",
        "city": "Boston",
        "state": "MA",
        "zip": "02101",
        "glucose": "130",
        "diagnosis": "E11.65",
    },
]


async def main() -> None:
    """Run the CSV de-identification example."""
    # Write sample CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_ROWS[0].keys())
        writer.writeheader()
        writer.writerows(SAMPLE_ROWS)
        csv_path = f.name

    print(f"Sample CSV written to: {csv_path}")
    print()

    # Step 1: Ingest
    print("=== Ingest ===")
    dataset = await hp.ingest([
        hp.CSVSource(path=csv_path, mapping="auto"),
    ])
    print(f"Records ingested: {len(dataset.records)}")
    print(f"  Patients: {dataset.patients.count()}")
    print(f"  Observations: {dataset.observations.count()}")
    print(f"  Conditions: {dataset.conditions.count()}")
    print()

    # Step 2: De-identify
    print("=== De-identify ===")
    deidentified = await hp.deidentify(
        dataset,
        date_shift=True,
        date_shift_range=(-180, 180),
        llm_verification=False,
    )
    print(f"PHI items removed: {deidentified.audit_log.phi_removed_count}")
    print()

    # Show audit summary
    summary = deidentified.audit_log.summary
    print("Audit summary:")
    print(f"  Total entries: {summary['total_entries']}")
    print(f"  By layer: {summary['by_layer']}")
    print(f"  By category: {summary['by_category']}")
    print()

    # Step 3: Save results
    output_dir = Path(tempfile.mkdtemp())
    audit_path = output_dir / "audit_log.json"
    deidentified.audit_log.save(audit_path)
    print(f"Audit log saved to: {audit_path}")

    # Compliance report
    reporter = hp.ComplianceReporter()
    report = reporter.generate(
        audit_log=deidentified.audit_log,
        record_count=len(deidentified.records),
    )
    report_path = output_dir / "compliance_report.json"
    hp.ComplianceReporter.save_json(report, report_path)
    print(f"Compliance report saved to: {report_path}")

    # Cleanup
    Path(csv_path).unlink()


if __name__ == "__main__":
    asyncio.run(main())
