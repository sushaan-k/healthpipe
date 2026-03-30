"""Example: End-to-end FHIR pipeline.

Demonstrates ingesting a FHIR Bundle, running de-identification,
computing differentially private statistics, and generating a
compliance report.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import healthpipe as hp

# -- Sample FHIR Bundle -------------------------------------------------------

SAMPLE_BUNDLE = {
    "resourceType": "Bundle",
    "type": "searchset",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "patient-001",
                "name": [{"family": "Anderson", "given": ["Sarah"]}],
                "birthDate": "1978-11-22",
                "gender": "female",
                "address": [
                    {
                        "line": ["456 Oak Avenue"],
                        "city": "Chicago",
                        "state": "IL",
                        "postalCode": "60601",
                    }
                ],
                "telecom": [
                    {"system": "phone", "value": "(312) 555-0199"},
                    {"system": "email", "value": "s.anderson@email.com"},
                ],
                "identifier": [
                    {"system": "SSN", "value": "987-65-4321"},
                    {"system": "MRN", "value": "MRN-78901"},
                ],
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-001",
                "status": "final",
                "code": {
                    "system": "LOINC",
                    "code": "2345-7",
                    "display": "Glucose [Mass/volume] in Serum or Plasma",
                },
                "subject": {"reference": "Patient/patient-001"},
                "effectiveDateTime": "2025-03-15T10:30:00Z",
                "valueQuantity": {"value": 95, "unit": "mg/dL"},
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-001",
                "code": {
                    "system": "ICD-10-CM",
                    "code": "E11.9",
                    "display": "Type 2 diabetes mellitus without complications",
                },
                "subject": {"reference": "Patient/patient-001"},
                "clinicalStatus": "active",
            }
        },
    ],
}


async def main() -> None:
    """Run the example pipeline."""
    # Step 1: Write sample FHIR bundle to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(SAMPLE_BUNDLE, f)
        bundle_path = f.name

    # Step 2: Ingest from FHIR Bundle
    print("=== Step 1: Ingest ===")
    dataset = await hp.ingest([hp.FHIRSource(url=bundle_path)])
    print(f"Patients: {dataset.patients.count()}")
    print(f"Observations: {dataset.observations.count()}")
    print(f"Conditions: {dataset.conditions.count()}")
    print()

    # Step 3: De-identify
    print("=== Step 2: De-identify (Safe Harbor) ===")
    deidentified = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift=True,
        date_shift_range=(-365, 365),
        llm_verification=False,
    )
    print(f"Records de-identified: {len(deidentified.records)}")
    print(f"PHI items removed: {deidentified.audit_log.phi_removed_count}")
    print()
    print("Audit log:")
    print(deidentified.audit_log)
    print()

    # Step 4: Differentially private statistics
    print("=== Step 3: Private Statistics ===")
    stats = hp.private_stats(
        deidentified,
        epsilon=1.0,
        queries=[
            hp.Count(field="patient"),
            hp.Histogram(field="diagnosis"),
        ],
    )
    print(f"Results: {stats.results}")
    print(f"Privacy budget remaining: {stats.budget_remaining:.4f}")
    print()

    # Step 5: Compliance report
    print("=== Step 4: Compliance Report ===")
    reporter = hp.ComplianceReporter()
    report = reporter.generate(
        audit_log=deidentified.audit_log,
        record_count=len(deidentified.records),
    )
    print(hp.ComplianceReporter.to_markdown(report))

    # Cleanup
    Path(bundle_path).unlink()


if __name__ == "__main__":
    asyncio.run(main())
