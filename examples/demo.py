#!/usr/bin/env python3
"""Offline demo for healthpipe."""

from __future__ import annotations

import asyncio

from healthpipe import ClinicalDataset, ClinicalRecord, ResourceType, deidentify


async def main() -> None:
    dataset = ClinicalDataset(
        records=[
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                source_format="demo",
                data={
                    "name": [{"given": ["Jane"], "family": "Doe"}],
                    "birthDate": "1987-04-10",
                    "telecom": [{"system": "phone", "value": "555-111-2222"}],
                    "address": [
                        {"line": ["742 Evergreen Terrace"], "city": "Springfield"}
                    ],
                },
            ),
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                source_format="demo",
                data={
                    "code": {"text": "Hemoglobin A1c"},
                    "subject": {"display": "Jane Doe"},
                    "effectiveDateTime": "2025-02-18T14:30:00Z",
                    "valueString": "A1c 7.2% for Jane Doe",
                },
            ),
        ]
    )

    result = await deidentify(
        dataset,
        date_shift=True,
        date_shift_salt="demo-secret",
        llm_verification=False,
    )

    print("healthpipe demo")
    print(f"records processed: {len(result.records)}")
    print(f"phi removed: {result.audit_log.phi_removed_count}")
    print(f"audit entries: {len(result.audit_log.entries)}")
    print("\nfirst sanitized record:\n")
    print(result.records[0].model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
