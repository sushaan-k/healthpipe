"""Example: Synthetic data generation.

Demonstrates generating synthetic patient data from a de-identified
source, validating re-identification risk, and evaluating data utility.
"""

import asyncio

import numpy as np

import healthpipe as hp
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType


def _build_sample_dataset(n: int = 100) -> ClinicalDataset:
    """Build a realistic sample dataset for synthetic generation."""
    np.random.seed(42)
    records: list[ClinicalRecord] = []

    diagnoses = ["E11.9", "I10", "J45.909", "M54.5", "F32.9"]
    diagnosis_names = [
        "Type 2 diabetes",
        "Essential hypertension",
        "Asthma",
        "Low back pain",
        "Major depression",
    ]

    for _ in range(n):
        age = int(np.random.normal(55, 15))
        age = max(18, min(90, age))

        # Realistic correlations: older patients more likely to have
        # diabetes/hypertension
        if age > 60:
            dx_idx = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.3, 0.1, 0.2, 0.1])
        else:
            dx_idx = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.1, 0.3, 0.3, 0.2])

        glucose = float(np.random.normal(100 + (age - 50) * 0.5, 20))
        systolic = float(np.random.normal(120 + (age - 50) * 0.3, 15))
        hemoglobin = float(np.random.normal(14 - (age - 50) * 0.01, 1.5))

        records.append(
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={
                    "age": age,
                    "glucose": round(glucose, 1),
                    "systolic_bp": round(systolic, 1),
                    "hemoglobin": round(hemoglobin, 1),
                    "diagnosis_code": diagnoses[dx_idx],
                    "diagnosis_name": diagnosis_names[dx_idx],
                },
                source_format="SYNTHETIC_EXAMPLE",
            )
        )

    return ClinicalDataset(records=records)


async def main() -> None:
    """Run the synthetic generation example."""
    # Step 1: Build sample de-identified dataset
    print("=== Step 1: Prepare Source Data ===")
    dataset = _build_sample_dataset(n=100)
    deidentified = hp.DeidentifiedDataset(dataset=dataset)
    print(f"Source records: {len(deidentified.records)}")
    print()

    # Step 2: Generate synthetic data
    print("=== Step 2: Generate Synthetic Data ===")
    synthetic = await hp.synthesize(
        deidentified,
        n_patients=200,
        method="gaussian_copula",
        preserve_correlations=True,
        validate=True,
        seed=42,
    )
    print(f"Synthetic records generated: {len(synthetic.records)}")
    print()

    # Step 3: Evaluate utility
    print("=== Step 3: Evaluate Utility ===")
    utility = hp.evaluate_utility(synthetic, deidentified)
    print(f"Statistical fidelity: {utility.fidelity:.2%}")
    print(f"ML utility (correlation preservation): {utility.ml_utility:.2%}")
    print(f"Re-identification risk: {utility.reidentification_risk:.6f}")
    print()

    if utility.column_scores:
        print("Per-column fidelity scores:")
        for col, score in sorted(
            utility.column_scores.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {col}: {score:.2%}")
    print()

    # Step 4: Privacy-preserving statistics on synthetic data
    print("=== Step 4: Statistics (for reference) ===")
    synth_deid = hp.DeidentifiedDataset(dataset=synthetic)
    stats = hp.private_stats(
        synth_deid,
        epsilon=1.0,
        queries=[
            hp.Count(field="observation"),
            hp.Histogram(field="diagnosis_code"),
        ],
    )
    print(f"DP query results: {stats.results}")
    print(f"Privacy budget remaining: {stats.budget_remaining:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
