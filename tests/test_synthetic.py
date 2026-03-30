"""Tests for synthetic data generation and validation."""

from __future__ import annotations

import pytest

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.exceptions import ReidentificationRiskError
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType
from healthpipe.synthetic.generator import SyntheticGenerator
from healthpipe.synthetic.utility import UtilityReport, evaluate_utility
from healthpipe.synthetic.validator import ReidentificationValidator, RiskReport


def _make_numeric_dataset(n: int = 50) -> DeidentifiedDataset:
    """Create a dataset with numeric data for testing."""
    import numpy as np

    np.random.seed(42)
    records = []
    for _ in range(n):
        records.append(
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={
                    "glucose": float(np.random.normal(100, 15)),
                    "hemoglobin": float(np.random.normal(14, 2)),
                    "age": float(np.random.randint(20, 80)),
                    "diagnosis": np.random.choice(
                        ["diabetes", "hypertension", "healthy"]
                    ),
                },
                source_format="TEST",
            )
        )
    return DeidentifiedDataset(dataset=ClinicalDataset(records=records))


class TestSyntheticGenerator:
    def test_gaussian_copula_generation(self) -> None:
        source = _make_numeric_dataset(50)
        gen = SyntheticGenerator(
            n_patients=20,
            method="gaussian_copula",
            seed=42,
        )
        result = gen.generate(source)
        assert len(result.records) == 20

    def test_preserves_record_structure(self) -> None:
        source = _make_numeric_dataset(30)
        gen = SyntheticGenerator(n_patients=10, seed=42)
        result = gen.generate(source)

        for record in result.records:
            assert record.source_format == "SYNTHETIC"
            assert record.resource_type is not None

    def test_empty_source(self) -> None:
        empty = DeidentifiedDataset(dataset=ClinicalDataset())
        gen = SyntheticGenerator(n_patients=10)
        result = gen.generate(empty)
        assert len(result.records) == 0

    def test_seed_reproducibility(self) -> None:
        source = _make_numeric_dataset(30)

        gen1 = SyntheticGenerator(n_patients=10, seed=123)
        result1 = gen1.generate(source)

        gen2 = SyntheticGenerator(n_patients=10, seed=123)
        result2 = gen2.generate(source)

        # Same seed should produce identical results
        for r1, r2 in zip(result1.records, result2.records, strict=True):
            assert r1.data == r2.data


class TestReidentificationValidator:
    def test_validate_passes(self) -> None:
        source = _make_numeric_dataset(50)
        gen = SyntheticGenerator(n_patients=20, seed=42)
        synthetic = gen.generate(source)

        validator = ReidentificationValidator(
            dcr_threshold=0.01,
            max_exact_match_rate=0.1,
        )
        report = validator.validate(source=source, synthetic=synthetic)
        assert isinstance(report, RiskReport)
        assert report.passed

    def test_empty_dataset(self) -> None:
        source = DeidentifiedDataset(dataset=ClinicalDataset())
        synthetic = ClinicalDataset()
        validator = ReidentificationValidator()
        report = validator.validate(source=source, synthetic=synthetic)
        assert report.passed

    def test_risk_report_fields(self) -> None:
        source = _make_numeric_dataset(30)
        gen = SyntheticGenerator(n_patients=15, seed=42)
        synthetic = gen.generate(source)

        validator = ReidentificationValidator()
        report = validator.validate(source=source, synthetic=synthetic)

        assert report.min_dcr >= 0
        assert report.mean_dcr >= 0
        assert 0 <= report.exact_match_rate <= 1

    def test_dcr_threshold_is_enforced(self) -> None:
        source = _make_numeric_dataset(20)
        gen = SyntheticGenerator(n_patients=10, seed=7)
        synthetic = gen.generate(source)

        validator = ReidentificationValidator(
            dcr_threshold=2.0,
            max_exact_match_rate=1.0,
        )
        with pytest.raises(ReidentificationRiskError, match="DCR below threshold"):
            validator.validate(source=source, synthetic=synthetic)


class TestUtilityEvaluation:
    def test_evaluate_utility(self) -> None:
        source = _make_numeric_dataset(50)
        gen = SyntheticGenerator(n_patients=30, seed=42)
        synthetic = gen.generate(source)

        report = evaluate_utility(synthetic, source)
        assert isinstance(report, UtilityReport)
        assert 0 <= report.fidelity <= 1
        assert 0 <= report.ml_utility <= 1

    def test_empty_datasets(self) -> None:
        empty_source = DeidentifiedDataset(dataset=ClinicalDataset())
        empty_synthetic = ClinicalDataset()
        report = evaluate_utility(empty_synthetic, empty_source)
        assert report.fidelity == 0.0
