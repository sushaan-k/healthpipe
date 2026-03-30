"""Tests for privacy edge cases.

Covers budget exhaustion edge cases, k-anonymity with edge group sizes,
l-diversity edge cases, and differential privacy helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.exceptions import BudgetExhaustedError, KAnonymityError
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType
from healthpipe.privacy.budget import PrivacyBudget
from healthpipe.privacy.differential import (
    Count,
    DPResult,
    GaussianMechanism,
    Histogram,
    Mean,
    _compute_histogram,
    _compute_mean,
    _describe,
    private_stats,
)
from healthpipe.privacy.k_anonymity import (
    KAnonymityChecker,
    LDiversityChecker,
    QuasiIdentifierConfig,
    _range_generalise,
)


class TestBudgetExhaustionEdgeCases:
    def test_exact_budget_spend(self) -> None:
        """Spending exactly the remaining budget should succeed."""
        budget = PrivacyBudget(total_epsilon=1.0)
        budget.spend(1.0)
        assert budget.epsilon_remaining == pytest.approx(0.0)

    def test_tiny_overspend_rejected(self) -> None:
        """Overspending even slightly should raise an error."""
        budget = PrivacyBudget(total_epsilon=1.0)
        budget.spend(0.999)
        # 0.002 more than remaining ~0.001, beyond the 1e-12 tolerance
        with pytest.raises(BudgetExhaustedError):
            budget.spend(0.002)

    def test_multiple_small_spends(self) -> None:
        """Many small spends should accumulate correctly."""
        budget = PrivacyBudget(total_epsilon=1.0)
        for _ in range(100):
            budget.spend(0.01)
        assert budget.epsilon_remaining == pytest.approx(0.0, abs=1e-10)

    def test_budget_exhausted_error_message(self) -> None:
        budget = PrivacyBudget(total_epsilon=0.5)
        budget.spend(0.5)
        with pytest.raises(BudgetExhaustedError) as exc_info:
            budget.spend(0.1)
        assert "exhausted" in str(exc_info.value)
        assert exc_info.value.epsilon_remaining == pytest.approx(0.0, abs=1e-10)

    def test_delta_exhaustion(self) -> None:
        """Delta budget should also be tracked and enforced."""
        budget = PrivacyBudget(total_epsilon=10.0, total_delta=1e-5)
        budget.spend(0.1, delta=1e-5)
        with pytest.raises(BudgetExhaustedError):
            budget.spend(0.1, delta=1e-5)

    def test_zero_total_epsilon_fraction(self) -> None:
        """Edge case: total_epsilon <= 0."""
        budget = PrivacyBudget(total_epsilon=0.0)
        assert budget.fraction_remaining == 0.0

    def test_warn_threshold(self) -> None:
        """Budget warning when below threshold."""
        budget = PrivacyBudget(total_epsilon=1.0, warn_threshold=0.5)
        # Spend 60% - should trigger warning
        budget.spend(0.6, description="big query")
        assert budget.fraction_remaining < 0.5

    def test_delta_remaining(self) -> None:
        budget = PrivacyBudget(total_delta=1e-4)
        budget.spend(0.1, delta=5e-5)
        assert budget.delta_remaining == pytest.approx(5e-5)

    def test_delta_spent(self) -> None:
        budget = PrivacyBudget(total_delta=1e-4)
        budget.spend(0.1, delta=3e-5)
        budget.spend(0.1, delta=2e-5)
        assert budget.delta_spent == pytest.approx(5e-5)


class TestKAnonymityEdgeCases:
    def test_k_equals_2_boundary(self) -> None:
        """k=2: groups of exactly 2 should pass."""
        df = pd.DataFrame(
            {
                "age": [25, 25, 30, 30],
                "gender": ["M", "M", "F", "F"],
            }
        )
        checker = KAnonymityChecker(
            k=2,
            quasi_identifiers=[
                QuasiIdentifierConfig(name="age"),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        assert checker.check(df)

    def test_singleton_group_violates(self) -> None:
        """A group of size 1 should violate k=2."""
        df = pd.DataFrame(
            {
                "age": [25, 30],
                "gender": ["M", "F"],
            }
        )
        checker = KAnonymityChecker(
            k=2,
            quasi_identifiers=[
                QuasiIdentifierConfig(name="age"),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        with pytest.raises(KAnonymityError, match="k-anonymity"):
            checker.check(df)

    def test_no_qi_columns_returns_true(self) -> None:
        """If no QI columns match the DataFrame, return True."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        checker = KAnonymityChecker(
            k=2,
            quasi_identifiers=[QuasiIdentifierConfig(name="nonexistent")],
        )
        assert checker.check(df)

    def test_enforce_with_suppression(self) -> None:
        """Enforcement should suppress records in too-small groups."""
        df = pd.DataFrame(
            {
                "age": [25, 25, 25, 30, 31, 32],
                "gender": ["M", "M", "M", "F", "F", "F"],
                "diagnosis": ["A", "B", "C", "D", "E", "F"],
            }
        )
        checker = KAnonymityChecker(
            k=3,
            quasi_identifiers=[
                QuasiIdentifierConfig(name="age"),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        enforced = checker.enforce(df)
        # The (25, M) group has 3 records; each (30, F), (31, F),
        # (32, F) has 1. After enforce, the singletons are suppressed.
        assert len(enforced) == 3

    def test_enforce_with_range_generalization(self) -> None:
        """Range generalization should merge nearby ages."""
        df = pd.DataFrame(
            {
                "age": [20, 21, 22, 23, 24, 30, 31, 32, 33, 34],
                "gender": [
                    "M",
                    "M",
                    "M",
                    "M",
                    "M",
                    "F",
                    "F",
                    "F",
                    "F",
                    "F",
                ],
            }
        )
        checker = KAnonymityChecker(
            k=5,
            quasi_identifiers=[
                QuasiIdentifierConfig(
                    name="age",
                    generalization="range",
                    range_step=5,
                ),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        enforced = checker.enforce(df)
        # All ages 20-24 -> "20-24", all 30-34 -> "30-34"
        assert len(enforced) == 10

    def test_enforce_with_prefix_generalization(self) -> None:
        df = pd.DataFrame(
            {
                "zip": [
                    "90210",
                    "90211",
                    "90212",
                    "94102",
                    "94103",
                    "94104",
                ],
                "gender": ["M", "M", "M", "F", "F", "F"],
            }
        )
        checker = KAnonymityChecker(
            k=3,
            quasi_identifiers=[
                QuasiIdentifierConfig(
                    name="zip",
                    generalization="prefix",
                    prefix_length=3,
                ),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        enforced = checker.enforce(df)
        # All should survive with prefix generalization
        assert len(enforced) == 6

    def test_enforce_with_suppress_generalization(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 25, 25],
            }
        )
        checker = KAnonymityChecker(
            k=3,
            quasi_identifiers=[
                QuasiIdentifierConfig(name="name", generalization="suppress"),
                QuasiIdentifierConfig(name="age"),
            ],
        )
        enforced = checker.enforce(df)
        # All names should be "*"
        assert all(enforced["name"] == "*")

    def test_enforce_no_qi_columns(self) -> None:
        """Enforcing with no matching QI columns returns the same df."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        checker = KAnonymityChecker(
            k=2,
            quasi_identifiers=[QuasiIdentifierConfig(name="missing")],
        )
        enforced = checker.enforce(df)
        assert len(enforced) == 3

    def test_range_generalise_function(self) -> None:
        assert _range_generalise(23, 5) == "20-24"
        assert _range_generalise(25, 5) == "25-29"
        assert _range_generalise(0, 10) == "0-9"

    def test_range_generalise_non_numeric(self) -> None:
        assert _range_generalise("abc", 5) == "abc"

    def test_range_generalise_float(self) -> None:
        assert _range_generalise(23.7, 5) == "20-24"


class TestLDiversityEdgeCases:
    def test_missing_sensitive_column(self) -> None:
        """If sensitive column doesn't exist, return True."""
        df = pd.DataFrame({"age": [25, 25], "gender": ["M", "M"]})
        checker = LDiversityChecker(
            l=2,
            quasi_identifiers=["age", "gender"],
            sensitive_column="diagnosis",
        )
        assert checker.check(df)

    def test_no_qi_columns(self) -> None:
        """If no QI columns match, return True."""
        df = pd.DataFrame(
            {"diagnosis": ["A", "B"]},
        )
        checker = LDiversityChecker(
            l=2,
            quasi_identifiers=["nonexistent"],
            sensitive_column="diagnosis",
        )
        assert checker.check(df)

    def test_l_equals_1(self) -> None:
        """l=1 should always pass (even with all same values)."""
        df = pd.DataFrame(
            {
                "age": [25, 25, 25],
                "gender": ["M", "M", "M"],
                "diagnosis": ["diabetes", "diabetes", "diabetes"],
            }
        )
        checker = LDiversityChecker(
            l=1,
            quasi_identifiers=["age", "gender"],
            sensitive_column="diagnosis",
        )
        assert checker.check(df)

    def test_exactly_l_distinct(self) -> None:
        """Exactly l distinct values should pass."""
        df = pd.DataFrame(
            {
                "age": [25, 25],
                "diagnosis": ["diabetes", "asthma"],
            }
        )
        checker = LDiversityChecker(
            l=2,
            quasi_identifiers=["age"],
            sensitive_column="diagnosis",
        )
        assert checker.check(df)


class TestPrivateStatsEdgeCases:
    def test_empty_queries(self) -> None:
        records = [ClinicalRecord(resource_type=ResourceType.PATIENT, data={})]
        deid = DeidentifiedDataset(dataset=ClinicalDataset(records=records))
        result = private_stats(deid, epsilon=1.0, queries=[])
        assert result.results == {}
        assert result.epsilon_spent == 0.0

    def test_mean_no_matching_records(self) -> None:
        records = [
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                data={"name": "test"},
            )
        ]
        deid = DeidentifiedDataset(dataset=ClinicalDataset(records=records))
        result = private_stats(
            deid,
            epsilon=1.0,
            queries=[Mean(field="nonexistent_field")],
        )
        # Mean of no values defaults to 0.0
        assert "mean:nonexistent_field" in result.results

    def test_multiple_queries_split_budget(self) -> None:
        records = [
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                data={"glucose": "100"},
            )
        ]
        deid = DeidentifiedDataset(dataset=ClinicalDataset(records=records))
        budget = PrivacyBudget(total_epsilon=2.0)
        result = private_stats(
            deid,
            epsilon=2.0,
            queries=[
                Count(field="patient"),
                Mean(field="glucose"),
            ],
            budget=budget,
        )
        # Each query gets epsilon/2 = 1.0
        assert result.epsilon_spent == pytest.approx(2.0)

    def test_describe_count(self) -> None:
        assert _describe(Count(field="patient")) == "COUNT(patient)"

    def test_describe_mean(self) -> None:
        assert _describe(Mean(field="glucose")) == "MEAN(glucose)"

    def test_describe_histogram(self) -> None:
        assert _describe(Histogram(field="diagnosis")) == "HISTOGRAM(diagnosis)"

    def test_compute_mean_with_nested_field(self) -> None:
        records = [
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={"valueQuantity": {"value": 100}},
            ),
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={"valueQuantity": {"value": 200}},
            ),
        ]
        mean = _compute_mean(
            records,
            Mean(field="valueQuantity.value", clamp_range=(0, 500)),
        )
        assert mean == pytest.approx(150.0)

    def test_compute_mean_non_numeric_values(self) -> None:
        records = [
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={"value": "not-a-number"},
            ),
        ]
        mean = _compute_mean(
            records,
            Mean(field="value", clamp_range=(0, 100)),
        )
        assert mean == 0.0

    def test_compute_histogram_empty(self) -> None:
        records = [
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={"other": 42},
            ),
        ]
        counts = _compute_histogram(records, Histogram(field="diagnosis"))
        assert counts == {}

    def test_gaussian_mechanism_sigma_calibration(self) -> None:
        """Higher epsilon should produce smaller sigma."""
        mech_low = GaussianMechanism(sensitivity=1.0, epsilon=0.1, delta=1e-5)
        mech_high = GaussianMechanism(sensitivity=1.0, epsilon=10.0, delta=1e-5)
        assert mech_low.sigma > mech_high.sigma

    def test_gaussian_invalid_delta_one(self) -> None:
        with pytest.raises(ValueError, match="delta must be in"):
            GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1.0)

    def test_dp_result_model(self) -> None:
        result = DPResult(
            results={"count:patient": 42},
            budget_remaining=0.5,
            epsilon_spent=0.5,
        )
        assert result.results["count:patient"] == 42
