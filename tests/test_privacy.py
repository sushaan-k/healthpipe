"""Tests for privacy mechanisms: differential privacy, k-anonymity, budget."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.exceptions import BudgetExhaustedError, KAnonymityError
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType
from healthpipe.privacy.budget import PrivacyBudget
from healthpipe.privacy.differential import (
    Count,
    GaussianMechanism,
    Histogram,
    LaplaceMechanism,
    Mean,
    private_stats,
)
from healthpipe.privacy.k_anonymity import (
    KAnonymityChecker,
    LDiversityChecker,
    QuasiIdentifierConfig,
)


class TestLaplaceMechanism:
    def test_release_adds_noise(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        results = [mech.release(100.0) for _ in range(100)]
        # The results should not all be 100
        assert not all(r == 100.0 for r in results)
        # But the mean should be close to 100
        assert abs(np.mean(results) - 100.0) < 5.0

    def test_higher_epsilon_less_noise(self) -> None:
        mech_low = LaplaceMechanism(sensitivity=1.0, epsilon=0.1)
        mech_high = LaplaceMechanism(sensitivity=1.0, epsilon=10.0)
        assert mech_low.scale > mech_high.scale

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LaplaceMechanism(sensitivity=1.0, epsilon=0.0)

    def test_release_array(self) -> None:
        mech = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        values = np.array([10.0, 20.0, 30.0])
        noised = mech.release_array(values)
        assert noised.shape == values.shape
        assert not np.array_equal(noised, values)


class TestGaussianMechanism:
    def test_release_adds_noise(self) -> None:
        mech = GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=1e-5)
        results = [mech.release(50.0) for _ in range(100)]
        assert abs(np.mean(results) - 50.0) < 5.0

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=-1.0)

    def test_invalid_delta(self) -> None:
        with pytest.raises(ValueError):
            GaussianMechanism(sensitivity=1.0, epsilon=1.0, delta=0.0)


class TestPrivacyBudget:
    def test_initial_state(self) -> None:
        budget = PrivacyBudget(total_epsilon=5.0)
        assert budget.epsilon_remaining == 5.0
        assert budget.epsilon_spent == 0.0
        assert budget.fraction_remaining == 1.0

    def test_spend(self) -> None:
        budget = PrivacyBudget(total_epsilon=5.0)
        budget.spend(1.0, description="query1")
        assert budget.epsilon_remaining == pytest.approx(4.0)
        assert len(budget.entries) == 1

    def test_budget_exhaustion(self) -> None:
        budget = PrivacyBudget(total_epsilon=1.0)
        budget.spend(0.8)
        with pytest.raises(BudgetExhaustedError):
            budget.spend(0.5)

    def test_can_afford(self) -> None:
        budget = PrivacyBudget(total_epsilon=2.0)
        budget.spend(1.5)
        assert budget.can_afford(0.5)
        assert not budget.can_afford(1.0)

    def test_reset(self) -> None:
        budget = PrivacyBudget(total_epsilon=1.0)
        budget.spend(0.5)
        budget.reset()
        assert budget.epsilon_remaining == 1.0
        assert len(budget.entries) == 0


class TestPrivateStats:
    def setup_method(self) -> None:
        records = [
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                data={"diagnosis": "diabetes", "glucose": "110"},
            ),
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                data={"diagnosis": "hypertension", "glucose": "95"},
            ),
            ClinicalRecord(
                resource_type=ResourceType.PATIENT,
                data={"diagnosis": "diabetes", "glucose": "130"},
            ),
        ]
        dataset = ClinicalDataset(records=records)
        self.deid = DeidentifiedDataset(dataset=dataset)

    def test_count_query(self) -> None:
        result = private_stats(
            self.deid,
            epsilon=1.0,
            queries=[Count(field="patient")],
        )
        assert "count:patient" in result.results
        assert result.epsilon_spent > 0

    def test_mean_query(self) -> None:
        result = private_stats(
            self.deid,
            epsilon=1.0,
            queries=[Mean(field="glucose", clamp_range=(50.0, 300.0))],
        )
        assert "mean:glucose" in result.results

    def test_histogram_query(self) -> None:
        result = private_stats(
            self.deid,
            epsilon=1.0,
            queries=[Histogram(field="diagnosis")],
        )
        assert "histogram:diagnosis" in result.results

    def test_grouped_queries(self) -> None:
        result = private_stats(
            self.deid,
            epsilon=1.0,
            queries=[
                Count(field="patient", group_by="diagnosis"),
                Mean(field="glucose", group_by="diagnosis", clamp_range=(0.0, 300.0)),
            ],
        )
        count_key = "count:patient|group_by:diagnosis"
        mean_key = "mean:glucose|group_by:diagnosis"
        assert count_key in result.results
        assert mean_key in result.results
        assert set(result.results[count_key]) == {"diabetes", "hypertension"}
        assert set(result.results[mean_key]) == {"diabetes", "hypertension"}

    def test_budget_tracking(self) -> None:
        budget = PrivacyBudget(total_epsilon=2.0)
        result = private_stats(
            self.deid,
            epsilon=1.0,
            queries=[Count(field="patient")],
            budget=budget,
        )
        assert budget.epsilon_spent > 0
        assert result.budget_remaining == budget.epsilon_remaining


class TestKAnonymity:
    def test_satisfies_k(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25, 25, 25, 30, 30, 30],
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
        assert checker.check(df)

    def test_violates_k(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25, 25, 30],
                "gender": ["M", "F", "M"],
            }
        )
        checker = KAnonymityChecker(
            k=3,
            quasi_identifiers=[
                QuasiIdentifierConfig(name="age"),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        with pytest.raises(KAnonymityError):
            checker.check(df)

    def test_enforce(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25, 25, 25, 30, 30, 31],
                "gender": ["M", "M", "M", "F", "F", "F"],
            }
        )
        checker = KAnonymityChecker(
            k=3,
            quasi_identifiers=[
                QuasiIdentifierConfig(
                    name="age", generalization="range", range_step=10
                ),
                QuasiIdentifierConfig(name="gender"),
            ],
        )
        enforced = checker.enforce(df)
        assert len(enforced) >= 3

    def test_k_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 2"):
            KAnonymityChecker(k=1)


class TestLDiversity:
    def test_satisfies_l(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25, 25, 25],
                "gender": ["M", "M", "M"],
                "diagnosis": ["diabetes", "hypertension", "asthma"],
            }
        )
        checker = LDiversityChecker(
            l=3,
            quasi_identifiers=["age", "gender"],
            sensitive_column="diagnosis",
        )
        assert checker.check(df)

    def test_violates_l(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25, 25, 25],
                "gender": ["M", "M", "M"],
                "diagnosis": ["diabetes", "diabetes", "diabetes"],
            }
        )
        checker = LDiversityChecker(
            l=2,
            quasi_identifiers=["age", "gender"],
            sensitive_column="diagnosis",
        )
        with pytest.raises(KAnonymityError, match="l-diversity"):
            checker.check(df)
