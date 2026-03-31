"""Tests for the ReidentificationRisk scoring module."""

from __future__ import annotations

import pytest

from healthpipe.ingest.schema import ClinicalRecord, ResourceType
from healthpipe.privacy.reidentification_risk import (
    ReidentificationRisk,
    RiskScoreReport,
)


def _make_record(**data: object) -> ClinicalRecord:
    return ClinicalRecord(resource_type=ResourceType.PATIENT, data=dict(data))


class TestReidentificationRisk:
    def test_low_risk_identical_quasi_identifiers(self) -> None:
        """All records share the same QI combo -> uniqueness 0 -> low risk."""
        records = [
            _make_record(age="30", gender="M", diagnosis="flu"),
            _make_record(age="30", gender="M", diagnosis="cold"),
            _make_record(age="30", gender="M", diagnosis="asthma"),
        ]
        scorer = ReidentificationRisk(quasi_identifiers=["age", "gender"])
        report = scorer.score(records)

        assert report.risk_level == "low"
        assert report.uniqueness_ratio == 0.0
        assert report.min_class_size == 3
        assert report.prosecutor_risk == pytest.approx(1 / 3, abs=0.001)

    def test_high_risk_all_unique(self) -> None:
        """Every record has a unique QI combo -> high risk."""
        records = [
            _make_record(age="25", gender="M"),
            _make_record(age="30", gender="F"),
            _make_record(age="35", gender="M"),
            _make_record(age="40", gender="F"),
            _make_record(age="45", gender="M"),
        ]
        scorer = ReidentificationRisk(quasi_identifiers=["age", "gender"])
        report = scorer.score(records)

        assert report.risk_level == "high"
        assert report.uniqueness_ratio == 1.0
        assert report.min_class_size == 1
        assert report.prosecutor_risk == 1.0

    def test_medium_risk(self) -> None:
        """Some records unique, some grouped -> medium risk."""
        records = [
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            # one unique record -> 1/20 = 0.05 uniqueness
            _make_record(age="99", gender="F"),
            _make_record(age="30", gender="M"),
        ]
        scorer = ReidentificationRisk(
            quasi_identifiers=["age", "gender"],
            medium_risk_threshold=0.04,
            high_risk_threshold=0.20,
        )
        report = scorer.score(records)

        assert report.risk_level == "medium"
        assert 0.04 <= report.uniqueness_ratio < 0.20

    def test_empty_records(self) -> None:
        scorer = ReidentificationRisk(quasi_identifiers=["age"])
        report = scorer.score([])
        assert isinstance(report, RiskScoreReport)
        assert report.risk_level == "low"

    def test_missing_qi_columns(self) -> None:
        """Records that don't have the quasi-identifier keys."""
        records = [_make_record(diagnosis="flu")]
        scorer = ReidentificationRisk(quasi_identifiers=["age", "gender"])
        report = scorer.score(records)
        assert report.risk_level == "low"

    def test_requires_at_least_one_qi(self) -> None:
        with pytest.raises(ValueError, match="At least one quasi-identifier"):
            ReidentificationRisk(quasi_identifiers=[])

    def test_marketer_risk_computation(self) -> None:
        records = [
            _make_record(age="30", gender="M"),
            _make_record(age="30", gender="M"),
            _make_record(age="40", gender="F"),
            _make_record(age="40", gender="F"),
        ]
        scorer = ReidentificationRisk(quasi_identifiers=["age", "gender"])
        report = scorer.score(records)

        # 2 distinct classes, 4 records -> marketer_risk = 0.5
        assert report.marketer_risk == pytest.approx(0.5, abs=0.001)
        assert report.journalist_risk == pytest.approx(0.5, abs=0.001)
