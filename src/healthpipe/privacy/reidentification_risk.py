"""Re-identification risk scoring for de-identified datasets.

Estimates how likely it is that an adversary could re-identify individuals
in a de-identified dataset by analysing the uniqueness of quasi-identifier
combinations.  This is distinct from the synthetic-vs-real DCR check in
``healthpipe.synthetic.validator``; it operates purely on the de-identified
output to flag records that remain at elevated risk.

Key metrics:
- **Uniqueness ratio**: fraction of records with a unique quasi-identifier
  combination (higher = riskier).
- **Prosecutor risk**: maximum probability of re-identifying any single
  individual (1 / minimum equivalence class size).
- **Journalist risk**: average probability across all equivalence classes.
- **Marketer risk**: overall probability computed as distinct-classes / N.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel

from healthpipe.ingest.schema import ClinicalRecord

logger = logging.getLogger(__name__)


class RiskScoreReport(BaseModel):
    """Quantified re-identification risk for a de-identified dataset.

    Attributes:
        uniqueness_ratio: Fraction of records whose quasi-identifier
            combination appears exactly once.
        prosecutor_risk: Worst-case re-identification probability
            (1 / smallest equivalence class).
        journalist_risk: Average per-class re-identification probability.
        marketer_risk: Overall re-identification probability
            (distinct classes / total records).
        min_class_size: Size of the smallest equivalence class.
        median_class_size: Median equivalence class size.
        risk_level: Human-readable risk label (``"low"``, ``"medium"``,
            or ``"high"``).
    """

    uniqueness_ratio: float = 0.0
    prosecutor_risk: float = 0.0
    journalist_risk: float = 0.0
    marketer_risk: float = 0.0
    min_class_size: int = 0
    median_class_size: float = 0.0
    risk_level: str = "low"


class ReidentificationRisk:
    """Compute re-identification risk scores for de-identified datasets.

    Analyses quasi-identifier columns to estimate how distinguishable
    individual records are after de-identification.

    Args:
        quasi_identifiers: Column names (keys in ``ClinicalRecord.data``)
            to treat as quasi-identifiers.
        high_risk_threshold: Uniqueness ratio above which the dataset is
            labelled ``"high"`` risk.
        medium_risk_threshold: Uniqueness ratio above which the dataset is
            labelled ``"medium"`` risk.
    """

    def __init__(
        self,
        quasi_identifiers: list[str],
        *,
        high_risk_threshold: float = 0.20,
        medium_risk_threshold: float = 0.05,
    ) -> None:
        if not quasi_identifiers:
            raise ValueError("At least one quasi-identifier column is required")
        self.quasi_identifiers = quasi_identifiers
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold

    def score(self, records: list[ClinicalRecord]) -> RiskScoreReport:
        """Analyse *records* and return a ``RiskScoreReport``.

        Records whose ``data`` dict does not contain any of the configured
        quasi-identifiers are silently skipped.
        """
        df = self._records_to_df(records)
        if df.empty:
            logger.warning("No quasi-identifier data found; returning zero-risk report")
            return RiskScoreReport()

        qi_cols = [c for c in self.quasi_identifiers if c in df.columns]
        if not qi_cols:
            logger.warning(
                "None of the configured quasi-identifiers (%s) found in data",
                self.quasi_identifiers,
            )
            return RiskScoreReport()

        group_sizes = df.groupby(qi_cols, dropna=False).size()
        n_records = len(df)
        n_classes = len(group_sizes)

        unique_count = int((group_sizes == 1).sum())
        uniqueness_ratio = unique_count / n_records if n_records > 0 else 0.0

        min_size = int(group_sizes.min())
        median_size = float(group_sizes.median())

        prosecutor_risk = 1.0 / min_size if min_size > 0 else 1.0
        journalist_risk = float((1.0 / group_sizes).mean()) if n_classes > 0 else 0.0
        marketer_risk = n_classes / n_records if n_records > 0 else 0.0

        if uniqueness_ratio >= self.high_risk_threshold:
            risk_level = "high"
        elif uniqueness_ratio >= self.medium_risk_threshold:
            risk_level = "medium"
        else:
            risk_level = "low"

        report = RiskScoreReport(
            uniqueness_ratio=round(uniqueness_ratio, 6),
            prosecutor_risk=round(prosecutor_risk, 6),
            journalist_risk=round(journalist_risk, 6),
            marketer_risk=round(marketer_risk, 6),
            min_class_size=min_size,
            median_class_size=round(median_size, 1),
            risk_level=risk_level,
        )

        logger.info(
            "Re-identification risk: level=%s uniqueness=%.2f%% "
            "prosecutor=%.4f journalist=%.4f marketer=%.4f",
            risk_level,
            uniqueness_ratio * 100,
            prosecutor_risk,
            journalist_risk,
            marketer_risk,
        )
        return report

    def _records_to_df(self, records: list[ClinicalRecord]) -> pd.DataFrame:
        """Extract quasi-identifier values from records into a DataFrame."""
        rows: list[dict[str, Any]] = []
        for rec in records:
            row: dict[str, Any] = {}
            for qi in self.quasi_identifiers:
                val = rec.data.get(qi)
                if val is not None:
                    row[qi] = val
            if row:
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
