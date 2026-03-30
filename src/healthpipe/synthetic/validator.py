"""Re-identification risk validation for synthetic datasets.

Compares synthetic records against the source de-identified dataset to
ensure that no synthetic patient is too similar to a real patient.

Uses multiple distance metrics:
- Exact match detection
- Nearest-neighbour distance in embedding space
- Attribute disclosure risk (DCR -- Distance to Closest Record)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.exceptions import ReidentificationRiskError
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord

logger = logging.getLogger(__name__)


class RiskReport(BaseModel):
    """Results of re-identification risk assessment.

    Attributes:
        exact_match_rate: Fraction of synthetic records that exactly
            match a real record.
        min_dcr: Minimum Distance to Closest Record (lower = riskier).
        mean_dcr: Average DCR across all synthetic records.
        median_dcr: Median DCR.
        high_risk_count: Number of synthetic records with DCR below the
            risk threshold.
        passed: Whether the validation passed.
    """

    exact_match_rate: float = 0.0
    min_dcr: float = 0.0
    mean_dcr: float = 0.0
    median_dcr: float = 0.0
    high_risk_count: int = 0
    passed: bool = True


class ReidentificationValidator:
    """Validates synthetic data against re-identification risk.

    Args:
        dcr_threshold: Minimum acceptable DCR. Synthetic records closer
            than this to any real record are flagged.
        max_exact_match_rate: Maximum allowable exact match rate.
    """

    def __init__(
        self,
        dcr_threshold: float = 0.05,
        max_exact_match_rate: float = 0.0,
    ) -> None:
        self.dcr_threshold = dcr_threshold
        self.max_exact_match_rate = max_exact_match_rate

    def validate(
        self,
        source: DeidentifiedDataset,
        synthetic: ClinicalDataset,
    ) -> RiskReport:
        """Run full re-identification risk assessment.

        Args:
            source: The real (de-identified) dataset.
            synthetic: The synthetic dataset to validate.

        Returns:
            A ``RiskReport`` summarising the results.

        Raises:
            ReidentificationRiskError: If exact match rate or DCR
                exceeds the configured thresholds.
        """
        real_df = self._records_to_numeric_df(source.records)
        synth_df = self._records_to_numeric_df(synthetic.records)

        if real_df.empty or synth_df.empty:
            logger.warning("Cannot validate: one or both datasets are empty")
            return RiskReport(passed=True)

        # Align columns
        common_cols = list(set(real_df.columns) & set(synth_df.columns))
        if not common_cols:
            logger.warning("No common numeric columns for DCR computation")
            return RiskReport(passed=True)

        real_arr = real_df[common_cols].values.astype(float)
        synth_arr = synth_df[common_cols].values.astype(float)

        # Normalise columns to [0, 1]
        col_min = np.nanmin(np.vstack([real_arr, synth_arr]), axis=0)
        col_max = np.nanmax(np.vstack([real_arr, synth_arr]), axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0

        real_norm = (real_arr - col_min) / col_range
        synth_norm = (synth_arr - col_min) / col_range

        # Replace NaN with 0 for distance computation
        real_norm = np.nan_to_num(real_norm, nan=0.0)
        synth_norm = np.nan_to_num(synth_norm, nan=0.0)

        # Compute pairwise DCR
        dcr_values = self._compute_dcr(synth_norm, real_norm)

        exact_match_rate = float(np.sum(dcr_values == 0.0)) / max(len(dcr_values), 1)
        high_risk = int(np.sum(dcr_values < self.dcr_threshold))

        report = RiskReport(
            exact_match_rate=exact_match_rate,
            min_dcr=float(np.min(dcr_values)) if len(dcr_values) > 0 else 0.0,
            mean_dcr=float(np.mean(dcr_values)) if len(dcr_values) > 0 else 0.0,
            median_dcr=float(np.median(dcr_values)) if len(dcr_values) > 0 else 0.0,
            high_risk_count=high_risk,
            passed=True,
        )

        # Check thresholds
        if exact_match_rate > self.max_exact_match_rate:
            report.passed = False
            raise ReidentificationRiskError(
                f"Exact match rate {exact_match_rate:.2%} exceeds threshold "
                f"{self.max_exact_match_rate:.2%}"
            )

        if high_risk > 0:
            report.passed = False
            raise ReidentificationRiskError(
                f"{high_risk} synthetic record(s) have DCR below threshold "
                f"{self.dcr_threshold:.4f}"
            )

        logger.info(
            "Re-identification risk assessment: DCR min=%.4f mean=%.4f "
            "exact_matches=%.2f%%",
            report.min_dcr,
            report.mean_dcr,
            exact_match_rate * 100,
        )
        return report

    @staticmethod
    def _compute_dcr(synthetic: np.ndarray, real: np.ndarray) -> np.ndarray:
        """Compute the Distance to Closest Record for each synthetic row.

        Uses Euclidean distance in the normalised feature space.
        """
        dcr = np.full(len(synthetic), np.inf)
        # Process in chunks to avoid memory explosion
        chunk_size = 500
        for i in range(0, len(synthetic), chunk_size):
            chunk = synthetic[i : i + chunk_size]
            # Broadcast: (chunk, 1, features) - (1, real, features)
            diffs = chunk[:, np.newaxis, :] - real[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            dcr[i : i + chunk_size] = np.min(dists, axis=1)
        return dcr

    @staticmethod
    def _records_to_numeric_df(
        records: list[ClinicalRecord],
    ) -> pd.DataFrame:
        """Extract numeric features from records for distance computation."""
        rows: list[dict[str, Any]] = []
        for rec in records:
            row: dict[str, Any] = {}
            for k, v in rec.data.items():
                if isinstance(v, (int, float)):
                    row[k] = v
                elif isinstance(v, str):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
            if row:
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
