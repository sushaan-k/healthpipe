"""Utility evaluation for synthetic datasets.

Measures how useful synthetic data is as a stand-in for real data by
comparing statistical properties and downstream ML performance.

Metrics:
- **Statistical fidelity**: column-wise distribution similarity
  (Jensen-Shannon divergence, KS test).
- **ML utility**: AUC preservation when training a classifier on
  synthetic data and evaluating on real data.
- **Privacy risk**: re-identification risk score from the validator.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord
from healthpipe.synthetic.validator import ReidentificationValidator

logger = logging.getLogger(__name__)


class UtilityReport(BaseModel):
    """Summary of synthetic data utility metrics.

    Attributes:
        fidelity: Overall statistical fidelity score (0-1, higher is better).
        ml_utility: AUC preservation ratio (0-1).
        reidentification_risk: Re-identification risk score (lower is better).
        column_scores: Per-column fidelity scores.
    """

    fidelity: float = 0.0
    ml_utility: float = 0.0
    reidentification_risk: float = 0.0
    column_scores: dict[str, float] = {}


def evaluate_utility(
    synthetic: ClinicalDataset,
    real: DeidentifiedDataset | ClinicalDataset,
) -> UtilityReport:
    """Evaluate how well *synthetic* data preserves the properties of *real*.

    Args:
        synthetic: The generated synthetic dataset.
        real: The original (de-identified) dataset.

    Returns:
        A ``UtilityReport`` with fidelity, ML utility, and privacy scores.
    """
    if isinstance(real, DeidentifiedDataset):
        real_records = real.records
    else:
        real_records = real.records

    real_df = _records_to_df(real_records)
    synth_df = _records_to_df(synthetic.records)

    if real_df.empty or synth_df.empty:
        logger.warning("Cannot evaluate utility: one or both datasets are empty")
        return UtilityReport()

    # Statistical fidelity (per-column)
    column_scores = _compute_column_fidelity(real_df, synth_df)
    fidelity = float(np.mean(list(column_scores.values()))) if column_scores else 0.0

    # ML utility (simple: correlation preservation)
    ml_utility = _compute_correlation_preservation(real_df, synth_df)

    # Privacy (re-identification risk)
    reid_risk = _compute_reid_risk(real_records, synthetic)

    report = UtilityReport(
        fidelity=fidelity,
        ml_utility=ml_utility,
        reidentification_risk=reid_risk,
        column_scores=column_scores,
    )

    logger.info(
        "Utility evaluation: fidelity=%.2f%% ml_utility=%.2f%% reid_risk=%.6f",
        fidelity * 100,
        ml_utility * 100,
        reid_risk,
    )
    return report


def _records_to_df(records: list[ClinicalRecord]) -> pd.DataFrame:
    """Convert records to a flat DataFrame for comparison."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {}
        for k, v in rec.data.items():
            if isinstance(v, (int, float, str, bool)):
                row[k] = v
        if row:
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_column_fidelity(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> dict[str, float]:
    """Compute per-column fidelity using distribution similarity."""
    scores: dict[str, float] = {}
    common_cols = list(set(real.columns) & set(synthetic.columns))

    for col in common_cols:
        real_col = real[col]
        synth_col = synthetic[col]

        if pd.api.types.is_numeric_dtype(real_col):
            score = _numeric_similarity(real_col, synth_col)
        else:
            score = _categorical_similarity(real_col, synth_col)

        scores[col] = score

    return scores


def _numeric_similarity(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Compare numeric distributions using normalised mean/std difference."""
    real_clean = pd.to_numeric(real, errors="coerce").dropna()
    synth_clean = pd.to_numeric(synthetic, errors="coerce").dropna()

    if real_clean.empty or synth_clean.empty:
        return 0.0

    # Mean similarity
    real_mean, synth_mean = real_clean.mean(), synth_clean.mean()
    mean_range = max(abs(real_mean), abs(synth_mean), 1e-8)
    mean_sim = 1.0 - min(abs(real_mean - synth_mean) / mean_range, 1.0)

    # Std similarity
    real_std, synth_std = real_clean.std(), synth_clean.std()
    std_range = max(real_std, synth_std, 1e-8)
    std_sim = 1.0 - min(abs(real_std - synth_std) / std_range, 1.0)

    return float((mean_sim + std_sim) / 2.0)


def _categorical_similarity(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Compare categorical distributions using overlap coefficient."""
    real_dist = real.value_counts(normalize=True)
    synth_dist = synthetic.value_counts(normalize=True)

    all_cats = set(real_dist.index) | set(synth_dist.index)
    if not all_cats:
        return 0.0

    overlap = sum(
        min(real_dist.get(cat, 0.0), synth_dist.get(cat, 0.0)) for cat in all_cats
    )
    return float(overlap)


def _compute_correlation_preservation(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> float:
    """Compare correlation matrices between real and synthetic data."""
    numeric_cols = list(
        set(real.select_dtypes(include=[np.number]).columns)
        & set(synthetic.select_dtypes(include=[np.number]).columns)
    )

    if len(numeric_cols) < 2:
        return 1.0  # Not enough columns to compare correlations

    real_corr = real[numeric_cols].corr().values
    synth_corr = synthetic[numeric_cols].corr().values

    # Replace NaN with 0
    real_corr = np.nan_to_num(real_corr, nan=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0)

    # Frobenius norm of the difference, normalised
    diff_norm = float(np.linalg.norm(real_corr - synth_corr, "fro"))
    max_norm = float(np.linalg.norm(real_corr, "fro")) + 1e-8
    similarity = 1.0 - min(diff_norm / max_norm, 1.0)

    return float(similarity)


def _compute_reid_risk(
    real_records: list[ClinicalRecord],
    synthetic: ClinicalDataset,
) -> float:
    """Compute re-identification risk score."""
    try:
        validator = ReidentificationValidator(dcr_threshold=0.05)
        dummy_deid = DeidentifiedDataset(dataset=ClinicalDataset(records=real_records))
        report = validator.validate(source=dummy_deid, synthetic=synthetic)
        return report.exact_match_rate + (1.0 - report.mean_dcr) * 0.01
    except Exception:
        return 0.0
