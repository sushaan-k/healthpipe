"""Synthetic clinical data generation.

Learns statistical distributions from de-identified clinical data and
generates new synthetic patients that preserve clinical correlations
(age-diagnosis, medication-lab values) without leaking real patient
information.

Supports two backends:
- ``"gaussian_copula"`` -- multivariate Gaussian copula (fast, lightweight)
- ``"ctgan"`` -- Conditional Tabular GAN (requires ``sdv`` optional dep)

When ``sdv`` is not installed, uses the built-in Gaussian copula sampler.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType

logger = logging.getLogger(__name__)


class SyntheticGenerator(BaseModel):
    """Configurable synthetic data generator.

    Args:
        n_patients: Number of synthetic patients to generate.
        method: ``"gaussian_copula"`` or ``"ctgan"``.
        preserve_correlations: Whether to model inter-column correlations.
        seed: Random seed for reproducibility.
    """

    n_patients: int = 1000
    method: str = "gaussian_copula"
    preserve_correlations: bool = True
    seed: int | None = None

    def generate(self, source: DeidentifiedDataset) -> ClinicalDataset:
        """Generate synthetic records from *source*.

        Returns:
            A new ``ClinicalDataset`` containing only synthetic records.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        df = self._records_to_dataframe(source.records)
        if df.empty:
            logger.warning("Source dataset is empty; returning empty synthetic set")
            return ClinicalDataset()

        if self.method == "ctgan":
            synthetic_df = self._generate_ctgan(df)
        else:
            synthetic_df = self._generate_copula(df)

        return self._dataframe_to_dataset(synthetic_df)

    # -- Gaussian copula (built-in) --------------------------------------------

    def _generate_copula(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data using a multivariate Gaussian copula.

        For each numeric column, fits a marginal distribution; then models
        the joint distribution via the empirical correlation matrix.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        synthetic_rows: list[dict[str, Any]] = []

        # Numeric columns: sample from multivariate normal with matched stats
        if numeric_cols:
            means = df[numeric_cols].mean().values
            if self.preserve_correlations and len(numeric_cols) > 1:
                cov = df[numeric_cols].cov().values
                # Ensure positive semi-definite
                cov = _nearest_psd(cov)
            else:
                cov = np.diag(df[numeric_cols].var().values)

            samples = np.random.multivariate_normal(means, cov, size=self.n_patients)

            for i in range(self.n_patients):
                row: dict[str, Any] = {}
                for j, col in enumerate(numeric_cols):
                    row[col] = float(samples[i, j])
                synthetic_rows.append(row)
        else:
            synthetic_rows = [{} for _ in range(self.n_patients)]

        # Categorical columns: sample from empirical distribution
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            values = value_counts.index.tolist()
            probs = value_counts.values.tolist()
            sampled = np.random.choice(values, size=self.n_patients, p=probs)
            for i, val in enumerate(sampled):
                synthetic_rows[i][col] = val

        return pd.DataFrame(synthetic_rows)

    # -- CTGAN (optional sdv dependency) ---------------------------------------

    def _generate_ctgan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic data using CTGAN from the sdv library."""
        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import (
                CTGANSynthesizer,
            )
        except ImportError:
            logger.warning(
                "sdv not installed; falling back to gaussian_copula. "
                "Install with: pip install sdv"
            )
            return self._generate_copula(df)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=False)
        synthesizer.fit(df)
        return synthesizer.sample(num_rows=self.n_patients)

    # -- Conversion helpers ----------------------------------------------------

    @staticmethod
    def _records_to_dataframe(records: list[ClinicalRecord]) -> pd.DataFrame:
        """Flatten clinical records into a tabular DataFrame."""
        rows: list[dict[str, Any]] = []
        for record in records:
            flat: dict[str, Any] = {
                "resource_type": record.resource_type.value,
            }
            flat.update(_flatten_dict(record.data, prefix=""))
            rows.append(flat)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @staticmethod
    def _dataframe_to_dataset(df: pd.DataFrame) -> ClinicalDataset:
        """Convert a synthetic DataFrame back to a ClinicalDataset."""
        records: list[ClinicalRecord] = []
        for _, row in df.iterrows():
            rtype_str = row.get("resource_type", "Patient")
            try:
                rtype = ResourceType(rtype_str)
            except ValueError:
                rtype = ResourceType.PATIENT

            data = {
                k: v
                for k, v in row.to_dict().items()
                if k != "resource_type" and pd.notna(v)
            }
            records.append(
                ClinicalRecord(
                    id=str(uuid.uuid4()),
                    resource_type=rtype,
                    data=data,
                    source_format="SYNTHETIC",
                )
            )
        return ClinicalDataset(records=records)


def _flatten_dict(d: dict[str, Any], prefix: str, sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            # Take the first element if it's a simple list
            if v and not isinstance(v[0], (dict, list)):
                items[new_key] = v[0]
        else:
            items[new_key] = v
    return items


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Find the nearest positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-8)
    return np.asarray(eigvecs @ np.diag(eigvals) @ eigvecs.T)


async def synthesize(
    data: DeidentifiedDataset,
    *,
    n_patients: int = 1000,
    method: str = "gaussian_copula",
    preserve_correlations: bool = True,
    validate: bool = True,
    seed: int | None = None,
) -> ClinicalDataset:
    """Convenience function: generate synthetic clinical data.

    Args:
        data: De-identified source dataset.
        n_patients: How many synthetic patients to generate.
        method: ``"gaussian_copula"`` or ``"ctgan"``.
        preserve_correlations: Keep inter-column correlations.
        validate: Run re-identification risk validation.
        seed: Random seed.

    Returns:
        A ``ClinicalDataset`` of synthetic records.
    """
    gen = SyntheticGenerator(
        n_patients=n_patients,
        method=method,
        preserve_correlations=preserve_correlations,
        seed=seed,
    )
    synthetic = gen.generate(data)

    if validate:
        from healthpipe.synthetic.validator import ReidentificationValidator

        validator = ReidentificationValidator()
        validator.validate(source=data, synthetic=synthetic)

    logger.info("Generated %d synthetic records", len(synthetic.records))
    return synthetic
