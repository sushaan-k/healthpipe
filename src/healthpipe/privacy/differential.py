"""Differential privacy mechanisms.

Provides Laplace and Gaussian noise mechanisms for answering aggregate
queries over clinical datasets with formal privacy guarantees.

Each query consumes part of a shared ``PrivacyBudget``.  The module
also exposes higher-level helpers (``Count``, ``Mean``, ``Histogram``)
that mirror the public API shown in the spec.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
from healthpipe.privacy.budget import PrivacyBudget

logger = logging.getLogger(__name__)


class LaplaceMechanism:
    """Add Laplace noise calibrated to sensitivity / epsilon.

    The Laplace mechanism satisfies pure (epsilon, 0)-differential privacy.

    Args:
        sensitivity: The L1 sensitivity of the query function.
        epsilon: Privacy parameter for this query.
    """

    def __init__(self, sensitivity: float, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon

    def release(self, true_value: float) -> float:
        """Add noise and return the privatised value."""
        noise = np.random.laplace(0, self.scale)
        return float(true_value + noise)

    def release_array(self, values: np.ndarray) -> np.ndarray:
        """Add independent noise to every element of *values*."""
        noise = np.random.laplace(0, self.scale, size=values.shape)
        return values + noise


class GaussianMechanism:
    """Add Gaussian noise for (epsilon, delta)-differential privacy.

    Uses the analytic Gaussian mechanism calibration.

    Args:
        sensitivity: The L2 sensitivity of the query function.
        epsilon: Privacy parameter.
        delta: Probability of pure-DP failure.
    """

    def __init__(self, sensitivity: float, epsilon: float, delta: float = 1e-5) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        # sigma from the analytic Gaussian mechanism
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def release(self, true_value: float) -> float:
        """Add Gaussian noise and return the privatised value."""
        noise = np.random.normal(0, self.sigma)
        return float(true_value + noise)


# --- High-level query types ---------------------------------------------------


class Count(BaseModel):
    """Count query with optional ``group_by``.

    When ``group_by`` is set, ``private_stats`` returns a per-group
    mapping under the query result key.
    """

    field: str
    group_by: str = ""


class Mean(BaseModel):
    """Mean query with optional ``group_by``.

    When ``group_by`` is set, ``private_stats`` returns a per-group
    mapping under the query result key.
    """

    field: str
    group_by: str = ""
    clamp_range: tuple[float, float] = (0.0, 1000.0)


class Histogram(BaseModel):
    """Histogram query over a categorical field."""

    field: str
    bins: int = 10


# --- Result container ---------------------------------------------------------


class DPResult(BaseModel):
    """Container for differentially private query results.

    Attributes:
        results: Mapping of query descriptions to noised values.
        budget_remaining: Epsilon remaining after all queries.
        epsilon_spent: Total epsilon consumed by these queries.
    """

    results: dict[str, Any] = Field(default_factory=dict)
    budget_remaining: float = 0.0
    epsilon_spent: float = 0.0


# --- Public function ----------------------------------------------------------


def private_stats(
    data: DeidentifiedDataset,
    *,
    epsilon: float = 1.0,
    queries: list[Count | Mean | Histogram] | None = None,
    budget: PrivacyBudget | None = None,
) -> DPResult:
    """Run differentially private aggregate queries over *data*.

    Splits the total *epsilon* evenly across queries.  Grouped Count and
    Mean queries return per-group dictionaries keyed by the grouping value.

    Args:
        data: De-identified dataset to query.
        epsilon: Total privacy budget for this batch.
        queries: List of query specifications.
        budget: Optional shared budget tracker.  If provided, queries are
            charged against it.

    Returns:
        A ``DPResult`` containing the noised answers.
    """
    if queries is None:
        queries = []

    if budget is None:
        budget = PrivacyBudget(total_epsilon=epsilon)

    n_queries = max(len(queries), 1)
    per_query_eps = epsilon / n_queries

    results: dict[str, Any] = {}
    total_spent = 0.0
    records = data.records

    for query in queries:
        budget.spend(per_query_eps, description=_describe(query))
        total_spent += per_query_eps

        if isinstance(query, Count):
            mech = LaplaceMechanism(sensitivity=1.0, epsilon=per_query_eps)
            if query.group_by:
                grouped = _compute_grouped_count(records, query)
                results[f"count:{query.field}|group_by:{query.group_by}"] = {
                    group: max(0, round(mech.release(float(count))))
                    for group, count in grouped.items()
                }
            else:
                true_val = _compute_count(records, query)
                noised = mech.release(float(true_val))
                results[f"count:{query.field}"] = max(0, round(noised))

        elif isinstance(query, Mean):
            lo, hi = query.clamp_range
            sensitivity = (hi - lo) / max(len(records), 1)
            mech = LaplaceMechanism(sensitivity=sensitivity, epsilon=per_query_eps)
            if query.group_by:
                grouped_means = _compute_grouped_mean(records, query)
                results[f"mean:{query.field}|group_by:{query.group_by}"] = {
                    group: round(mech.release(true_val), 4)
                    for group, true_val in grouped_means.items()
                }
            else:
                mean_value = _compute_mean(records, query)
                noised_mean = mech.release(mean_value)
                results[f"mean:{query.field}"] = round(noised_mean, 4)

        elif isinstance(query, Histogram):
            counts = _compute_histogram(records, query)
            mech = LaplaceMechanism(sensitivity=1.0, epsilon=per_query_eps)
            noised_counts = {
                k: max(0, round(mech.release(float(v)))) for k, v in counts.items()
            }
            results[f"histogram:{query.field}"] = noised_counts

    return DPResult(
        results=results,
        budget_remaining=budget.epsilon_remaining,
        epsilon_spent=total_spent,
    )


# --- Internal computation helpers --------------------------------------------


def _compute_count(records: list[Any], query: Count) -> int:
    """Count records, optionally grouping."""
    return sum(1 for r in records if _record_matches_count_query(r, query.field))


def _compute_mean(records: list[Any], query: Mean) -> float:
    """Compute the mean of a numeric field, clamping to range."""
    values: list[float] = []
    lo, hi = query.clamp_range

    for r in records:
        val = _extract_field_value(r.data, query.field)
        if val is not None:
            try:
                v = float(val)
                v = max(lo, min(hi, v))
                values.append(v)
            except (ValueError, TypeError):
                continue

    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_grouped_count(records: list[Any], query: Count) -> dict[str, int]:
    """Count records within each group."""
    grouped: dict[str, int] = {}
    for r in records:
        if not _record_matches_count_query(r, query.field):
            continue
        key = _group_key(r.data, query.group_by)
        grouped[key] = grouped.get(key, 0) + 1
    return grouped


def _compute_grouped_mean(records: list[Any], query: Mean) -> dict[str, float]:
    """Compute the mean of a field within each group."""
    grouped_values: dict[str, list[float]] = {}
    lo, hi = query.clamp_range
    for r in records:
        value = _extract_field_value(r.data, query.field)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (ValueError, TypeError):
            continue
        numeric = max(lo, min(hi, numeric))
        key = _group_key(r.data, query.group_by)
        grouped_values.setdefault(key, []).append(numeric)

    grouped_means: dict[str, float] = {}
    for key, values in grouped_values.items():
        grouped_means[key] = sum(values) / len(values) if values else 0.0
    return grouped_means


def _compute_histogram(records: list[Any], query: Histogram) -> dict[str, int]:
    """Compute value counts for a categorical field."""
    counts: dict[str, int] = {}
    for r in records:
        val = _extract_field_value(r.data, query.field)
        if isinstance(val, str) and val:
            counts[val] = counts.get(val, 0) + 1
    return counts


def _record_matches_count_query(record: Any, field: str) -> bool:
    """Return True when *record* should be included in a Count query."""
    value = _extract_field_value(record.data, field)
    if value not in (None, "", [], {}):
        return True
    return field.strip().lower() in str(record.resource_type).lower()


def _extract_field_value(data: Any, path: str) -> Any:
    """Follow a dotted path through nested dictionaries."""
    if not path:
        return None

    value: Any = data
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _group_key(data: Any, path: str) -> str:
    """Resolve a grouping key from record data."""
    value = _extract_field_value(data, path)
    if value is None or value == "":
        return "ungrouped"
    return str(value)


def _describe(query: Count | Mean | Histogram) -> str:
    """Human-readable description for audit logging."""
    if isinstance(query, Count):
        return f"COUNT({query.field})"
    if isinstance(query, Mean):
        return f"MEAN({query.field})"
    return f"HISTOGRAM({query.field})"
