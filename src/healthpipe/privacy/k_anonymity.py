"""k-Anonymity and l-Diversity checking and enforcement.

k-Anonymity ensures that every combination of quasi-identifier values
(e.g. age, gender, ZIP prefix) appears in at least *k* records, making
it impossible to single out an individual.

l-Diversity additionally requires that the sensitive attribute (e.g.
diagnosis) has at least *l* distinct values within each equivalence
class, preventing attribute disclosure.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel

from healthpipe.exceptions import KAnonymityError

logger = logging.getLogger(__name__)


class QuasiIdentifierConfig(BaseModel):
    """Configuration for a single quasi-identifier column.

    Attributes:
        name: Column name in the dataset.
        generalization: How to generalise: ``"suppress"``, ``"range"``,
            ``"prefix"``, or ``None`` for no change.
        range_step: Step size when ``generalization="range"`` (e.g. 5 for
            age ranges like 20-24, 25-29).
        prefix_length: Number of characters to keep when
            ``generalization="prefix"`` (e.g. 3 for ZIP 941**).
    """

    name: str
    generalization: str | None = None
    range_step: int = 5
    prefix_length: int = 3


class KAnonymityChecker:
    """Checks and enforces k-anonymity on tabular data.

    Args:
        k: Minimum group size for quasi-identifier equivalence classes.
        quasi_identifiers: List of quasi-identifier configurations.
    """

    def __init__(
        self,
        k: int = 5,
        quasi_identifiers: list[QuasiIdentifierConfig] | None = None,
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2 for meaningful anonymity")
        self.k = k
        self.quasi_identifiers = quasi_identifiers or []

    def check(self, df: pd.DataFrame) -> bool:
        """Return True if *df* satisfies k-anonymity.

        Raises:
            KAnonymityError: With details about violating groups.
        """
        qi_cols = [qi.name for qi in self.quasi_identifiers if qi.name in df.columns]
        if not qi_cols:
            logger.warning("No quasi-identifier columns found in DataFrame")
            return True

        groups = df.groupby(qi_cols, dropna=False).size()
        violating = groups[groups < self.k]

        if len(violating) > 0:
            raise KAnonymityError(
                f"k-anonymity (k={self.k}) violated: "
                f"{len(violating)} equivalence classes have fewer than {self.k} records. "
                f"Smallest group size: {violating.min()}"
            )
        return True

    def enforce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generalise and suppress to enforce k-anonymity.

        Returns a new DataFrame that satisfies k-anonymity, potentially
        with some records suppressed (dropped) if generalisation alone
        is insufficient.
        """
        result = df.copy()

        # Step 1: Apply generalisations
        for qi in self.quasi_identifiers:
            if qi.name not in result.columns:
                continue
            result[qi.name] = self._generalise_column(result[qi.name], qi)

        # Step 2: Suppress groups that are still too small
        qi_cols = [
            qi.name for qi in self.quasi_identifiers if qi.name in result.columns
        ]
        if qi_cols:
            group_sizes = result.groupby(qi_cols, dropna=False)[qi_cols[0]].transform(
                "size"
            )
            mask = group_sizes >= self.k
            suppressed_count = int((~mask).sum())
            if suppressed_count > 0:
                logger.info(
                    "Suppressed %d records to achieve %d-anonymity",
                    suppressed_count,
                    self.k,
                )
            result = result[mask].reset_index(drop=True)

        return result

    @staticmethod
    def _generalise_column(
        series: pd.Series,
        config: QuasiIdentifierConfig,
    ) -> pd.Series:
        """Apply a generalisation strategy to a column."""
        if config.generalization == "range":
            return series.apply(lambda x: _range_generalise(x, config.range_step))
        if config.generalization == "prefix":
            return series.astype(str).str[: config.prefix_length]
        if config.generalization == "suppress":
            return pd.Series(["*"] * len(series), index=series.index)
        return series


class LDiversityChecker:
    """Checks l-diversity on tabular data.

    Args:
        l: Minimum number of distinct sensitive values per equivalence class.
        quasi_identifiers: Quasi-identifier column names.
        sensitive_column: The sensitive attribute column name.
    """

    def __init__(
        self,
        l: int = 2,  # noqa: E741
        quasi_identifiers: list[str] | None = None,
        sensitive_column: str = "diagnosis",
    ) -> None:
        self.l_value = l
        self.quasi_identifiers = quasi_identifiers or []
        self.sensitive_column = sensitive_column

    def check(self, df: pd.DataFrame) -> bool:
        """Return True if *df* satisfies l-diversity.

        Raises:
            KAnonymityError: With details about violating groups.
        """
        if self.sensitive_column not in df.columns:
            logger.warning(
                "Sensitive column '%s' not in DataFrame", self.sensitive_column
            )
            return True

        qi_cols = [c for c in self.quasi_identifiers if c in df.columns]
        if not qi_cols:
            return True

        groups = df.groupby(qi_cols, dropna=False)[self.sensitive_column]
        for name, group in groups:
            n_distinct = group.nunique()
            if n_distinct < self.l_value:
                raise KAnonymityError(
                    f"l-diversity (l={self.l_value}) violated for group {name}: "
                    f"only {n_distinct} distinct values for '{self.sensitive_column}'"
                )
        return True


def _range_generalise(value: Any, step: int) -> str:
    """Generalise a numeric value into a range string."""
    try:
        v = int(float(value))
        lo = (v // step) * step
        return f"{lo}-{lo + step - 1}"
    except (ValueError, TypeError):
        return str(value)
