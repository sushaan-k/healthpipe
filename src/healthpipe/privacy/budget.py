"""Privacy budget tracking.

Implements a composable privacy budget (epsilon) tracker that enforces
the total privacy loss across multiple queries.  Uses simple sequential
composition by default; advanced composition theorems (Renyi, etc.) can
be plugged in via subclassing.

A budget can be shared across multiple downstream consumers so that
the total disclosure is bounded even when different analysts run
separate queries against the same dataset.
"""

from __future__ import annotations

import logging
import threading
from _thread import LockType
from datetime import UTC, datetime

from pydantic import BaseModel, Field, PrivateAttr

from healthpipe.exceptions import BudgetExhaustedError

logger = logging.getLogger(__name__)


class BudgetEntry(BaseModel):
    """A single privacy-budget expenditure."""

    query_description: str
    epsilon_spent: float
    delta_spent: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PrivacyBudget(BaseModel):
    """Thread-safe differential-privacy budget tracker.

    Args:
        total_epsilon: Maximum allowable privacy loss.
        total_delta: Maximum allowable delta (for approximate DP).
        warn_threshold: Fraction of budget remaining that triggers a warning.
    """

    total_epsilon: float = 1.0
    total_delta: float = 1e-5
    warn_threshold: float = 0.2
    entries: list[BudgetEntry] = Field(default_factory=list)
    _lock: LockType = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._lock = threading.Lock()

    @property
    def epsilon_spent(self) -> float:
        """Total epsilon consumed so far."""
        return sum(e.epsilon_spent for e in self.entries)

    @property
    def delta_spent(self) -> float:
        """Total delta consumed so far."""
        return sum(e.delta_spent for e in self.entries)

    @property
    def epsilon_remaining(self) -> float:
        """Epsilon still available."""
        return max(0.0, self.total_epsilon - self.epsilon_spent)

    @property
    def delta_remaining(self) -> float:
        """Delta still available."""
        return max(0.0, self.total_delta - self.delta_spent)

    @property
    def fraction_remaining(self) -> float:
        """Fraction of the epsilon budget remaining (0.0 - 1.0)."""
        if self.total_epsilon <= 0:
            return 0.0
        return self.epsilon_remaining / self.total_epsilon

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        description: str = "",
    ) -> None:
        """Consume *epsilon* (and optionally *delta*) from the budget.

        Raises:
            BudgetExhaustedError: If the budget would be exceeded.
        """
        with self._lock:
            if epsilon > self.epsilon_remaining + 1e-12:
                raise BudgetExhaustedError(self.epsilon_remaining)
            if delta > self.delta_remaining + 1e-12:
                raise BudgetExhaustedError(self.epsilon_remaining)

            self.entries.append(
                BudgetEntry(
                    query_description=description,
                    epsilon_spent=epsilon,
                    delta_spent=delta,
                )
            )

            if self.fraction_remaining < self.warn_threshold:
                logger.warning(
                    "Privacy budget running low: %.1f%% remaining "
                    "(epsilon=%.4f of %.4f)",
                    self.fraction_remaining * 100,
                    self.epsilon_remaining,
                    self.total_epsilon,
                )

    def can_afford(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check whether *epsilon* can be spent without exceeding the budget."""
        return (
            epsilon <= self.epsilon_remaining + 1e-12
            and delta <= self.delta_remaining + 1e-12
        )

    def reset(self) -> None:
        """Clear all expenditure entries (for testing only)."""
        with self._lock:
            self.entries.clear()
