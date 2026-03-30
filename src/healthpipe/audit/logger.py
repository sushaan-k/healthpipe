"""HIPAA-compliant audit logging.

Every de-identification action, privacy query, and data access is recorded
as an ``AuditEntry`` within an ``AuditLog``.  Logs are structured JSON by
default so they can be ingested by SIEM tools, indexed for compliance
reviews, and used to prove Safe Harbor adherence during audits.

Sensitive values (the original PHI) are hashed in the log rather than
stored in cleartext to prevent the audit trail itself from becoming a
data-breach vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AuditEntry(BaseModel):
    """A single auditable event in the de-identification pipeline.

    By default, the ``original`` PHI value is hashed with SHA-256 at
    construction time and stored as ``original_hash``.  The raw value
    is **not** retained unless ``store_raw=True`` is explicitly passed.

    Attributes:
        record_id: Identifier of the clinical record affected.
        action: What happened (e.g. ``PHI_REMOVED``, ``DATE_SHIFTED``).
        layer: Which pipeline layer produced this entry.
        original_hash: SHA-256 hash of the original PHI value.
        replacement: The replacement value or placeholder.
        category: HIPAA identifier category (e.g. ``PATIENT_NAME``).
        confidence: Detection confidence (0.0 - 1.0).
        timestamp: When the action occurred.
        operator: Identity of the operator or system component.
        store_raw: If True, keep the raw PHI in ``_original_raw``.
            Defaults to False for HIPAA compliance.
    """

    record_id: str = ""
    action: str = ""
    layer: str = ""
    original_hash: str = ""
    replacement: str = ""
    category: str = ""
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    operator: str = "healthpipe"
    store_raw: bool = Field(default=False, exclude=True)
    _original_raw: str | None = None

    def __init__(self, *, original: str = "", **kwargs: Any) -> None:
        """Hash the ``original`` field at construction time.

        The raw PHI is only retained if ``store_raw=True``.

        When reconstructing from a serialised dict that already has
        ``original_hash``, the ``original`` parameter (if it equals
        the existing hash) is treated as a pre-hashed value.
        """
        store_raw = kwargs.pop("store_raw", False)
        existing_hash = kwargs.get("original_hash", "")

        if existing_hash and not original:
            # Reconstructing from serialised data -- hash already present
            hashed = existing_hash
        elif existing_hash and original == existing_hash:
            # Round-trip: original was set to the hash in to_safe_dict
            hashed = existing_hash
        elif original:
            hashed = hashlib.sha256(original.encode()).hexdigest()[:16]
        else:
            hashed = ""

        kwargs["original_hash"] = hashed
        super().__init__(store_raw=store_raw, **kwargs)
        if store_raw:
            self._original_raw = original

    @property
    def original(self) -> str:
        """Return the raw PHI if stored, otherwise the hash.

        This property provides backwards-compatible access.  When
        ``store_raw=False`` (the default), it returns the hash.
        """
        if self._original_raw is not None:
            return self._original_raw
        return self.original_hash

    def to_safe_dict(self) -> dict[str, Any]:
        """Serialise without exposing the raw PHI value.

        The ``original`` key is set to the hash for backwards
        compatibility with consumers that expect this field name.
        """
        data = self.model_dump()
        data["original"] = self.original_hash
        data["timestamp"] = self.timestamp.isoformat()
        return data


class AuditLog(BaseModel):
    """Ordered collection of audit entries for a pipeline run.

    Provides iteration, filtering, persistence, and summary statistics.
    """

    entries: list[AuditEntry] = Field(default_factory=list)
    pipeline_run_id: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add(self, entry: AuditEntry) -> None:
        """Append an entry to the log."""
        self.entries.append(entry)

    def filter_by_layer(self, layer: str) -> list[AuditEntry]:
        """Return entries produced by a specific pipeline layer."""
        return [e for e in self.entries if e.layer == layer]

    def filter_by_action(self, action: str) -> list[AuditEntry]:
        """Return entries matching a specific action type."""
        return [e for e in self.entries if e.action == action]

    def filter_by_category(self, category: str) -> list[AuditEntry]:
        """Return entries for a specific PHI category."""
        return [e for e in self.entries if e.category == category]

    @property
    def phi_removed_count(self) -> int:
        """Total number of PHI items removed."""
        return sum(1 for e in self.entries if e.action == "PHI_REMOVED")

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregate statistics for the audit log."""
        by_layer: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_action: dict[str, int] = {}

        for entry in self.entries:
            by_layer[entry.layer] = by_layer.get(entry.layer, 0) + 1
            by_category[entry.category] = by_category.get(entry.category, 0) + 1
            by_action[entry.action] = by_action.get(entry.action, 0) + 1

        return {
            "total_entries": len(self.entries),
            "phi_removed": self.phi_removed_count,
            "by_layer": by_layer,
            "by_category": by_category,
            "by_action": by_action,
        }

    def to_json(self, safe: bool = True) -> str:
        """Serialise the log to a JSON string.

        Args:
            safe: If True, hash original PHI values to prevent leakage.
        """
        if safe:
            records = [e.to_safe_dict() for e in self.entries]
        else:
            records = [e.model_dump(mode="json") for e in self.entries]
        return json.dumps(
            {
                "pipeline_run_id": self.pipeline_run_id,
                "started_at": self.started_at.isoformat(),
                "summary": self.summary,
                "entries": records,
            },
            indent=2,
        )

    def save(self, path: str | Path, safe: bool = True) -> Path:
        """Write the audit log to a JSON file.

        Args:
            path: Destination file path.
            safe: If True, hash original PHI values.

        Returns:
            The resolved path that was written.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.to_json(safe=safe), encoding="utf-8")
        logger.info("Audit log saved to %s (%d entries)", dest, len(self.entries))
        return dest

    def __iter__(self) -> Iterator[AuditEntry]:  # type: ignore[override]
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __str__(self) -> str:
        lines: list[str] = []
        for entry in self.entries:
            if entry.action == "PHI_REMOVED":
                lines.append(
                    f"[PHI Removed] {entry.category} "
                    f'"{entry.original_hash}" -> "{entry.replacement}" '
                    f"({entry.layer}, confidence: {entry.confidence:.2f})"
                )
            elif entry.action == "DATE_SHIFTED":
                lines.append(f"[Date Shifted] {entry.replacement} ({entry.layer})")
            elif entry.action == "PHI_VERIFIED_CLEAN":
                lines.append(
                    "[PHI Verified] LLM found no additional PII in free-text fields"
                )
            elif entry.action == "PHI_DETECTED_LLM":
                lines.append(
                    f"[PHI Detected] LLM found: {entry.original_hash} "
                    f"({entry.category}, confidence: {entry.confidence:.2f})"
                )
            elif entry.action == "PHI_VERIFICATION_SKIPPED":
                lines.append(
                    "[PHI Verification Skipped] LLM verification was not run "
                    f"({entry.replacement})"
                )
            elif entry.action == "PHI_VERIFICATION_ERROR":
                lines.append(
                    "[PHI Verification Error] LLM verification failed "
                    f"({entry.replacement})"
                )
            else:
                lines.append(
                    f"[{entry.action}] {entry.category}: "
                    f"{entry.replacement} ({entry.layer})"
                )
        return "\n".join(lines)
