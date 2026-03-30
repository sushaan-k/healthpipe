"""Compliance report generation.

Produces structured reports that document how a dataset was processed,
what PHI was removed, which privacy mechanisms were applied, and whether
the output satisfies HIPAA Safe Harbor or Expert Determination criteria.

Reports can be exported as JSON (for programmatic consumption) or as
Markdown (for human review and PDF conversion).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from healthpipe.audit.logger import AuditLog

logger = logging.getLogger(__name__)

# The 18 HIPAA Safe Harbor identifiers (45 CFR 164.514(b)(2))
SAFE_HARBOR_IDENTIFIERS: list[str] = [
    "Names",
    "Geographic data (smaller than state)",
    "Dates (except year) related to an individual",
    "Phone numbers",
    "Fax numbers",
    "Email addresses",
    "Social Security Numbers",
    "Medical record numbers",
    "Health plan beneficiary numbers",
    "Account numbers",
    "Certificate/license numbers",
    "Vehicle identifiers and serial numbers",
    "Device identifiers and serial numbers",
    "Web URLs",
    "IP addresses",
    "Biometric identifiers",
    "Full-face photographs",
    "Any other unique identifying number or code",
]

# Map our internal categories to Safe Harbor identifier indices (1-based)
_CATEGORY_TO_IDENTIFIER: dict[str, int] = {
    "PATIENT_NAME": 1,
    "LOCATION": 2,
    "DATE": 3,
    "PHONE": 4,
    "FAX": 5,
    "EMAIL": 6,
    "SSN": 7,
    "MRN": 8,
    "HEALTH_PLAN": 9,
    "ACCOUNT_NUMBER": 10,
    "LICENSE": 11,
    "VEHICLE": 12,
    "DEVICE": 13,
    "URL": 14,
    "IP_ADDRESS": 15,
    "BIOMETRIC": 16,
    "PHOTOGRAPH": 17,
    "ZIP_CODE": 2,
    "FACILITY": 2,
    "ORGANIZATION": 2,
    "STREET_ADDRESS": 2,
}

# Identifiers that cannot be auto-detected and require manual review
# or specialized processing (image analysis, biometric sensor data, etc.)
MANUAL_REVIEW_IDENTIFIERS: dict[int, str] = {
    16: (
        "Biometric identifiers (fingerprints, retinal scans, voiceprints) "
        "cannot be detected via pattern matching. Requires specialized "
        "sensor data analysis or manual review."
    ),
    17: (
        "Full-face photographs and comparable images cannot be detected "
        "via pattern matching. Requires computer vision / image processing "
        "pipelines or manual review."
    ),
    18: (
        "Any other unique identifying number, characteristic, or code "
        "is a catch-all category. Use LLM verification (Layer 4) or "
        "manual review to detect identifiers not covered by other categories."
    ),
}


class ComplianceReport(BaseModel):
    """Structured HIPAA compliance report.

    Attributes:
        title: Report title.
        generated_at: When the report was created.
        method: De-identification method used.
        dataset_record_count: Number of records processed.
        phi_removed_count: Total PHI items removed.
        identifiers_addressed: Which of the 18 identifiers were handled.
        audit_summary: Aggregate audit statistics.
        safe_harbor_compliant: Whether the dataset meets Safe Harbor criteria.
        notes: Additional observations or warnings.
    """

    title: str = "HIPAA Compliance Report"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    method: str = "safe_harbor"
    dataset_record_count: int = 0
    phi_removed_count: int = 0
    identifiers_addressed: dict[str, bool] = Field(default_factory=dict)
    audit_summary: dict[str, Any] = Field(default_factory=dict)
    safe_harbor_compliant: bool = False
    notes: list[str] = Field(default_factory=list)
    privacy_budget_used: float | None = None
    privacy_budget_remaining: float | None = None


class ComplianceReporter:
    """Generates compliance reports from audit logs and pipeline metadata.

    Usage::

        reporter = ComplianceReporter()
        report = reporter.generate(
            audit_log=deidentified.audit_log,
            record_count=len(deidentified.records),
        )
        reporter.save_json(report, "reports/compliance.json")
    """

    def generate(
        self,
        audit_log: AuditLog,
        record_count: int = 0,
        method: str = "safe_harbor",
        privacy_budget_used: float | None = None,
        privacy_budget_remaining: float | None = None,
    ) -> ComplianceReport:
        """Build a compliance report from the audit log.

        Args:
            audit_log: The de-identification audit log.
            record_count: Number of records in the dataset.
            method: De-identification method used.
            privacy_budget_used: Epsilon spent (if DP was applied).
            privacy_budget_remaining: Epsilon remaining.

        Returns:
            A populated ``ComplianceReport``.
        """
        summary = audit_log.summary

        # Determine which Safe Harbor identifiers were addressed
        addressed = self._check_identifiers(audit_log)

        # Determine compliance
        notes: list[str] = []
        compliant = method == "safe_harbor"

        if method == "safe_harbor":
            unaddressed = [
                f"#{i + 1}: {name}"
                for i, (name, handled) in enumerate(
                    zip(
                        SAFE_HARBOR_IDENTIFIERS,
                        addressed.values(),
                        strict=True,
                    )
                )
                if not handled and (i + 1) not in MANUAL_REVIEW_IDENTIFIERS
            ]
            if unaddressed:
                notes.append(
                    "The following Safe Harbor identifiers were not "
                    f"detected in the dataset: {', '.join(unaddressed)}. "
                    "This may be expected if the data does not contain "
                    "these identifier types."
                )

            # Always note which identifiers require manual review
            for idx, description in MANUAL_REVIEW_IDENTIFIERS.items():
                name = SAFE_HARBOR_IDENTIFIERS[idx - 1]
                if not addressed.get(name, False):
                    notes.append(
                        f"#{idx} ({name}): MANUAL REVIEW REQUIRED. {description}"
                    )

        # Check LLM verification
        llm_entries = audit_log.filter_by_layer("LLM_VERIFY")
        if llm_entries:
            llm_clean = [e for e in llm_entries if e.action == "PHI_VERIFIED_CLEAN"]
            llm_detected = [e for e in llm_entries if e.action == "PHI_DETECTED_LLM"]
            llm_skipped = [
                e for e in llm_entries if e.action == "PHI_VERIFICATION_SKIPPED"
            ]
            llm_errors = [
                e for e in llm_entries if e.action == "PHI_VERIFICATION_ERROR"
            ]

            if llm_detected:
                notes.append(
                    f"LLM verification detected residual PHI in "
                    f"{len(llm_detected)}/{len(llm_entries)} records."
                )
                compliant = False
            elif llm_skipped or llm_errors:
                notes.append(
                    "LLM verification did not complete for all records: "
                    f"{len(llm_clean)} clean, {len(llm_skipped)} skipped, "
                    f"{len(llm_errors)} errored."
                )
                compliant = False
            else:
                notes.append(
                    f"LLM verification completed: {len(llm_clean)}/{len(llm_entries)} "
                    f"records verified clean."
                )
        else:
            notes.append(
                "LLM verification was not enabled. Consider enabling "
                "for additional assurance."
            )

        return ComplianceReport(
            method=method,
            dataset_record_count=record_count,
            phi_removed_count=summary.get("phi_removed", 0),
            identifiers_addressed=addressed,
            audit_summary=summary,
            safe_harbor_compliant=compliant,
            notes=notes,
            privacy_budget_used=privacy_budget_used,
            privacy_budget_remaining=privacy_budget_remaining,
        )

    @staticmethod
    def _check_identifiers(audit_log: AuditLog) -> dict[str, bool]:
        """Check which of the 18 Safe Harbor identifiers were addressed."""
        found_categories: set[str] = set()
        for entry in audit_log.entries:
            if entry.category:
                found_categories.add(entry.category)

        addressed: dict[str, bool] = {}
        for i, name in enumerate(SAFE_HARBOR_IDENTIFIERS):
            idx = i + 1
            cats_for_id = [
                cat
                for cat, mapped_idx in _CATEGORY_TO_IDENTIFIER.items()
                if mapped_idx == idx
            ]
            addressed[name] = any(c in found_categories for c in cats_for_id)

        return addressed

    @staticmethod
    def save_json(report: ComplianceReport, path: str | Path) -> Path:
        """Write the compliance report as JSON.

        Args:
            path: Destination file path.

        Returns:
            The resolved path that was written.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = report.model_dump(mode="json")
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Compliance report saved to %s", dest)
        return dest

    @staticmethod
    def to_markdown(report: ComplianceReport) -> str:
        """Render the compliance report as Markdown.

        Suitable for human review or conversion to PDF.
        """
        lines: list[str] = [
            f"# {report.title}",
            "",
            f"**Generated:** {report.generated_at.isoformat()}",
            f"**Method:** {report.method}",
            f"**Records processed:** {report.dataset_record_count}",
            f"**PHI items removed:** {report.phi_removed_count}",
            f"**Safe Harbor compliant:** "
            f"{'Yes' if report.safe_harbor_compliant else 'No'}",
            "",
        ]

        if report.privacy_budget_used is not None:
            lines.extend(
                [
                    "## Differential Privacy",
                    "",
                    f"- Epsilon spent: {report.privacy_budget_used:.4f}",
                    f"- Epsilon remaining: {report.privacy_budget_remaining or 0:.4f}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Safe Harbor Identifiers",
                "",
                "| # | Identifier | Addressed |",
                "|---|-----------|-----------|",
            ]
        )

        for i, (name, handled) in enumerate(report.identifiers_addressed.items()):
            status = "Yes" if handled else "N/A"
            lines.append(f"| {i + 1} | {name} | {status} |")

        lines.extend(["", "## Audit Summary", ""])
        for key, value in report.audit_summary.items():
            lines.append(f"- **{key}:** {value}")

        if report.notes:
            lines.extend(["", "## Notes", ""])
            for note in report.notes:
                lines.append(f"- {note}")

        lines.append("")
        return "\n".join(lines)
