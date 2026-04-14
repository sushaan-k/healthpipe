"""Regex-based pattern matching for HIPAA identifiers.

Detects and replaces the following PHI patterns:
- Social Security Numbers (SSN)
- Phone numbers (US formats)
- Email addresses
- Medical Record Numbers (MRN)
- Account / insurance numbers
- Health plan beneficiary numbers
- Certificate / license numbers
- Vehicle identifiers (VIN)
- Device identifiers (UDI)
- IP addresses
- URLs
- US postal (ZIP) codes (reduces to 3-digit prefix per Safe Harbor)
- Street addresses (basic detection)
- Dates (handled separately by date_shift but detected here for audit)

Note on identifiers that cannot be auto-detected via regex:
- Biometric identifiers (#16): fingerprints, retinal scans, voiceprints,
  etc. require specialized sensor data analysis. Manual review required.
- Full-face photographs (#17): image content requires computer vision /
  image processing pipelines. Manual review required.
- Any other unique identifying number or code (#18): this is a catch-all
  category. Use LLM verification (Layer 4) or manual review to address.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar


class DetectionMethod:
    """Constants for how a PHI match was detected."""

    PATTERN = "pattern"
    NER = "ner"
    CONTEXT = "context"

    #: Default confidence scores per detection method.
    DEFAULT_CONFIDENCE: ClassVar[dict[str, float]] = {
        "pattern": 0.95,
        "ner": 0.85,
        "context": 0.70,
    }


@dataclass
class PHIMatch:
    """A single detected PHI occurrence."""

    category: str
    original: str
    replacement: str
    start: int
    end: int
    confidence: float = 0.95
    detection_method: str = DetectionMethod.PATTERN


@dataclass
class PatternMatcher:
    """Rule-based PHI detector using compiled regular expressions.

    Each pattern is associated with a HIPAA Safe Harbor identifier
    category.  Replacements use bracketed placeholders so downstream
    consumers can easily locate redacted spans.
    """

    extra_patterns: dict[str, str] = field(default_factory=dict)
    _compiled: list[tuple[str, re.Pattern[str], str]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        """Compile all patterns once at initialisation."""
        base: list[tuple[str, str, str]] = [
            # SSN: requires dashes (XXX-XX-XXXX) or context keywords.
            # Excludes invalid SSA ranges: 000, 666, 900-999 in first group.
            (
                "SSN",
                r"(?:"
                # Dashed format with SSA-valid first group
                r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
                r"|"
                # Context keyword followed by 9 digits (with or without dashes)
                r"(?i:(?:SSN|Social\s+Security|SS#)\s*:?\s*)"
                r"(?!000|666|9\d{2})\d{3}-?(?!00)\d{2}-?(?!0000)\d{4}"
                r")",
                "[SSN]",
            ),
            # US phone: (123) 456-7890, 123-456-7890, 123.456.7890
            (
                "PHONE",
                r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
                "[PHONE]",
            ),
            # Email
            (
                "EMAIL",
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "[EMAIL]",
            ),
            # MRN patterns: MR-12345, MRN12345, MRN: 12345
            (
                "MRN",
                r"\b(?:MR[N]?[-:\s]?\d{4,10})\b",
                "[MRN]",
            ),
            # Account / insurance numbers (generic numeric IDs 6+ digits)
            (
                "ACCOUNT_NUMBER",
                r"\b(?:Account|Acct|Policy|Insurance)\s*#?\s*:?\s*\d{6,}\b",
                "[ACCOUNT]",
            ),
            # Health plan beneficiary numbers (#9)
            (
                "HEALTH_PLAN",
                r"\b(?:(?:Health\s*Plan|Beneficiary|Member(?:\s*ID)?|Group|Subscriber)"
                r"\s*#?\s*:?\s*[A-Z0-9]{4,20})\b",
                "[HEALTH_PLAN]",
            ),
            # Certificate / license numbers (#11)
            (
                "LICENSE",
                r"\b(?:"
                r"(?:License|Licence|Certificate|Cert|DEA|NPI|UPIN)"
                r"\s*#?\s*:?\s*[A-Z0-9]{4,15}"
                r")\b",
                "[LICENSE]",
            ),
            # Vehicle identifiers -- VIN (17 alphanumeric, no I/O/Q) (#12)
            (
                "VEHICLE",
                r"\b(?:"
                r"(?:VIN|Vehicle)\s*#?\s*:?\s*"
                r"[A-HJ-NPR-Z0-9]{17}"
                r"|"
                r"[A-HJ-NPR-Z0-9]{17}"
                r"(?=\s|$|[.,;])"
                r")\b",
                "[VEHICLE]",
            ),
            # Device identifiers -- UDI / serial numbers (#13)
            (
                "DEVICE",
                r"\b(?:"
                r"(?:UDI|Device|Serial)\s*#?\s*:?\s*"
                r"[A-Z0-9()/-]{6,30}"
                r")\b",
                "[DEVICE]",
            ),
            # IP addresses
            (
                "IP_ADDRESS",
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                "[IP]",
            ),
            # URLs
            (
                "URL",
                r"https?://[^\s<>\"]+",
                "[URL]",
            ),
            # US ZIP codes -- reduce to 3-digit prefix
            (
                "ZIP_CODE",
                r"\b\d{5}(?:-\d{4})?\b",
                "[ZIP]",
            ),
            # Street addresses (basic detection) (#2 geographic)
            (
                "STREET_ADDRESS",
                r"\b\d+\s+\w+(?:\s+\w+)*\s+(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl|"
                r"Street|Avenue|Boulevard|Drive|Road|Lane|Court|Place)\b",
                "[ADDRESS]",
            ),
        ]
        for category, pattern, replacement in base:
            self._compiled.append(
                (category, re.compile(pattern, re.IGNORECASE), replacement)
            )
        for category, pattern in self.extra_patterns.items():
            self._compiled.append(
                (category, re.compile(pattern), f"[{category.upper()}]")
            )

    def scan(self, text: str) -> list[PHIMatch]:
        """Find all PHI matches in *text* without modifying it.

        Each match receives a confidence score reflecting the detection
        method.  Pattern (regex) matches default to 0.95 confidence.

        Returns:
            A list of ``PHIMatch`` objects sorted by position.
        """
        matches: list[PHIMatch] = []
        for category, regex, replacement in self._compiled:
            for m in regex.finditer(text):
                matches.append(
                    PHIMatch(
                        category=category,
                        original=m.group(),
                        replacement=replacement,
                        start=m.start(),
                        end=m.end(),
                        confidence=DetectionMethod.DEFAULT_CONFIDENCE[
                            DetectionMethod.PATTERN
                        ],
                        detection_method=DetectionMethod.PATTERN,
                    )
                )
        matches.sort(key=lambda m: m.start)
        return matches

    def redact(self, text: str) -> tuple[str, list[PHIMatch]]:
        """Scan *text* and replace all PHI matches with placeholders.

        Returns:
            A tuple of (redacted_text, list_of_matches).
        """
        matches = self.scan(text)
        if not matches:
            return text, []

        # Replace from right to left so positions stay valid
        result = text
        for m in reversed(matches):
            result = result[: m.start] + m.replacement + result[m.end :]
        return result, matches

    def redact_dict(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, Any], list[PHIMatch]]:
        """Recursively redact string values in a nested dict.

        Returns:
            A tuple of (redacted_dict, all_matches).
        """
        all_matches: list[PHIMatch] = []
        redacted = self._walk_dict(data, all_matches)
        return redacted, all_matches

    def _walk_dict(self, obj: Any, matches: list[PHIMatch]) -> Any:
        """Recursively walk a nested structure and redact strings."""
        if isinstance(obj, str):
            redacted, found = self.redact(obj)
            matches.extend(found)
            return redacted
        if isinstance(obj, dict):
            return {k: self._walk_dict(v, matches) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk_dict(item, matches) for item in obj]
        return obj
