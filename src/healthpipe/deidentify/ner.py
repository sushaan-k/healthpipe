"""Clinical Named Entity Recognition for PHI detection.

Uses spaCy's NER pipeline (with the ``en_core_sci_lg`` model when
available) to identify patient names, locations, organisations and other
entities that constitute Protected Health Information under HIPAA.

When spaCy is not installed, falls back to a lightweight rule-based
approach using common name lists and contextual clues.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Entity labels that map to PHI categories
_PHI_LABELS: dict[str, str] = {
    "PERSON": "PATIENT_NAME",
    "PER": "PATIENT_NAME",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "FACILITY",
    "ORG": "ORGANIZATION",
    "DATE": "DATE",
}

# Common title prefixes that precede patient names
_NAME_PREFIXES = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Patient|Pt)\b\.?\s+",
    re.IGNORECASE,
)

# Fallback: simple pattern that catches "Firstname Lastname" patterns
# after contextual cues like "Patient:", "Name:", etc.
_NAME_CONTEXT_RE = re.compile(
    r"(?:Patient|Name|Pt|Resident|Client)\s*:?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
)


@dataclass
class NEREntity:
    """A named entity detected by NER."""

    text: str
    label: str
    phi_category: str
    start: int
    end: int
    confidence: float
    detection_method: str = "ner"


@dataclass
class ClinicalNER:
    """Multi-strategy clinical NER for PHI detection.

    Attempts to use spaCy first; if unavailable, falls back to
    lightweight heuristics.

    Attributes:
        model_name: spaCy model to load (default ``en_core_web_sm``).
        use_fallback: If True, always use the rule-based fallback
            instead of spaCy.
    """

    model_name: str = "en_core_web_sm"
    use_fallback: bool = False
    _nlp: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Try to load the spaCy model."""
        if self.use_fallback:
            return
        try:
            import spacy

            self._nlp = spacy.load(self.model_name)
            logger.info("Loaded spaCy model: %s", self.model_name)
        except (ImportError, OSError):
            logger.info(
                "spaCy or model '%s' not available; using rule-based NER fallback",
                self.model_name,
            )
            self._nlp = None

    def extract(self, text: str) -> list[NEREntity]:
        """Detect PHI entities in *text*.

        Returns:
            Sorted list of ``NEREntity`` objects.
        """
        if self._nlp is not None:
            return self._spacy_extract(text)
        return self._fallback_extract(text)

    def redact(self, text: str) -> tuple[str, list[NEREntity]]:
        """Replace detected entities with category placeholders.

        Returns:
            Tuple of (redacted_text, entities).
        """
        entities = self.extract(text)
        if not entities:
            return text, []

        result = text
        for ent in reversed(entities):
            placeholder = f"[{ent.phi_category}]"
            result = result[: ent.start] + placeholder + result[ent.end :]
        return result, entities

    # -- spaCy strategy --------------------------------------------------------

    def _spacy_extract(self, text: str) -> list[NEREntity]:
        """Use spaCy NER pipeline."""
        doc = self._nlp(text)
        entities: list[NEREntity] = []
        for ent in doc.ents:
            phi_cat = _PHI_LABELS.get(ent.label_)
            if phi_cat is None:
                continue
            entities.append(
                NEREntity(
                    text=ent.text,
                    label=ent.label_,
                    phi_category=phi_cat,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,
                    detection_method="ner",
                )
            )
        entities.sort(key=lambda e: e.start)
        return entities

    # -- Fallback strategy -----------------------------------------------------

    def _fallback_extract(self, text: str) -> list[NEREntity]:
        """Rule-based fallback when spaCy is unavailable."""
        entities: list[NEREntity] = []

        # Detect names preceded by title prefixes
        for m in _NAME_PREFIXES.finditer(text):
            # Grab the next 1-3 capitalised words after the prefix
            remainder = text[m.end() :]
            name_match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", remainder)
            if name_match:
                name_text = name_match.group(1)
                start = m.end()
                entities.append(
                    NEREntity(
                        text=name_text,
                        label="PERSON",
                        phi_category="PATIENT_NAME",
                        start=start,
                        end=start + len(name_text),
                        confidence=0.70,
                        detection_method="context",
                    )
                )

        # Detect names from context cues
        for m in _NAME_CONTEXT_RE.finditer(text):
            name = m.group(1)
            entities.append(
                NEREntity(
                    text=name,
                    label="PERSON",
                    phi_category="PATIENT_NAME",
                    start=m.start(1),
                    end=m.end(1),
                    confidence=0.75,
                    detection_method="context",
                )
            )

        # De-duplicate overlapping spans (keep higher confidence)
        entities = self._deduplicate(entities)
        entities.sort(key=lambda e: e.start)
        return entities

    @staticmethod
    def _deduplicate(entities: list[NEREntity]) -> list[NEREntity]:
        """Remove overlapping entities, keeping the highest confidence."""
        if not entities:
            return []
        entities.sort(key=lambda e: (e.start, -e.confidence))
        result: list[NEREntity] = [entities[0]]
        for ent in entities[1:]:
            if ent.start >= result[-1].end:
                result.append(ent)
            elif ent.confidence > result[-1].confidence:
                result[-1] = ent
        return result
