"""Tests for PHI detection confidence scores across detection methods."""

from __future__ import annotations

from healthpipe.deidentify.ner import ClinicalNER
from healthpipe.deidentify.patterns import DetectionMethod, PatternMatcher


class TestPatternMatcherConfidence:
    def setup_method(self) -> None:
        self.matcher = PatternMatcher()

    def test_pattern_matches_have_095_confidence(self) -> None:
        text = "SSN: 123-45-6789 and email test@example.com"
        matches = self.matcher.scan(text)
        assert len(matches) >= 2
        for m in matches:
            assert m.confidence == 0.95
            assert m.detection_method == DetectionMethod.PATTERN

    def test_redact_preserves_confidence(self) -> None:
        text = "Contact at test@example.com"
        _redacted, matches = self.matcher.redact(text)
        email_matches = [m for m in matches if m.category == "EMAIL"]
        assert len(email_matches) == 1
        assert email_matches[0].confidence == 0.95

    def test_redact_dict_preserves_confidence(self) -> None:
        data = {"ssn": "123-45-6789", "nested": {"email": "a@b.com"}}
        _redacted, matches = self.matcher.redact_dict(data)
        assert all(m.confidence == 0.95 for m in matches)
        assert all(m.detection_method == DetectionMethod.PATTERN for m in matches)


class TestNERConfidence:
    def test_context_detection_gets_lower_confidence(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Patient: John Smith was admitted."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1
        for name in names:
            # Context-based detections should have confidence <= 0.75
            assert name.confidence <= 0.75
            assert name.detection_method == "context"

    def test_title_prefix_detection_confidence(self) -> None:
        ner = ClinicalNER(use_fallback=True)
        text = "Dr. Jane Wilson ordered the lab work."
        entities = ner.extract(text)
        names = [e for e in entities if e.phi_category == "PATIENT_NAME"]
        assert len(names) >= 1
        assert names[0].confidence == 0.70
        assert names[0].detection_method == "context"


class TestDetectionMethodConstants:
    def test_default_confidence_values(self) -> None:
        assert DetectionMethod.DEFAULT_CONFIDENCE["pattern"] == 0.95
        assert DetectionMethod.DEFAULT_CONFIDENCE["ner"] == 0.85
        assert DetectionMethod.DEFAULT_CONFIDENCE["context"] == 0.70

    def test_method_constants(self) -> None:
        assert DetectionMethod.PATTERN == "pattern"
        assert DetectionMethod.NER == "ner"
        assert DetectionMethod.CONTEXT == "context"
