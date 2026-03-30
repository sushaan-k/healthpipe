"""Tests for audit logging, lineage tracking, and compliance reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthpipe.audit.compliance import (
    ComplianceReport,
    ComplianceReporter,
)
from healthpipe.audit.lineage import LineageTracker
from healthpipe.audit.logger import AuditEntry, AuditLog
from healthpipe.exceptions import LineageError


class TestAuditEntry:
    def test_original_hash(self) -> None:
        entry = AuditEntry(original="John Smith")
        assert entry.original_hash
        assert len(entry.original_hash) == 16
        assert entry.original_hash != "John Smith"

    def test_empty_original_hash(self) -> None:
        entry = AuditEntry(original="")
        assert entry.original_hash == ""

    def test_to_safe_dict(self) -> None:
        entry = AuditEntry(
            record_id="r1",
            action="PHI_REMOVED",
            original="123-45-6789",
            category="SSN",
        )
        safe = entry.to_safe_dict()
        assert safe["original"] != "123-45-6789"
        assert safe["category"] == "SSN"


class TestAuditLog:
    def test_add_and_count(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED"))
        log.add(AuditEntry(action="DATE_SHIFTED"))
        assert len(log) == 2

    def test_filter_by_layer(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(layer="NER", action="PHI_REMOVED"))
        log.add(AuditEntry(layer="PATTERN", action="PHI_REMOVED"))
        log.add(AuditEntry(layer="NER", action="PHI_REMOVED"))
        assert len(log.filter_by_layer("NER")) == 2
        assert len(log.filter_by_layer("PATTERN")) == 1

    def test_filter_by_action(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED"))
        log.add(AuditEntry(action="DATE_SHIFTED"))
        assert len(log.filter_by_action("PHI_REMOVED")) == 1

    def test_phi_removed_count(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED"))
        log.add(AuditEntry(action="PHI_REMOVED"))
        log.add(AuditEntry(action="DATE_SHIFTED"))
        assert log.phi_removed_count == 2

    def test_summary(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(layer="NER", action="PHI_REMOVED", category="PATIENT_NAME"))
        log.add(AuditEntry(layer="PATTERN", action="PHI_REMOVED", category="SSN"))
        summary = log.summary
        assert summary["total_entries"] == 2
        assert summary["phi_removed"] == 2
        assert summary["by_layer"]["NER"] == 1
        assert summary["by_category"]["SSN"] == 1

    def test_to_json(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(original="secret", action="PHI_REMOVED"))
        json_str = log.to_json(safe=True)
        parsed = json.loads(json_str)
        # The original value should be hashed
        assert parsed["entries"][0]["original"] != "secret"
        assert "summary" in parsed

    def test_save(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED"))
        result_path = log.save(tmp_path / "audit.json")
        assert result_path.exists()
        content = json.loads(result_path.read_text())
        assert len(content["entries"]) == 1

    def test_str_representation(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                category="SSN",
                original="123-45-6789",
                replacement="[SSN]",
                layer="PATTERN",
                confidence=1.0,
            )
        )
        output = str(log)
        assert "[PHI Removed]" in output

    def test_iteration(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="A"))
        log.add(AuditEntry(action="B"))
        actions = [e.action for e in log]
        assert actions == ["A", "B"]


class TestLineageTracker:
    def test_record_operation(self) -> None:
        tracker = LineageTracker()
        node = tracker.record_operation(
            record_id="r1",
            operation="ingest",
            metadata={"source": "csv"},
        )
        assert node.record_id == "r1"
        assert node.operation == "ingest"

    def test_get_history(self) -> None:
        tracker = LineageTracker()
        tracker.record_operation(record_id="r1", operation="ingest")
        tracker.record_operation(record_id="r1", operation="deidentify")
        history = tracker.get_history("r1")
        assert len(history) == 2
        assert history[0].operation == "ingest"
        assert history[1].operation == "deidentify"

    def test_auto_parent_linking(self) -> None:
        tracker = LineageTracker()
        n1 = tracker.record_operation(record_id="r1", operation="ingest")
        n2 = tracker.record_operation(record_id="r1", operation="deidentify")
        assert n1.node_id in n2.parent_ids

    def test_get_node(self) -> None:
        tracker = LineageTracker()
        node = tracker.record_operation(record_id="r1", operation="ingest")
        retrieved = tracker.get_node(node.node_id)
        assert retrieved.operation == "ingest"

    def test_get_node_not_found(self) -> None:
        tracker = LineageTracker()
        with pytest.raises(LineageError):
            tracker.get_node("nonexistent")

    def test_trace_to_source(self) -> None:
        tracker = LineageTracker()
        tracker.record_operation(record_id="r1", operation="ingest")
        tracker.record_operation(record_id="r1", operation="deidentify")
        n3 = tracker.record_operation(record_id="r1", operation="synthesize")

        path = tracker.trace_to_source(n3.node_id)
        assert len(path) == 3
        assert path[0].operation == "ingest"
        assert path[-1].operation == "synthesize"

    def test_all_records(self) -> None:
        tracker = LineageTracker()
        tracker.record_operation(record_id="r1", operation="ingest")
        tracker.record_operation(record_id="r2", operation="ingest")
        assert set(tracker.all_records) == {"r1", "r2"}

    def test_to_dict(self) -> None:
        tracker = LineageTracker()
        tracker.record_operation(record_id="r1", operation="ingest")
        data = tracker.to_dict()
        assert "nodes" in data
        assert "records" in data


class TestComplianceReporter:
    def test_generate_report(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                layer="NER",
                category="PATIENT_NAME",
            )
        )
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                layer="PATTERN",
                category="SSN",
            )
        )
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log, record_count=10)

        assert isinstance(report, ComplianceReport)
        assert report.dataset_record_count == 10
        assert report.phi_removed_count == 2

    def test_safe_harbor_identifiers_checked(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED", category="PATIENT_NAME"))
        log.add(AuditEntry(action="PHI_REMOVED", category="SSN"))
        log.add(AuditEntry(action="PHI_REMOVED", category="PHONE"))

        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)

        assert report.identifiers_addressed["Names"] is True
        assert report.identifiers_addressed["Social Security Numbers"] is True
        assert report.identifiers_addressed["Phone numbers"] is True

    def test_save_json(self, tmp_path: Path) -> None:
        report = ComplianceReport(
            dataset_record_count=5,
            phi_removed_count=3,
        )
        reporter = ComplianceReporter()
        path = reporter.save_json(report, tmp_path / "report.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["phi_removed_count"] == 3

    def test_to_markdown(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED", category="PATIENT_NAME"))
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log, record_count=5)
        md = reporter.to_markdown(report)

        assert "# HIPAA Compliance Report" in md
        assert "Safe Harbor Identifiers" in md
        assert "Audit Summary" in md

    def test_privacy_budget_in_report(self) -> None:
        log = AuditLog()
        reporter = ComplianceReporter()
        report = reporter.generate(
            audit_log=log,
            privacy_budget_used=0.5,
            privacy_budget_remaining=0.5,
        )
        md = reporter.to_markdown(report)
        assert "Differential Privacy" in md
