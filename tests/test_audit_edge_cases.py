"""Tests for audit module edge cases: lineage DAG, compliance reports."""

from __future__ import annotations

import json
from pathlib import Path

from healthpipe.audit.compliance import (
    ComplianceReport,
    ComplianceReporter,
)
from healthpipe.audit.lineage import LineageNode, LineageTracker
from healthpipe.audit.logger import AuditEntry, AuditLog


class TestLineageDAGConstruction:
    def test_multiple_records_independent(self) -> None:
        """Different records should have independent lineage chains."""
        tracker = LineageTracker()
        tracker.record_operation(record_id="r1", operation="ingest")
        tracker.record_operation(record_id="r2", operation="ingest")
        tracker.record_operation(record_id="r1", operation="deidentify")
        tracker.record_operation(record_id="r2", operation="deidentify")

        h1 = tracker.get_history("r1")
        h2 = tracker.get_history("r2")
        assert len(h1) == 2
        assert len(h2) == 2
        # r1's deidentify should only have r1's ingest as parent
        assert h1[1].parent_ids == [h1[0].node_id]

    def test_explicit_parent_ids(self) -> None:
        """Setting parent_ids explicitly should override auto-linking."""
        tracker = LineageTracker()
        n1 = tracker.record_operation(record_id="r1", operation="ingest")
        n2 = tracker.record_operation(record_id="r2", operation="ingest")
        # Record r3 has both r1 and r2 as parents (merge operation)
        n3 = tracker.record_operation(
            record_id="r3",
            operation="merge",
            parent_ids=[n1.node_id, n2.node_id],
        )
        assert n1.node_id in n3.parent_ids
        assert n2.node_id in n3.parent_ids

    def test_get_parents(self) -> None:
        tracker = LineageTracker()
        n1 = tracker.record_operation(record_id="r1", operation="ingest")
        n2 = tracker.record_operation(record_id="r1", operation="deidentify")
        parents = tracker.get_parents(n2.node_id)
        assert len(parents) == 1
        assert parents[0].node_id == n1.node_id

    def test_get_parents_of_root_node(self) -> None:
        """Root node has no parents."""
        tracker = LineageTracker()
        n1 = tracker.record_operation(record_id="r1", operation="ingest")
        parents = tracker.get_parents(n1.node_id)
        assert len(parents) == 0

    def test_trace_to_source_deep_chain(self) -> None:
        """Trace through a long chain of operations."""
        tracker = LineageTracker()
        tracker.record_operation(record_id="r1", operation="ingest")
        tracker.record_operation(record_id="r1", operation="ner")
        tracker.record_operation(record_id="r1", operation="pattern_match")
        tracker.record_operation(record_id="r1", operation="date_shift")
        n5 = tracker.record_operation(record_id="r1", operation="llm_verify")

        path = tracker.trace_to_source(n5.node_id)
        assert len(path) == 5
        assert path[0].operation == "ingest"
        assert path[-1].operation == "llm_verify"

    def test_trace_to_source_avoids_cycles(self) -> None:
        """trace_to_source should handle visited-set correctly."""
        tracker = LineageTracker()
        n1 = tracker.record_operation(record_id="r1", operation="ingest")
        # Force a duplicate parent reference (shouldn't happen in practice)
        n2 = tracker.record_operation(
            record_id="r1",
            operation="deidentify",
            parent_ids=[n1.node_id, n1.node_id],
        )
        path = tracker.trace_to_source(n2.node_id)
        # Should deduplicate via visited set
        assert len(path) == 2

    def test_get_parents_with_missing_parent_id(self) -> None:
        """Parents that don't exist in tracker should be silently skipped."""
        tracker = LineageTracker()
        n1 = tracker.record_operation(
            record_id="r1",
            operation="ingest",
            parent_ids=["nonexistent-node"],
        )
        parents = tracker.get_parents(n1.node_id)
        assert len(parents) == 0

    def test_empty_history(self) -> None:
        tracker = LineageTracker()
        history = tracker.get_history("nonexistent")
        assert history == []

    def test_lineage_node_has_metadata(self) -> None:
        tracker = LineageTracker()
        node = tracker.record_operation(
            record_id="r1",
            operation="ingest",
            metadata={"source": "hl7", "encoding": "utf-8"},
            checksum_before="abc123",
            checksum_after="def456",
        )
        assert node.metadata["source"] == "hl7"
        assert node.checksum_before == "abc123"
        assert node.checksum_after == "def456"

    def test_lineage_node_defaults(self) -> None:
        node = LineageNode()
        assert node.node_id  # UUID should be auto-generated
        assert node.record_id == ""
        assert node.parent_ids == []


class TestComplianceReportEdgeCases:
    def test_report_with_llm_verification_entries(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_VERIFIED_CLEAN",
                layer="LLM_VERIFY",
                category="NONE",
            )
        )
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)
        assert any("LLM verification completed" in n for n in report.notes)

    def test_report_with_llm_skip_is_not_marked_clean(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_VERIFICATION_SKIPPED",
                layer="LLM_VERIFY",
                category="NONE",
                replacement="no API key",
            )
        )
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)
        assert report.safe_harbor_compliant is False
        assert any("did not complete" in n for n in report.notes)

    def test_report_without_llm_verification(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                layer="NER",
                category="PATIENT_NAME",
            )
        )
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)
        assert any("LLM verification was not enabled" in n for n in report.notes)

    def test_report_with_llm_detected_phi(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_DETECTED_LLM",
                layer="LLM_VERIFY",
                category="PATIENT_NAME",
                original="Dr. Smith",
            )
        )
        log.add(
            AuditEntry(
                action="PHI_VERIFIED_CLEAN",
                layer="LLM_VERIFY",
                category="NONE",
            )
        )
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)
        assert any("detected residual PHI" in n for n in report.notes)
        assert report.safe_harbor_compliant is False

    def test_all_mapped_identifiers_addressed(self) -> None:
        """A report where all mapped identifiers are covered.

        Identifier #18 ('Any other unique identifying number or code')
        has no category mapping in the source, which is expected -- it
        is a catch-all that does not correspond to a specific detection
        pattern.
        """
        log = AuditLog()
        all_categories = [
            "PATIENT_NAME",
            "LOCATION",
            "DATE",
            "PHONE",
            "FAX",
            "EMAIL",
            "SSN",
            "MRN",
            "HEALTH_PLAN",
            "ACCOUNT_NUMBER",
            "LICENSE",
            "VEHICLE",
            "DEVICE",
            "URL",
            "IP_ADDRESS",
            "BIOMETRIC",
            "PHOTOGRAPH",
            "ZIP_CODE",
        ]
        for cat in all_categories:
            log.add(
                AuditEntry(
                    action="PHI_REMOVED",
                    layer="PATTERN",
                    category=cat,
                )
            )
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log)
        # All identifiers that have category mappings should be addressed
        # Identifier #18 is a catch-all with no specific mapping
        for name, addressed in report.identifiers_addressed.items():
            if name == "Any other unique identifying number or code":
                continue
            assert addressed, f"Identifier '{name}' should be addressed"

    def test_expert_determination_method(self) -> None:
        """Non-safe-harbor method should not produce Safe Harbor notes."""
        log = AuditLog()
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log, method="expert_determination")
        # Should not have the "identifiers not detected" note
        assert not any("Safe Harbor identifiers" in n for n in report.notes)
        assert report.safe_harbor_compliant is False

    def test_markdown_without_privacy_budget(self) -> None:
        report = ComplianceReport(
            dataset_record_count=10,
            phi_removed_count=5,
        )
        md = ComplianceReporter.to_markdown(report)
        assert "Differential Privacy" not in md

    def test_markdown_with_notes(self) -> None:
        report = ComplianceReport(
            notes=["Important note 1", "Important note 2"],
        )
        md = ComplianceReporter.to_markdown(report)
        assert "Important note 1" in md
        assert "Notes" in md

    def test_empty_audit_log_report(self) -> None:
        log = AuditLog()
        reporter = ComplianceReporter()
        report = reporter.generate(audit_log=log, record_count=0)
        assert report.phi_removed_count == 0
        assert report.dataset_record_count == 0


class TestAuditLogEdgeCases:
    def test_str_date_shifted(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="DATE_SHIFTED",
                layer="DATE_SHIFT",
                replacement="offset=42 days",
            )
        )
        output = str(log)
        assert "[Date Shifted]" in output

    def test_str_phi_verified_clean(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_VERIFIED_CLEAN", layer="LLM_VERIFY"))
        output = str(log)
        assert "[PHI Verified]" in output

    def test_str_phi_detected_llm(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_DETECTED_LLM",
                layer="LLM_VERIFY",
                original="Dr. Jones",
                category="PATIENT_NAME",
                confidence=0.92,
            )
        )
        output = str(log)
        assert "[PHI Detected]" in output
        # original is now hashed, so "Dr. Jones" should NOT appear
        assert "Dr. Jones" not in output

    def test_str_other_action(self) -> None:
        log = AuditLog()
        log.add(
            AuditEntry(
                action="CUSTOM_ACTION",
                layer="CUSTOM",
                category="TEST",
                replacement="test_value",
            )
        )
        output = str(log)
        assert "[CUSTOM_ACTION]" in output

    def test_to_json_unsafe(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(original="secret_phi", action="PHI_REMOVED"))
        json_str = log.to_json(safe=False)
        parsed = json.loads(json_str)
        # Even in unsafe mode, the original is now hashed at construction
        # time. The original_hash field contains the hash.
        assert parsed["entries"][0]["original_hash"] != ""
        assert parsed["entries"][0]["original_hash"] != "secret_phi"

    def test_to_json_unsafe_with_store_raw(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(original="secret_phi", action="PHI_REMOVED", store_raw=True))
        # When store_raw=True, the raw value is accessible via .original
        entry = log.entries[0]
        assert entry.original == "secret_phi"

    def test_filter_by_category(self) -> None:
        log = AuditLog()
        log.add(AuditEntry(category="SSN", action="PHI_REMOVED"))
        log.add(AuditEntry(category="PHONE", action="PHI_REMOVED"))
        log.add(AuditEntry(category="SSN", action="PHI_REMOVED"))
        filtered = log.filter_by_category("SSN")
        assert len(filtered) == 2

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED"))
        nested = tmp_path / "deep" / "nested" / "audit.json"
        result_path = log.save(nested)
        assert result_path.exists()
