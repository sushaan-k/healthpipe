"""Tests for the CLI interface using Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from healthpipe.cli import main


class TestCLIMain:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "healthpipe" in result.output
        assert "Privacy-preserving" in result.output

    def test_verbose_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["-v", "--help"])
        assert result.exit_code == 0


class TestCLIIngest:
    def test_ingest_csv(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,last_name,dob\nP001,Alice,Smith,1990-01-15\n"
        )
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(fpath)])
        assert result.exit_code == 0
        assert "Ingested" in result.output

    def test_ingest_csv_with_output(self, tmp_path: Path) -> None:
        csv_content = "patient_id,first_name,last_name\nP001,Alice,Smith\n"
        fpath = tmp_path / "patients.csv"
        fpath.write_text(csv_content)
        out_path = tmp_path / "output" / "result.json"

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(fpath), "-o", str(out_path)])
        assert result.exit_code == 0
        assert "saved to" in result.output
        assert out_path.exists()

    def test_ingest_fhir_json(self, tmp_path: Path) -> None:
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1"}},
            ],
        }
        fpath = tmp_path / "bundle.json"
        fpath.write_text(json.dumps(bundle))

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(fpath)])
        assert result.exit_code == 0
        assert "Ingested" in result.output

    def test_ingest_hl7_file(self, tmp_path: Path) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|1|P|2.5\r\n"
            "PID|||12345^^^MRN||Doe^Jane||19900101|F\r\n"
        )
        fpath = tmp_path / "message.hl7"
        fpath.write_text(msg_text)

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(fpath)])
        assert result.exit_code == 0
        assert "Ingested" in result.output

    def test_ingest_explicit_format(self, tmp_path: Path) -> None:
        csv_content = "patient_id,first_name\nP001,Alice\n"
        fpath = tmp_path / "data.txt"
        fpath.write_text(csv_content)

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(fpath), "--format", "csv"])
        assert result.exit_code == 0

    def test_ingest_unknown_extension(self, tmp_path: Path) -> None:
        fpath = tmp_path / "data.xyz"
        fpath.write_text("some content")

        runner = CliRunner()
        result = runner.invoke(main, ["ingest", str(fpath)])
        assert result.exit_code != 0

    def test_ingest_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["ingest", "/nonexistent/file.csv"])
        assert result.exit_code != 0


class TestCLIDeidentify:
    def test_deidentify_command(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "id": "p1",
                        "name": [{"family": "Smith"}],
                    },
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "input.json"
        input_path.write_text(dataset.model_dump_json(indent=2))
        output_path = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "deidentify",
                str(input_path),
                "-o",
                str(output_path),
                "--date-shift-salt",
                "test-cli-salt",
            ],
        )
        assert result.exit_code == 0
        assert "De-identified" in result.output
        assert output_path.exists()

    def test_deidentify_command_uses_default_salt(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "id": "p1",
                        "birthDate": "1985-03-15",
                    },
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "input.json"
        input_path.write_text(dataset.model_dump_json(indent=2))
        output_path = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "deidentify",
                str(input_path),
                "-o",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        assert "De-identified" in result.output

    def test_deidentify_with_no_date_shift(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={"resourceType": "Patient", "id": "p1"},
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "input.json"
        input_path.write_text(dataset.model_dump_json(indent=2))
        output_path = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "deidentify",
                str(input_path),
                "-o",
                str(output_path),
                "--no-date-shift",
            ],
        )
        assert result.exit_code == 0

    def test_deidentify_with_audit_log(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={"resourceType": "Patient", "id": "p1"},
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "input.json"
        input_path.write_text(dataset.model_dump_json(indent=2))
        output_path = tmp_path / "output.json"
        audit_path = tmp_path / "audit.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "deidentify",
                str(input_path),
                "-o",
                str(output_path),
                "--date-shift-salt",
                "test-cli-audit-salt",
                "--audit-log",
                str(audit_path),
            ],
        )
        assert result.exit_code == 0
        assert "Audit log saved" in result.output
        assert audit_path.exists()


class TestCLIScan:
    def test_scan_summary_reports_phi_breakdown(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="record-1",
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "telecom": [
                            {"system": "phone", "value": "555-123-4567"},
                        ],
                        "note": "Contact patient@example.org",
                    },
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "dataset.json"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(input_path)])

        assert result.exit_code == 0
        assert "PHI Dry-Run Scan" in result.output
        assert "Records scanned: 1" in result.output
        assert "PHI findings:" in result.output
        assert "EMAIL" in result.output

    def test_scan_json_hashes_original_values_by_default(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="record-1",
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "identifier": [{"value": "123-45-6789"}],
                    },
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "dataset.json"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(input_path), "--format", "json"])

        assert result.exit_code == 0
        assert "123-45-6789" not in result.output
        parsed = json.loads(result.output)
        assert parsed["total_phi_found"] >= 1
        assert "original_hash" in parsed["findings"][0]
        assert "original" not in parsed["findings"][0]

    def test_scan_can_write_markdown_and_fail_on_phi(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="record-1",
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "note": "Call 555-123-4567",
                    },
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "dataset.json"
        output_path = tmp_path / "scan.md"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "scan",
                str(input_path),
                "--format",
                "markdown",
                "-o",
                str(output_path),
                "--fail-on-phi",
            ],
        )

        assert result.exit_code == 2
        assert output_path.exists()
        rendered = output_path.read_text()
        assert "# healthpipe PHI Dry-Run Report" in rendered
        assert "555-123-4567" not in rendered

    def test_scan_writes_baseline_and_reports_only_new_phi(
        self, tmp_path: Path
    ) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        baseline_dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="record-1",
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "email": "patient@example.org",
                    },
                    source_format="TEST",
                )
            ]
        )
        baseline_input = tmp_path / "baseline.json"
        baseline_input.write_text(baseline_dataset.model_dump_json(indent=2))
        baseline_path = tmp_path / "phi-baseline.json"
        report_path = tmp_path / "scan.txt"

        runner = CliRunner()
        baseline_result = runner.invoke(
            main,
            [
                "scan",
                str(baseline_input),
                "-o",
                str(report_path),
                "--write-baseline",
                str(baseline_path),
            ],
        )

        assert baseline_result.exit_code == 0
        assert baseline_path.exists()
        baseline_payload = json.loads(baseline_path.read_text())
        assert "patient@example.org" not in baseline_path.read_text()
        assert baseline_payload["findings"][0]["fingerprint"]

        current_dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="record-1",
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "email": "patient@example.org",
                    },
                    source_format="TEST",
                ),
                ClinicalRecord(
                    id="record-2",
                    resource_type=ResourceType.PATIENT,
                    data={
                        "resourceType": "Patient",
                        "ssn": "123-45-6789",
                    },
                    source_format="TEST",
                ),
            ]
        )
        current_input = tmp_path / "current.json"
        current_input.write_text(current_dataset.model_dump_json(indent=2))

        result = runner.invoke(
            main,
            [
                "scan",
                str(current_input),
                "--format",
                "json",
                "--baseline",
                str(baseline_path),
                "--only-new",
                "--fail-on-new",
            ],
        )

        assert result.exit_code == 3
        parsed = json.loads(result.output)
        assert parsed["total_phi_found"] == 1
        assert parsed["findings"][0]["category"] == "SSN"
        assert "123-45-6789" not in result.output
        assert parsed["baseline_comparison"]["new_count"] == 1
        assert parsed["baseline_comparison"]["unchanged_count"] == 1

    def test_scan_new_only_requires_baseline(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import ClinicalDataset

        input_path = tmp_path / "dataset.json"
        input_path.write_text(ClinicalDataset().model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(main, ["scan", str(input_path), "--only-new"])

        assert result.exit_code != 0
        assert "--only-new requires --baseline" in result.output


class TestCLIValidate:
    def test_validate_summary_passes_valid_dataset(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="patient-record",
                    resource_type=ResourceType.PATIENT,
                    data={"resourceType": "Patient", "id": "p1"},
                    source_format="TEST",
                ),
                ClinicalRecord(
                    id="obs-record",
                    resource_type=ResourceType.OBSERVATION,
                    data={
                        "resourceType": "Observation",
                        "subject": {"reference": "Patient/p1"},
                    },
                    source_format="TEST",
                ),
            ]
        )
        input_path = tmp_path / "dataset.json"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(input_path)])

        assert result.exit_code == 0
        assert "Dataset Validation" in result.output
        assert "Status: passed" in result.output
        assert "Patient: 1" in result.output
        assert "Observation: 1" in result.output

    def test_validate_json_fails_on_errors(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="bad-record",
                    resource_type=ResourceType.OBSERVATION,
                    data={
                        "resourceType": "Observation",
                        "subject": {"reference": "Patient/missing"},
                        "note": "Patient Alice Smith called.",
                    },
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "dataset.json"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["validate", str(input_path), "--format", "json", "--fail-on", "error"],
        )

        assert result.exit_code == 5
        assert "Alice Smith" not in result.output
        parsed = json.loads(result.output)
        assert parsed["passed"] is False
        assert parsed["error_count"] == 1
        assert parsed["findings"][0]["code"] == "reference.unknown_patient"

    def test_validate_writes_markdown_and_fails_on_warning(
        self, tmp_path: Path
    ) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    id="empty-patient",
                    resource_type=ResourceType.PATIENT,
                    data={},
                    source_format="TEST",
                )
            ]
        )
        input_path = tmp_path / "dataset.json"
        output_path = tmp_path / "validation.md"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "validate",
                str(input_path),
                "--format",
                "markdown",
                "-o",
                str(output_path),
                "--fail-on",
                "warning",
            ],
        )

        assert result.exit_code == 5
        assert output_path.exists()
        rendered = output_path.read_text()
        assert "# healthpipe Dataset Validation Report" in rendered
        assert "record.empty_data" in rendered


class TestCLISynthesize:
    def test_synthesize_command(self, tmp_path: Path) -> None:
        from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        records = [
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={"glucose": 100.0, "hemoglobin": 14.0},
                source_format="TEST",
            )
            for _ in range(10)
        ]
        deid = DeidentifiedDataset(dataset=ClinicalDataset(records=records))
        input_path = tmp_path / "deid.json"
        input_path.write_text(deid.model_dump_json(indent=2))
        output_path = tmp_path / "synthetic.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "synthesize",
                str(input_path),
                "-o",
                str(output_path),
                "--n-patients",
                "5",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0
        assert "Generated" in result.output
        assert output_path.exists()

    def test_synthesize_with_method(self, tmp_path: Path) -> None:
        from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        records = [
            ClinicalRecord(
                resource_type=ResourceType.OBSERVATION,
                data={"val": float(i)},
                source_format="TEST",
            )
            for i in range(10)
        ]
        deid = DeidentifiedDataset(dataset=ClinicalDataset(records=records))
        input_path = tmp_path / "deid.json"
        input_path.write_text(deid.model_dump_json(indent=2))
        output_path = tmp_path / "synthetic.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "synthesize",
                str(input_path),
                "-o",
                str(output_path),
                "--method",
                "gaussian_copula",
                "--n-patients",
                "5",
            ],
        )
        assert result.exit_code == 0


class TestCLIRisk:
    def test_risk_summary_scores_nested_quasi_identifiers(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        dataset = ClinicalDataset(
            records=[
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "patient": {"birthYear": 1970},
                        "address": [{"postalCode": "02139"}],
                    },
                    source_format="TEST",
                ),
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "patient": {"birthYear": 1970},
                        "address": [{"postalCode": "02139"}],
                    },
                    source_format="TEST",
                ),
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data={
                        "patient": {"birthYear": 1988},
                        "address": [{"postalCode": "94110"}],
                    },
                    source_format="TEST",
                ),
            ]
        )
        input_path = tmp_path / "dataset.json"
        input_path.write_text(dataset.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "risk",
                str(input_path),
                "--quasi-id",
                "patient.birthYear",
                "--quasi-id",
                "address.postalCode",
            ],
        )

        assert result.exit_code == 0
        assert "Re-Identification Risk" in result.output
        assert "Risk level: high" in result.output
        assert "Records scored: 3/3" in result.output
        assert "Equivalence classes: 2" in result.output

    def test_risk_json_loads_deidentified_dataset_and_fail_gate(
        self, tmp_path: Path
    ) -> None:
        from healthpipe.deidentify.safe_harbor import DeidentifiedDataset
        from healthpipe.ingest.schema import (
            ClinicalDataset,
            ClinicalRecord,
            ResourceType,
        )

        deidentified = DeidentifiedDataset(
            dataset=ClinicalDataset(
                records=[
                    ClinicalRecord(
                        resource_type=ResourceType.PATIENT,
                        data={"age": "30", "gender": "F"},
                        source_format="TEST",
                    ),
                    ClinicalRecord(
                        resource_type=ResourceType.PATIENT,
                        data={"age": "40", "gender": "M"},
                        source_format="TEST",
                    ),
                ]
            )
        )
        input_path = tmp_path / "deidentified.json"
        input_path.write_text(deidentified.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "risk",
                str(input_path),
                "--quasi-id",
                "age",
                "--quasi-id",
                "gender",
                "--format",
                "json",
                "--fail-on",
                "high",
            ],
        )

        assert result.exit_code == 4
        parsed = json.loads(result.output)
        assert parsed["risk_level"] == "high"
        assert parsed["records_scored"] == 2
        assert parsed["quasi_identifiers"] == ["age", "gender"]

    def test_risk_rejects_inverted_thresholds(self, tmp_path: Path) -> None:
        from healthpipe.ingest.schema import ClinicalDataset

        input_path = tmp_path / "dataset.json"
        input_path.write_text(ClinicalDataset().model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "risk",
                str(input_path),
                "--quasi-id",
                "age",
                "--medium-risk-threshold",
                "0.4",
                "--high-risk-threshold",
                "0.2",
            ],
        )

        assert result.exit_code != 0
        assert "--medium-risk-threshold" in result.output


class TestCLIAudit:
    def test_audit_summary(self, tmp_path: Path) -> None:
        from healthpipe.audit.logger import AuditEntry, AuditLog

        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                layer="NER",
                category="PATIENT_NAME",
                original="John",
                replacement="[PATIENT_NAME]",
            )
        )
        audit_path = tmp_path / "audit.json"
        log.save(audit_path, safe=True)

        runner = CliRunner()
        result = runner.invoke(main, ["audit", str(audit_path), "--format", "summary"])
        assert result.exit_code == 0
        assert "Audit Log Summary" in result.output
        assert "Total entries" in result.output

    def test_audit_json(self, tmp_path: Path) -> None:
        from healthpipe.audit.logger import AuditEntry, AuditLog

        log = AuditLog()
        log.add(AuditEntry(action="PHI_REMOVED", category="SSN"))
        audit_path = tmp_path / "audit.json"
        log.save(audit_path, safe=True)

        runner = CliRunner()
        result = runner.invoke(main, ["audit", str(audit_path), "--format", "json"])
        assert result.exit_code == 0
        # Output should be valid JSON
        parsed = json.loads(result.output)
        assert "entries" in parsed

    def test_audit_markdown(self, tmp_path: Path) -> None:
        from healthpipe.audit.logger import AuditEntry, AuditLog

        log = AuditLog()
        log.add(
            AuditEntry(
                action="PHI_REMOVED",
                layer="NER",
                category="PATIENT_NAME",
            )
        )
        audit_path = tmp_path / "audit.json"
        log.save(audit_path, safe=True)

        runner = CliRunner()
        result = runner.invoke(main, ["audit", str(audit_path), "--format", "markdown"])
        assert result.exit_code == 0
        assert "HIPAA Compliance Report" in result.output
