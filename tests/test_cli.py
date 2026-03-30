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
