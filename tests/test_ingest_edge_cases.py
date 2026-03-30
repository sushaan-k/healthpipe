"""Tests for ingest edge cases.

Covers HL7v2 parsing edge cases, CSV mapping with missing columns,
FHIR validation, and PDF OCR adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from healthpipe.exceptions import (
    IngestError,
)
from healthpipe.ingest.csv_mapper import CSVSource, _set_nested
from healthpipe.ingest.fhir import FHIRAuth, FHIRSource
from healthpipe.ingest.hl7v2 import (
    HL7Message,
    HL7v2Source,
    _parse_hl7_datetime,
)
from healthpipe.ingest.pdf_ocr import PDFSource
from healthpipe.ingest.schema import ResourceType


class TestHL7v2ParsingEdgeCases:
    def test_empty_message(self) -> None:
        msg = HL7Message("")
        assert msg.segments == {}

    def test_msh_only(self) -> None:
        msg = HL7Message("MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|1|P|2.5")
        assert "MSH" in msg.segments
        assert msg.message_type == "ADT^A01"

    def test_missing_pid_segment(self) -> None:
        """Message without PID should produce no patient record."""
        msg_text = "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ORU^R01|1|P|2.5\r\n"
        msg = HL7Message(msg_text)
        assert "PID" not in msg.segments

    def test_get_field_out_of_range(self) -> None:
        msg = HL7Message("MSH|^~\\&|SRC\rPID|||12345")
        # Request a field beyond the segment length
        assert msg.get_field("PID", 99) == ""

    def test_get_field_negative_index(self) -> None:
        msg = HL7Message("MSH|^~\\&|SRC\rPID|||12345")
        # MSH adjusts index by -1, so requesting 0 gives -1
        assert msg.get_field("MSH", 0) == ""

    def test_get_field_nonexistent_segment(self) -> None:
        msg = HL7Message("MSH|^~\\&|SRC")
        assert msg.get_field("ZZZ", 1) == ""

    def test_get_field_nonexistent_occurrence(self) -> None:
        msg = HL7Message("MSH|^~\\&|SRC\rOBX|1|NM|code||100")
        assert msg.get_field("OBX", 1, occurrence=5) == ""

    def test_get_components(self) -> None:
        msg = HL7Message("MSH|^~\\&|SRC\rPID|||12345||Smith^John^Michael")
        components = msg.get_components("PID", 5)
        assert components[0] == "Smith"
        assert components[1] == "John"
        assert components[2] == "Michael"

    def test_multiple_obx_segments(self) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ORU^R01|1|P|2.5\r\n"
            "PID|||12345||Doe^Jane||19900101|F\r\n"
            "OBX|1|NM|2345-7^Glucose^LOINC||110|mg/dL\r\n"
            "OBX|2|NM|718-7^Hemoglobin^LOINC||14.5|g/dL\r\n"
        )
        msg = HL7Message(msg_text)
        assert len(msg.segments.get("OBX", [])) == 2

    def test_dg1_segments(self) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|1|P|2.5\r\n"
            "PID|||12345||Doe^Jane||19900101|F\r\n"
            "DG1|1||E11.9^Type 2 Diabetes^ICD-10-CM|Diabetes|W\r\n"
        )
        msg = HL7Message(msg_text)
        assert "DG1" in msg.segments

    @pytest.mark.asyncio
    async def test_ingest_with_dg1(self, tmp_path: Path) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|1|P|2.5\r\n"
            "PID|||12345||Smith^John||19850315|M\r\n"
            "DG1|1||E11.9^Type 2 Diabetes^ICD-10-CM|Diabetes|W\r\n"
        )
        fpath = tmp_path / "msg.hl7"
        fpath.write_text(msg_text)

        source = HL7v2Source(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() == 1
        assert dataset.conditions.count() == 1

    @pytest.mark.asyncio
    async def test_ingest_with_obx(self, tmp_path: Path) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ORU^R01|1|P|2.5\r\n"
            "PID|||12345||Doe^Jane||19900101|F\r\n"
            "OBX|1|NM|2345-7^Glucose^LOINC||110|mg/dL|70-100|H||F\r\n"
        )
        fpath = tmp_path / "msg.hl7"
        fpath.write_text(msg_text)

        source = HL7v2Source(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() == 1
        assert dataset.observations.count() == 1

    @pytest.mark.asyncio
    async def test_ingest_multiple_messages_in_file(self, tmp_path: Path) -> None:
        msg_text = (
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|1|P|2.5\r\n"
            "PID|||111||Alice^Smith||19900101|F\r\n"
            "MSH|^~\\&|SRC|FAC|DST|FAC|20250315||ADT^A01|2|P|2.5\r\n"
            "PID|||222||Bob^Jones||19850601|M\r\n"
        )
        fpath = tmp_path / "multi.hl7"
        fpath.write_text(msg_text)

        source = HL7v2Source(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() == 2

    @pytest.mark.asyncio
    async def test_ingest_no_matching_files(self, tmp_path: Path) -> None:
        """Glob pattern that matches nothing should produce empty dataset."""
        source = HL7v2Source(path=str(tmp_path / "*.nonexistent"))
        dataset = await source.ingest()
        assert len(dataset.records) == 0

    def test_parse_hl7_datetime_empty(self) -> None:
        assert _parse_hl7_datetime("") is None

    def test_parse_hl7_datetime_invalid(self) -> None:
        assert _parse_hl7_datetime("invalid") is None

    def test_parse_hl7_datetime_whitespace(self) -> None:
        assert _parse_hl7_datetime("   ") is None

    def test_gender_mapping(self) -> None:
        msg_text = "MSH|^~\\&|SRC|FAC\rPID|||12345||Doe^Jane||19900101|F"
        msg = HL7Message(msg_text)
        patient = HL7v2Source._extract_patient(msg)
        assert patient["gender"] == "female"

    def test_gender_mapping_other(self) -> None:
        msg_text = "MSH|^~\\&|SRC|FAC\rPID|||12345||Doe^Pat||19900101|O"
        msg = HL7Message(msg_text)
        patient = HL7v2Source._extract_patient(msg)
        assert patient["gender"] == "other"

    def test_gender_mapping_unknown(self) -> None:
        msg_text = "MSH|^~\\&|SRC|FAC\rPID|||12345||Doe^Pat||19900101|U"
        msg = HL7Message(msg_text)
        patient = HL7v2Source._extract_patient(msg)
        assert patient["gender"] == "unknown"


class TestCSVMappingEdgeCases:
    @pytest.mark.asyncio
    async def test_csv_with_missing_columns(self, tmp_path: Path) -> None:
        """CSV with columns not in auto-mapping should produce empty map."""
        csv_content = "foo,bar,baz\n1,2,3\n"
        fpath = tmp_path / "unknown.csv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        # No recognized columns -> no patient records
        assert len(dataset.records) == 0

    @pytest.mark.asyncio
    async def test_csv_with_empty_values(self, tmp_path: Path) -> None:
        """Empty cell values should be handled gracefully."""
        csv_content = "patient_id,first_name,last_name\nP001,,Smith\n"
        fpath = tmp_path / "sparse.csv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() >= 1

    @pytest.mark.asyncio
    async def test_csv_tab_delimiter(self, tmp_path: Path) -> None:
        csv_content = "patient_id\tfirst_name\nP001\tAlice\n"
        fpath = tmp_path / "patients.tsv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath), delimiter="\t")
        dataset = await source.ingest()
        assert dataset.patients.count() >= 1

    @pytest.mark.asyncio
    async def test_csv_with_phone_and_email(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,phone,email,ssn\n"
            "P001,Alice,555-1234,alice@test.com,123-45-6789\n"
        )
        fpath = tmp_path / "contact.csv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        patient = dataset.patients
        assert patient.count() >= 1

    @pytest.mark.asyncio
    async def test_csv_with_address_fields(self, tmp_path: Path) -> None:
        csv_content = (
            "patient_id,first_name,address,city,state,zip_code\n"
            "P001,Alice,123 Main St,Springfield,IL,62704\n"
        )
        fpath = tmp_path / "address.csv"
        fpath.write_text(csv_content)

        source = CSVSource(path=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() >= 1

    def test_set_nested_telecom(self) -> None:
        data: dict = {}
        _set_nested(data, "telecom.phone", "555-1234")
        assert data["telecom"][0]["system"] == "phone"
        assert data["telecom"][0]["value"] == "555-1234"

    def test_set_nested_identifier(self) -> None:
        data: dict = {}
        _set_nested(data, "identifier.SSN", "123-45-6789")
        assert data["identifier"][0]["system"] == "SSN"

    def test_set_nested_generic_two_level(self) -> None:
        data: dict = {}
        _set_nested(data, "meta.versionId", "1")
        assert data["meta"]["versionId"] == "1"

    def test_set_nested_single_level(self) -> None:
        data: dict = {}
        _set_nested(data, "gender", "male")
        assert data["gender"] == "male"


class TestFHIRSourceEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_bundle(self, tmp_path: Path) -> None:
        """Bundle with no entries should produce empty dataset."""
        bundle = {"resourceType": "Bundle", "entry": []}
        fpath = tmp_path / "empty.json"
        fpath.write_text(json.dumps(bundle))

        source = FHIRSource(url=str(fpath))
        dataset = await source.ingest()
        assert len(dataset.records) == 0

    @pytest.mark.asyncio
    async def test_bundle_without_entry_key(self, tmp_path: Path) -> None:
        """Bundle without 'entry' key should still work (0 records)."""
        bundle = {"resourceType": "Bundle"}
        fpath = tmp_path / "no_entry.json"
        fpath.write_text(json.dumps(bundle))

        source = FHIRSource(url=str(fpath))
        dataset = await source.ingest()
        assert len(dataset.records) == 0

    @pytest.mark.asyncio
    async def test_bundle_entries_without_resource_key(self, tmp_path: Path) -> None:
        """Entries without a 'resource' key should use the entry itself."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resourceType": "Patient", "id": "p1"},
            ],
        }
        fpath = tmp_path / "flat.json"
        fpath.write_text(json.dumps(bundle))

        source = FHIRSource(url=str(fpath))
        dataset = await source.ingest()
        assert dataset.patients.count() == 1

    def test_fhir_auth_default(self) -> None:
        auth = FHIRAuth()
        assert auth.token is None

    def test_fhir_auth_with_token(self) -> None:
        auth = FHIRAuth(token="test-bearer-token")
        assert auth.token == "test-bearer-token"  # noqa: S105

    def test_build_headers_with_auth(self) -> None:
        source = FHIRSource(
            url="http://test",
            auth=FHIRAuth(token="my-token"),
        )
        headers = source._build_headers()
        assert headers["Authorization"] == "Bearer my-token"

    def test_build_headers_without_auth(self) -> None:
        source = FHIRSource(url="http://test")
        headers = source._build_headers()
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/fhir+json"


class TestPDFSourceEdgeCases:
    @pytest.mark.asyncio
    async def test_pdf_no_matching_files(self) -> None:
        source = PDFSource(path="/nonexistent/path/*.pdf")
        with pytest.raises(IngestError, match="No PDF files matched"):
            await source.ingest()

    def test_text_to_records_with_patient_info(self) -> None:
        text = (
            "Patient: John Smith\n"
            "DOB: 03/15/1985\n"
            "MRN: MR12345\n"
            "Diagnosis: E11.9 Type 2 Diabetes\n"
        )
        records = PDFSource._text_to_records(text, source_uri="test.pdf")
        # Should have at least: Patient + Condition + DocumentReference
        assert len(records) >= 2
        resource_types = [r.resource_type for r in records]
        assert ResourceType.PATIENT in resource_types
        assert ResourceType.DOCUMENT_REFERENCE in resource_types

    def test_text_to_records_no_patient_info(self) -> None:
        text = "This is a generic document with no patient data."
        records = PDFSource._text_to_records(text, source_uri="generic.pdf")
        # Should have at least the DocumentReference
        assert len(records) >= 1
        assert records[-1].resource_type == ResourceType.DOCUMENT_REFERENCE

    def test_text_to_records_icd_codes(self) -> None:
        text = "Diagnoses: E11.9, I10, J45.909"
        records = PDFSource._text_to_records(text, source_uri="test.pdf")
        condition_records = [
            r for r in records if r.resource_type == ResourceType.CONDITION
        ]
        assert len(condition_records) >= 2

    def test_text_to_records_single_name(self) -> None:
        """If only one name part found, it should still create a patient."""
        text = "Name: Alice"
        records = PDFSource._text_to_records(text, source_uri="test.pdf")
        patient_records = [
            r for r in records if r.resource_type == ResourceType.PATIENT
        ]
        assert len(patient_records) >= 1

    @pytest.mark.asyncio
    async def test_pdf_extract_text_fallback(self, tmp_path: Path) -> None:
        """When no PDF library is available, falls back to raw text read."""
        fpath = tmp_path / "test.pdf"
        # Write a text file with .pdf extension
        fpath.write_text("Patient: Jane Doe\nDOB: 01/01/1990\n")

        with (
            patch(
                "healthpipe.ingest.pdf_ocr._try_extract_text_pdfplumber",
                return_value=None,
            ),
            patch(
                "healthpipe.ingest.pdf_ocr._try_ocr",
                return_value=None,
            ),
        ):
            source = PDFSource(path=str(fpath))
            dataset = await source.ingest()
            # Should have extracted something from the raw text
            assert len(dataset.records) >= 1

    @pytest.mark.asyncio
    async def test_pdf_with_pdfplumber(self, tmp_path: Path) -> None:
        """Test path where pdfplumber extraction succeeds."""
        fpath = tmp_path / "test.pdf"
        fpath.write_text("dummy content")

        with patch(
            "healthpipe.ingest.pdf_ocr._try_extract_text_pdfplumber",
            return_value="Patient: Alice Smith\nDOB: 03/15/1990\n",
        ):
            source = PDFSource(path=str(fpath))
            dataset = await source.ingest()
            assert len(dataset.records) >= 1

    @pytest.mark.asyncio
    async def test_pdf_with_ocr(self, tmp_path: Path) -> None:
        """Test path where pdfplumber fails but OCR succeeds."""
        fpath = tmp_path / "test.pdf"
        fpath.write_text("dummy content")

        with (
            patch(
                "healthpipe.ingest.pdf_ocr._try_extract_text_pdfplumber",
                return_value=None,
            ),
            patch(
                "healthpipe.ingest.pdf_ocr._try_ocr",
                return_value="Patient: Bob Jones\n",
            ),
        ):
            source = PDFSource(path=str(fpath))
            dataset = await source.ingest()
            assert len(dataset.records) >= 1

    @pytest.mark.asyncio
    async def test_pdf_empty_content_raises(self, tmp_path: Path) -> None:
        """If all extraction methods fail and the file has only
        whitespace, should raise IngestError."""
        fpath = tmp_path / "empty.pdf"
        fpath.write_text("   \n  ")

        with (
            patch(
                "healthpipe.ingest.pdf_ocr._try_extract_text_pdfplumber",
                return_value=None,
            ),
            patch(
                "healthpipe.ingest.pdf_ocr._try_ocr",
                return_value=None,
            ),
        ):
            source = PDFSource(path=str(fpath))
            with pytest.raises(IngestError, match="Could not extract"):
                await source.ingest()
