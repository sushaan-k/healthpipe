"""PDF clinical-note extraction adapter.

Uses ``pytesseract`` (if available) for OCR and basic NLP heuristics to
pull patient demographics and clinical observations from scanned clinical
documents.  Falls back to ``pdfplumber`` / ``PyPDF2`` for text-layer PDFs.

When the heavy dependencies are absent the module still loads -- callers
get a clear ``IngestError`` explaining which optional package to install.
"""

from __future__ import annotations

import logging
import re
from glob import glob
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from healthpipe.exceptions import IngestError
from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    ResourceType,
)

logger = logging.getLogger(__name__)


def _try_extract_text_pdfplumber(path: Path) -> str | None:
    """Attempt text extraction with pdfplumber (text-layer PDFs)."""
    try:
        import pdfplumber
    except ImportError:
        return None

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n".join(pages_text) if pages_text else None


def _try_ocr(path: Path) -> str | None:
    """Attempt OCR with pytesseract + pdf2image."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return None

    images = convert_from_path(str(path))
    pages_text = [pytesseract.image_to_string(img) for img in images]
    return "\n".join(pages_text) if pages_text else None


class PDFSource(BaseModel):
    """Ingest adapter for PDF clinical documents.

    Args:
        path: Glob pattern or path to a single PDF.
    """

    path: str

    async def ingest(self) -> ClinicalDataset:
        """Extract text from PDFs and convert to ClinicalRecords."""
        files = [Path(p) for p in glob(self.path, recursive=True) if Path(p).is_file()]
        if not files:
            raise IngestError(f"No PDF files matched pattern: {self.path}")

        dataset = ClinicalDataset()
        for fpath in files:
            text = self._extract_text(fpath)
            records = self._text_to_records(text, source_uri=str(fpath))
            for rec in records:
                dataset.add_record(rec)

        logger.info(
            "Ingested %d records from %d PDFs", len(dataset.records), len(files)
        )
        return dataset

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _extract_text(path: Path) -> str:
        """Extract text from a PDF using the best available method."""
        # Try text-layer extraction first (fast, no OCR needed)
        text = _try_extract_text_pdfplumber(path)
        if text and text.strip():
            logger.debug("Extracted text-layer PDF: %s", path)
            return text

        # Fall back to OCR
        text = _try_ocr(path)
        if text and text.strip():
            logger.debug("OCR extracted text from PDF: %s", path)
            return text

        # If neither worked, read raw bytes as last resort
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            if raw.strip():
                return raw
        except OSError:
            pass

        raise IngestError(
            f"Could not extract text from {path}. "
            "Install 'pytesseract' and 'pdf2image' for OCR support, "
            "or 'pdfplumber' for text-layer PDFs."
        )

    @staticmethod
    def _text_to_records(text: str, source_uri: str) -> list[ClinicalRecord]:
        """Heuristically extract clinical data from free text.

        This is intentionally simple -- production use should pair with
        a clinical NLP model.  We look for common patterns like patient
        name headers, date of birth, and ICD codes.
        """
        records: list[ClinicalRecord] = []
        patient_data: dict[str, Any] = {"resourceType": "Patient"}

        # Name pattern: "Patient: Last, First" or "Name: First Last"
        name_match = re.search(
            r"(?:Patient|Name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            text,
            re.IGNORECASE,
        )
        if name_match:
            parts = name_match.group(1).strip().split()
            if len(parts) >= 2:
                patient_data["name"] = [{"given": parts[:-1], "family": parts[-1]}]
            elif parts:
                patient_data["name"] = [{"given": parts, "family": ""}]

        # DOB pattern
        dob_match = re.search(
            r"(?:DOB|Date of Birth|Birth\s*Date)\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text,
            re.IGNORECASE,
        )
        if dob_match:
            patient_data["birthDate"] = dob_match.group(1)

        # MRN pattern
        mrn_match = re.search(
            r"(?:MRN|Medical Record|Record Number)\s*:?\s*([A-Z0-9-]+)",
            text,
            re.IGNORECASE,
        )
        if mrn_match:
            patient_data["id"] = mrn_match.group(1)

        if len(patient_data) > 1:
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.PATIENT,
                    data=patient_data,
                    source_format="PDF",
                    source_uri=source_uri,
                )
            )

        # ICD-10 codes
        icd_matches = re.findall(r"\b([A-TV-Z]\d{2}(?:\.\d{1,4})?)\b", text)
        for code in set(icd_matches):
            records.append(
                ClinicalRecord(
                    resource_type=ResourceType.CONDITION,
                    data={
                        "resourceType": "Condition",
                        "code": {"system": "ICD-10-CM", "code": code},
                    },
                    source_format="PDF",
                    source_uri=source_uri,
                )
            )

        # Always store the full text as a DocumentReference
        records.append(
            ClinicalRecord(
                resource_type=ResourceType.DOCUMENT_REFERENCE,
                data={
                    "resourceType": "DocumentReference",
                    "content": text,
                    "description": f"Full text extracted from {source_uri}",
                },
                source_format="PDF",
                source_uri=source_uri,
            )
        )

        return records
