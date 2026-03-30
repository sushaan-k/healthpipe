"""Ingest layer: normalise clinical data from any source into FHIR R4."""

from healthpipe.ingest.csv_mapper import CSVSource
from healthpipe.ingest.fhir import FHIRSource
from healthpipe.ingest.hl7v2 import HL7v2Source
from healthpipe.ingest.pdf_ocr import PDFSource
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord

__all__ = [
    "CSVSource",
    "ClinicalDataset",
    "ClinicalRecord",
    "FHIRSource",
    "HL7v2Source",
    "PDFSource",
]
