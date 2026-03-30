"""FHIR R4 data source.

Connects to a FHIR R4 server (or reads FHIR Bundle JSON files) and
converts resources into the healthpipe unified schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from healthpipe.exceptions import FHIRValidationError, IngestError
from healthpipe.ingest.schema import (
    ClinicalDataset,
    ClinicalRecord,
    ResourceType,
)

logger = logging.getLogger(__name__)

_FHIR_TO_RESOURCE: dict[str, ResourceType] = {
    "Patient": ResourceType.PATIENT,
    "Observation": ResourceType.OBSERVATION,
    "Condition": ResourceType.CONDITION,
    "MedicationRequest": ResourceType.MEDICATION_REQUEST,
    "Encounter": ResourceType.ENCOUNTER,
    "DiagnosticReport": ResourceType.DIAGNOSTIC_REPORT,
    "Procedure": ResourceType.PROCEDURE,
    "AllergyIntolerance": ResourceType.ALLERGY_INTOLERANCE,
    "Immunization": ResourceType.IMMUNIZATION,
    "DocumentReference": ResourceType.DOCUMENT_REFERENCE,
}


class FHIRAuth(BaseModel):
    """Authentication configuration for a FHIR server."""

    token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    token_url: str | None = None


class FHIRSource(BaseModel):
    """Ingest adapter for FHIR R4 servers and Bundle JSON files.

    Args:
        url: FHIR server base URL *or* path to a local Bundle JSON file.
        auth: Optional authentication configuration.
        resource_types: Which FHIR resource types to fetch. Defaults to all
            supported types.
        page_size: Number of resources per page when paginating server results.
    """

    url: str
    auth: FHIRAuth | None = None
    resource_types: list[str] = Field(
        default_factory=lambda: list(_FHIR_TO_RESOURCE.keys())
    )
    page_size: int = 100

    # -- Public API ------------------------------------------------------------

    async def ingest(self) -> ClinicalDataset:
        """Fetch resources and return a ``ClinicalDataset``.

        If ``url`` points to a local ``.json`` file the Bundle is read from
        disk; otherwise an HTTP GET is issued to the FHIR server.
        """
        path = Path(self.url)
        if path.exists() and path.suffix == ".json":
            return self._ingest_bundle_file(path)
        return await self._ingest_server()

    # -- Private helpers -------------------------------------------------------

    def _ingest_bundle_file(self, path: Path) -> ClinicalDataset:
        """Parse a FHIR Bundle JSON file from disk."""
        import json

        logger.info("Ingesting FHIR Bundle from %s", path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise IngestError(f"Failed to read FHIR Bundle from {path}") from exc

        return self._bundle_to_dataset(raw, source_uri=str(path))

    async def _ingest_server(self) -> ClinicalDataset:
        """Fetch resources from a live FHIR R4 server."""
        logger.info("Connecting to FHIR server at %s", self.url)
        headers = self._build_headers()
        dataset = ClinicalDataset()

        async with httpx.AsyncClient(
            base_url=self.url.rstrip("/"),
            headers=headers,
            timeout=30.0,
        ) as client:
            for rtype in self.resource_types:
                if rtype not in _FHIR_TO_RESOURCE:
                    logger.warning("Skipping unsupported resource type: %s", rtype)
                    continue
                await self._fetch_resource_type(client, rtype, dataset)

        logger.info("Ingested %d records from FHIR server", len(dataset.records))
        return dataset

    async def _fetch_resource_type(
        self,
        client: httpx.AsyncClient,
        resource_type: str,
        dataset: ClinicalDataset,
    ) -> None:
        """Page through all resources of *resource_type*."""
        url = f"/{resource_type}?_count={self.page_size}"

        while url:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise IngestError(
                    f"FHIR server request failed for {resource_type}: {exc}"
                ) from exc

            bundle = resp.json()
            self._process_bundle_entries(bundle, dataset, source_uri=self.url)

            # Follow pagination links
            url = ""
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    url = link["url"]
                    # Make relative if same host
                    if url.startswith(self.url):
                        url = url[len(self.url.rstrip("/")) :]
                    break

    def _process_bundle_entries(
        self,
        bundle: dict[str, Any],
        dataset: ClinicalDataset,
        source_uri: str,
    ) -> None:
        """Extract entries from a FHIR Bundle dict."""
        entries = bundle.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", entry)
            rtype_str = resource.get("resourceType", "")
            mapped = _FHIR_TO_RESOURCE.get(rtype_str)
            if mapped is None:
                continue
            record = ClinicalRecord(
                resource_type=mapped,
                data=resource,
                source_format="FHIR_R4",
                source_uri=source_uri,
            )
            dataset.add_record(record)

    def _bundle_to_dataset(
        self, raw: dict[str, Any], source_uri: str
    ) -> ClinicalDataset:
        """Convert a raw Bundle dict into a ClinicalDataset."""
        if raw.get("resourceType") != "Bundle":
            raise FHIRValidationError(
                "Expected a FHIR Bundle but got "
                f"resourceType={raw.get('resourceType', 'unknown')}"
            )
        dataset = ClinicalDataset()
        self._process_bundle_entries(raw, dataset, source_uri=source_uri)
        logger.info("Parsed %d records from Bundle", len(dataset.records))
        return dataset

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including auth if configured."""
        headers: dict[str, str] = {
            "Accept": "application/fhir+json",
        }
        if self.auth and self.auth.token:
            headers["Authorization"] = f"Bearer {self.auth.token}"
        return headers
