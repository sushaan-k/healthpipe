"""Tests that de-identification preserves original field ordering."""

from __future__ import annotations

import pytest

from healthpipe.deidentify.safe_harbor import (
    SafeHarborConfig,
    SafeHarborEngine,
)
from healthpipe.ingest.schema import ClinicalDataset, ClinicalRecord, ResourceType


def _make_dataset(data: dict) -> ClinicalDataset:
    record = ClinicalRecord(
        resource_type=ResourceType.PATIENT,
        data=data,
        source_format="FHIR_R4",
        source_uri="test://field-ordering",
    )
    return ClinicalDataset(records=[record])


class TestFieldOrdering:
    @pytest.mark.asyncio
    async def test_output_keys_match_input_order(self) -> None:
        """After de-identification, field order should match the input."""
        original_data = {
            "resourceType": "Patient",
            "id": "patient-ordering-test",
            "name": [{"family": "Smith", "given": ["John"]}],
            "gender": "male",
            "birthDate": "1985-03-15",
            "address": [{"line": ["123 Main St"], "city": "Springfield"}],
            "telecom": [{"system": "phone", "value": "555-123-4567"}],
            "identifier": [{"system": "SSN", "value": "123-45-6789"}],
        }
        original_keys = list(original_data.keys())
        dataset = _make_dataset(original_data)

        config = SafeHarborConfig(
            date_shift=True,
            date_shift_salt="test-ordering-salt",
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        result = await engine.run(dataset)

        output_keys = list(result.records[0].data.keys())
        assert output_keys == original_keys

    @pytest.mark.asyncio
    async def test_field_order_preserved_without_date_shift(self) -> None:
        original_data = {
            "telecom": [{"system": "email", "value": "test@example.com"}],
            "resourceType": "Patient",
            "id": "p-001",
            "birthDate": "1990-01-01",
        }
        original_keys = list(original_data.keys())
        dataset = _make_dataset(original_data)

        config = SafeHarborConfig(
            date_shift=False,
            use_fallback_ner=True,
            llm_verification=False,
        )
        engine = SafeHarborEngine(config)
        result = await engine.run(dataset)

        output_keys = list(result.records[0].data.keys())
        assert output_keys == original_keys

    def test_restore_key_order_static_method(self) -> None:
        data = {"c": 3, "a": 1, "b": 2}
        original_order = ["a", "b", "c"]
        restored = SafeHarborEngine._restore_key_order(data, original_order)
        assert list(restored.keys()) == ["a", "b", "c"]

    def test_restore_key_order_with_new_keys(self) -> None:
        data = {"a": 1, "b": 2, "extra": 99}
        original_order = ["b", "a"]
        restored = SafeHarborEngine._restore_key_order(data, original_order)
        assert list(restored.keys()) == ["b", "a", "extra"]

    def test_restore_key_order_with_removed_keys(self) -> None:
        data = {"a": 1}
        original_order = ["a", "b", "c"]
        restored = SafeHarborEngine._restore_key_order(data, original_order)
        assert list(restored.keys()) == ["a"]
