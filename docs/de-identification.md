# De-identification

`healthpipe` exposes a multi-layer de-identification pipeline centered on
`deidentify()` and `SafeHarborEngine`.

## Entry Point

The primary entry point is:

```python
deidentified = await healthpipe.deidentify(
    dataset,
    method="safe_harbor",
    date_shift=True,
    date_shift_salt="secret-salt",
)
```

This returns a `DeidentifiedDataset` containing transformed records plus audit
metadata.

## Pipeline Stages

The de-identification stack is implemented across these modules:

- `healthpipe.deidentify.ner`
- `healthpipe.deidentify.patterns`
- `healthpipe.deidentify.date_shift`
- `healthpipe.deidentify.llm_verify`
- `healthpipe.deidentify.safe_harbor`

At a high level:

1. walk records and free-text content
2. detect direct identifiers with rule and NER passes
3. redact or replace detected spans
4. shift dates deterministically per patient when enabled
5. optionally run an LLM pass for contextual residual PHI
6. write audit entries for what changed

## Date Shifting

Date shifting is handled by `DateShifter`.

Important operational properties:

- requires a non-empty salt
- deterministically maps a patient identifier to an offset
- preserves relative intervals within a patient timeline
- can operate on date objects and recognized date strings

Typical configuration:

```python
deidentified = await hp.deidentify(
    dataset,
    method="safe_harbor",
    date_shift=True,
    date_shift_salt="rotate-this-secret",
    date_shift_range=(-365, 365),
)
```

## Pattern Matching

The pattern-matching layer targets structured identifiers that are reliably
captured with explicit rules, such as:

- phone numbers
- email addresses
- medical record identifiers
- social-security-like patterns
- network-style identifiers where configured

This is the fastest and most deterministic stage in the pipeline.

## Named Entity Recognition

`ClinicalNER` provides a higher-level pass for entities that are harder to
detect with raw patterns alone, especially in free text.

The implementation is designed to work with optional NLP dependencies and test
fallbacks, so teams can choose between:

- lightweight fallback behavior
- richer NLP-backed extraction paths

## LLM Verification

`LLMVerifier` is an optional secondary review pass for residual PHI in text
that survived the rule-based layers.

Use it when:

- you have dense unstructured notes
- you can justify sending text to an approved provider
- you want a second pass for audit/review, not a sole privacy control

Do not treat it as a replacement for deterministic identifier removal.

## Example

```python
import asyncio
import healthpipe as hp


async def main() -> None:
    dataset = await hp.ingest([hp.HL7v2Source(path="messages.hl7")])
    result = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift=True,
        date_shift_salt="secret",
        llm_verification=True,
        llm_model="claude-haiku-4-5",
    )

    print(result.audit_log.entries[0])


asyncio.run(main())
```

## Output Shape

The returned `DeidentifiedDataset` is intended to preserve downstream utility
while removing direct identifiers. Typical downstream uses are:

- differential privacy queries
- synthetic data generation
- analyst review on de-identified records

## Practical Boundaries

- OCR errors can leak through if the source text is degraded.
- Free-text notes need more scrutiny than strongly structured resources.
- Date shifting preserves intervals, which is useful, but it does not hide all
  temporal structure.
- De-identification quality depends on correct patient linkage and source
  mapping.
