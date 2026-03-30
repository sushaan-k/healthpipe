# HIPAA Compliance

`healthpipe` is built to support HIPAA-oriented data processing workflows, with
the strongest built-in support centered on de-identification, auditability, and
privacy-preserving downstream analysis.

This document describes what the package currently implements. It is not legal
advice and it is not a substitute for a formal compliance program.

## What `healthpipe` Covers Directly

The current package provides concrete support for:

- HIPAA Safe Harbor-style identifier removal and redaction
- deterministic date shifting with a required secret salt
- structured audit logging for de-identification operations
- compliance report generation from audit data
- privacy-budget tracking for differential privacy queries
- lineage tracking for dataset transformations

Relevant modules:

- `healthpipe.deidentify.safe_harbor`
- `healthpipe.deidentify.patterns`
- `healthpipe.deidentify.date_shift`
- `healthpipe.deidentify.llm_verify`
- `healthpipe.audit.logger`
- `healthpipe.audit.compliance`
- `healthpipe.audit.lineage`

## De-identification Workflow

The main de-identification path is:

1. ingest clinical data into the unified schema
2. run `deidentify(..., method="safe_harbor")`
3. optionally shift dates with a secret salt
4. optionally run an LLM verification pass for residual identifiers
5. persist the returned de-identified dataset plus audit log

Example:

```python
import asyncio
import healthpipe as hp


async def main() -> None:
    dataset = await hp.ingest([hp.CSVSource(path="patients.csv")])
    deidentified = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift=True,
        date_shift_salt="replace-with-a-secret-salt",
        date_shift_range=(-365, 365),
        llm_verification=False,
    )
    print(len(deidentified.audit_log.entries))


asyncio.run(main())
```

## Safe Harbor-Oriented Controls

The package combines several mechanisms instead of relying on a single regex
pass:

- named-entity detection for person, location, and organization-like strings
- explicit pattern matching for identifiers such as phone numbers, emails, MRNs
- date shifting for temporal fields and recognized date strings
- optional LLM review for residual free-text leaks

These layers work together, but they are still bounded by configuration and the
quality of the input data.

## Audit Evidence

`healthpipe` records structured audit entries for transformation steps. The
audit trail is designed to answer:

- what was removed or transformed
- which strategy performed the action
- when the action occurred
- what compliance checks were run

`ComplianceReporter` turns audit information into summary-oriented compliance
output for review workflows.

## Differential Privacy and Compliance

Differential privacy is not a HIPAA Safe Harbor requirement, but it is useful
for downstream analytics once identifiers have been removed.

`healthpipe.private_stats(...)` supports:

- `Count`
- `Mean`
- `Histogram`

along with budget accounting through `PrivacyBudget`.

This lets teams separate:

- record-level de-identification
- aggregate releases with explicit privacy spending

## What Requires External Process, Not Just Library Calls

No Python package can make an organization HIPAA compliant by itself. You still
need:

- access controls
- key management
- operational auditing and retention policy
- business associate agreements where relevant
- incident response and breach handling
- legal/compliance review

`healthpipe` helps with the data-processing layer inside that larger program.

## Recommended Operational Defaults

- keep `date_shift_salt` secret and environment-managed
- treat `llm_verification=True` as an explicit privacy review decision
- persist audit logs alongside de-identified outputs
- validate synthetic outputs before developer/test reuse
- document approved input sources and mappings per pipeline

## Known Boundaries

- LLM verification is optional and provider-dependent
- OCR quality affects PDF-derived PHI detection quality
- free-text clinical notes remain the highest-risk input class
- compliance output is only as trustworthy as the underlying audit trail and
  configuration
