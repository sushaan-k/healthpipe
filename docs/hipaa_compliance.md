# HIPAA Safe Harbor Compliance Guide

`healthpipe` is built to support HIPAA-oriented data processing workflows.
This guide maps each of the 18 Safe Harbor identifiers (45 CFR 164.514(b)(2))
to the healthpipe features that handle them, and explains which identifiers
require manual review.

> **Disclaimer:** This document describes what the software implements.
> It is not legal advice and does not replace a formal compliance program.

---

## Safe Harbor Identifier Coverage

The table below lists all 18 HIPAA Safe Harbor identifiers and how
healthpipe addresses each one.

| # | Identifier | Auto-handled | Layer | Notes |
|---|-----------|:---:|-------|-------|
| 1 | Names | Yes | NER (fallback + spaCy) | Context heuristics detect "Patient: First Last" patterns; spaCy PERSON labels used when available. Confidence 0.70-0.85 depending on method. |
| 2 | Geographic data (street, city, state, ZIP) | Partial | Pattern | Street addresses and ZIP codes detected by regex. ZIP codes reduced to 3-digit prefix per Safe Harbor. City/state removal requires NER or manual review. |
| 3 | Dates (except year) | Yes | Date Shift | All dates shifted by a consistent per-patient random offset. Preserves intervals. Ages over 89 collapsed to 90. |
| 4 | Telephone numbers | Yes | Pattern | US phone formats: (555) 123-4567, 555-123-4567, 555.123.4567. |
| 5 | Fax numbers | Yes | Pattern | Same regex as telephone covers fax numbers in text. |
| 6 | Email addresses | Yes | Pattern | Standard RFC-style email pattern matching. |
| 7 | Social Security Numbers | Yes | Pattern | Dashed format (XXX-XX-XXXX) with SSA-valid range checks, plus context-keyword triggered detection ("SSN:", "Social Security:"). |
| 8 | Medical Record Numbers | Yes | Pattern | Patterns like MRN-12345, MRN: 12345, MR-12345. |
| 9 | Health plan beneficiary numbers | Yes | Pattern | Detects "Health Plan", "Beneficiary", "Member ID", "Group", "Subscriber" followed by alphanumeric codes. |
| 10 | Account numbers | Yes | Pattern | Detects "Account", "Acct", "Policy", "Insurance" followed by 6+ digit numbers. |
| 11 | Certificate/license numbers | Yes | Pattern | Detects "License", "Certificate", "DEA", "NPI", "UPIN" followed by alphanumeric codes. |
| 12 | Vehicle identifiers (VIN) | Yes | Pattern | 17-character VIN format (no I/O/Q per standard). |
| 13 | Device identifiers (UDI) | Yes | Pattern | Detects "UDI", "Device", "Serial" followed by alphanumeric codes. |
| 14 | URLs | Yes | Pattern | HTTP/HTTPS URL matching. |
| 15 | IP addresses | Yes | Pattern | IPv4 dotted-quad format. |
| 16 | Biometric identifiers | **No** | Manual | Fingerprints, retinal scans, voiceprints require specialized sensor data analysis. healthpipe does not process biometric data. |
| 17 | Full-face photographs | **No** | Manual | Image-based identifiers require computer vision. healthpipe does not process images. |
| 18 | Any other unique identifying number | Partial | LLM Verify | The optional LLM verification layer (Layer 4) can catch context-dependent identifiers. Requires API key. Otherwise, manual review needed. |

### Summary

- **Fully automated:** 15 of 18 identifiers (1-15)
- **Requires manual review:** Biometric identifiers (16), photographs (17)
- **Partially automated:** Geographic sub-fields (2), catch-all codes (18)

---

## Configuration Examples

### Basic De-identification (Most Common)

```python
import asyncio
import healthpipe as hp

async def main():
    dataset = await hp.ingest([hp.CSVSource(path="patients.csv")])
    result = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift=True,
        date_shift_salt="your-secret-salt-here",  # REQUIRED: keep secret
        llm_verification=False,
    )
    print(f"Removed {result.audit_log.phi_removed_count} PHI items")

asyncio.run(main())
```

### Dry-Run Mode (Preview Before De-identifying)

Use dry-run to see what would be detected without modifying data:

```python
import asyncio
import healthpipe as hp

async def main():
    config = hp.PipelineConfig(
        dry_run=True,
        deid_config=hp.SafeHarborConfig(
            use_fallback_ner=True,
            llm_verification=False,
        ),
    )
    pipeline = hp.Pipeline(config)
    result = await pipeline.run([hp.CSVSource(path="patients.csv")])

    report = result.dry_run_report
    print(f"Scanned {report.total_records_scanned} records")
    print(f"Found {report.total_phi_found} PHI items")
    print(f"Categories: {report.categories_found}")

    for finding in report.findings:
        print(
            f"  [{finding.category}] '{finding.original}' "
            f"(confidence: {finding.confidence}, "
            f"method: {finding.detection_method})"
        )

asyncio.run(main())
```

### With LLM Verification (Highest Coverage)

Adds a final LLM pass to catch context-dependent identifiers that
regex and NER may miss:

```python
result = await hp.deidentify(
    dataset,
    method="safe_harbor",
    date_shift=True,
    date_shift_salt="your-secret-salt",
    llm_verification=True,
    llm_model="claude-haiku-4-5",
    llm_api_key="sk-ant-...",  # keep in environment variable
)
```

### Custom Pattern Extension

Add domain-specific patterns for identifiers not covered by the
built-in rules:

```python
from healthpipe.deidentify.patterns import PatternMatcher

matcher = PatternMatcher(
    extra_patterns={
        "INTERNAL_ID": r"INT-\d{8}",          # Internal system IDs
        "STUDY_CODE": r"STUDY-[A-Z]{3}\d{4}", # Clinical trial codes
    }
)
matches = matcher.scan("Patient enrolled in STUDY-ABC1234, ID INT-00012345")
```

### Re-identification Risk Assessment

After de-identification, assess residual risk based on
quasi-identifier uniqueness:

```python
from healthpipe.privacy import ReidentificationRisk

scorer = ReidentificationRisk(
    quasi_identifiers=["age", "gender", "zip_prefix"],
    high_risk_threshold=0.20,
    medium_risk_threshold=0.05,
)
report = scorer.score(deidentified.records)
print(f"Risk level: {report.risk_level}")
print(f"Uniqueness ratio: {report.uniqueness_ratio:.2%}")
print(f"Prosecutor risk: {report.prosecutor_risk:.4f}")
```

---

## Date Shifting Details

Date shifting is critical for preserving clinical utility while removing
temporal identifiers:

- Each patient receives a **deterministic** random offset derived from
  `HMAC(salt, patient_id)`.
- The same patient always gets the same shift across pipeline runs (given
  the same salt), ensuring consistency across related records.
- **Intervals are preserved**: if two lab results were 7 days apart in
  the original data, they remain 7 days apart after shifting.
- Ages over 89 are collapsed to 90 per Safe Harbor requirements.
- The `date_shift_salt` must be kept secret. If compromised, an attacker
  could reverse the date shifts. Store it in environment variables or a
  secrets manager, never in source code.

---

## Audit Trail

Every de-identification action produces a structured `AuditEntry`:

- **record_id**: which record was affected
- **action**: what happened (`PHI_REMOVED`, `DATE_SHIFTED`, etc.)
- **layer**: which detection layer (`NER`, `PATTERN`, `DATE_SHIFT`, `LLM_VERIFY`)
- **category**: HIPAA identifier category
- **confidence**: detection confidence (0.0-1.0)
- **original_hash**: SHA-256 hash of the original PHI (raw value not stored)

The audit log is designed to support compliance reviews without itself
becoming a data-breach vector.

---

## What healthpipe Cannot Do

No software library can make an organization HIPAA-compliant by itself.
You still need:

- **Access controls**: role-based access to PHI
- **Key management**: secure storage for the date-shift salt and API keys
- **Operational auditing**: log retention and monitoring policies
- **Business Associate Agreements**: where third-party services are involved
- **Incident response**: breach notification procedures
- **Legal/compliance review**: formal sign-off from qualified counsel

healthpipe handles the data-processing layer inside that larger program.

---

## Recommended Operational Defaults

- Keep `date_shift_salt` in environment variables, not in code
- Run `dry_run=True` on new data sources before full de-identification
- Persist audit logs alongside de-identified outputs
- Use `ReidentificationRisk` to validate quasi-identifier diversity
- Treat `llm_verification=True` as a deliberate privacy-review decision
- Document approved input sources and column mappings per pipeline
- Validate synthetic outputs before using them for development/testing
