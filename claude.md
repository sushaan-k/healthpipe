# healthpipe

## Privacy-Preserving Clinical Data Pipeline

### The Problem

Healthcare AI is stuck. Not because the models are bad — but because the **data infrastructure is a disaster**.

Every health tech startup (including vytus.health) hits the same wall:
1. Clinical data is messy (HL7v2, FHIR, CSV dumps, PDFs, faxes)
2. Privacy requirements are brutal (HIPAA, state laws, IRB requirements)
3. De-identification is done manually or with brittle regex
4. There's no standard pipeline for: ingest → de-identify → transform → analyze
5. Synthetic data generation for development/testing doesn't exist in a usable form

The result: 80% of health AI engineering time is spent on data wrangling, and most startups build bespoke pipelines that can't be reused or audited.

NVIDIA just expanded their open model families for healthcare AI (March 2026), but the data pipeline layer underneath is still the Wild West.

### The Solution

`healthpipe` is an open-source framework for ingesting, de-identifying, transforming, and analyzing clinical/health data with built-in differential privacy, HIPAA-compliant audit logging, and synthetic data generation.

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      healthpipe                           │
│                                                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │                  Ingest Layer                     │     │
│  │                                                   │     │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌───────┐  │     │
│  │  │ FHIR │ │HL7v2 │ │ CSV  │ │ PDF  │ │ CCD/  │  │     │
│  │  │      │ │      │ │      │ │(OCR) │ │ C-CDA │  │     │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬────┘  │     │
│  │     └────────┴────────┴────────┴────────┘        │     │
│  │                      │                            │     │
│  │                      ▼                            │     │
│  │            Unified Internal Schema                │     │
│  │         (FHIR R4 as canonical form)               │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              De-identification Engine              │     │
│  │                                                   │     │
│  │  Layer 1: Named Entity Recognition (NER)          │     │
│  │    - Patient names, dates, locations, MRNs        │     │
│  │    - Trained on clinical text (not general NER)   │     │
│  │                                                   │     │
│  │  Layer 2: Pattern Matching                        │     │
│  │    - SSN, phone, email, address patterns          │     │
│  │    - Medical record numbers, account numbers      │     │
│  │                                                   │     │
│  │  Layer 3: Date Shifting                           │     │
│  │    - Consistent date offsets per patient           │     │
│  │    - Preserves intervals (critical for clinical)  │     │
│  │                                                   │     │
│  │  Layer 4: LLM Verification                        │     │
│  │    - Final pass: LLM checks for any remaining PII │     │
│  │    - Catches context-dependent identifiers         │     │
│  │                                                   │     │
│  │  Output: De-identified data + audit log            │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │           Privacy Guarantees Layer                 │     │
│  │                                                   │     │
│  │  ┌──────────────────┐  ┌───────────────────────┐  │     │
│  │  │ Differential     │  │ k-Anonymity /         │  │     │
│  │  │ Privacy          │  │ l-Diversity           │  │     │
│  │  │                  │  │                       │  │     │
│  │  │ - Laplace noise  │  │ - Quasi-identifier    │  │     │
│  │  │ - Gaussian mech  │  │   generalization      │  │     │
│  │  │ - Privacy budget │  │ - Suppression rules   │  │     │
│  │  │   tracking (ε)   │  │ - Validation checks   │  │     │
│  │  └──────────────────┘  └───────────────────────┘  │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │          Synthetic Data Generator                  │     │
│  │                                                   │     │
│  │  - Learns statistical distributions from real data │     │
│  │  - Generates synthetic patients with realistic     │     │
│  │    correlations (age↔diagnosis, medication↔labs)   │     │
│  │  - Validates: synthetic data cannot be linked      │     │
│  │    back to any real patient                        │     │
│  │  - Utility metrics: how useful is synthetic data   │     │
│  │    compared to real data for model training?       │     │
│  └──────────────────────┬───────────────────────────┘     │
│                         │                                  │
│                         ▼                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Audit & Compliance                    │     │
│  │                                                   │     │
│  │  - Every operation logged (who, what, when)        │     │
│  │  - HIPAA Safe Harbor / Expert Determination audit  │     │
│  │  - Privacy budget consumption tracking             │     │
│  │  - Export compliance reports (PDF)                  │     │
│  │  - Data lineage: trace any output back to source   │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Universal Ingest

Handles every format you encounter in healthcare:

```python
import healthpipe as hp

# Ingest from any source
dataset = hp.ingest([
    hp.FHIRSource(url="https://fhir.hospital.org/R4", auth=...),
    hp.HL7v2Source(path="./hl7_messages/*.hl7"),
    hp.CSVSource(path="./patient_data.csv", mapping="auto"),  # auto-maps columns to FHIR
    hp.PDFSource(path="./clinical_notes/*.pdf"),              # OCR + NLP extraction
])

# Everything normalized to FHIR R4
print(dataset.patients.count())
print(dataset.observations.count())
print(dataset.conditions.count())
```

#### 2. Multi-Layer De-identification

```python
# De-identify with configurable aggressiveness
deidentified = hp.deidentify(
    dataset,
    method="safe_harbor",          # HIPAA Safe Harbor method (remove 18 identifiers)
    date_shift=True,               # Shift dates, preserve intervals
    date_shift_range=(-365, 365),  # Random offset per patient
    llm_verification=True,         # Final LLM pass for context-dependent PII
    llm_model="claude-haiku-4-5",  # Fast, cheap verification
)

# Audit what was removed
print(deidentified.audit_log)
# [PHI Removed] Patient name "John Smith" → "[PATIENT]" (NER, confidence: 0.99)
# [PHI Removed] Date "2025-03-15" → "2024-08-22" (date shift: -205 days)
# [PHI Removed] MRN "MR-12345" → "[MRN]" (pattern match)
# [PHI Verified] LLM found no additional PII in free-text fields
```

#### 3. Differential Privacy

```python
# Apply differential privacy to aggregate statistics
stats = hp.private_stats(
    deidentified,
    epsilon=1.0,                    # Privacy budget
    queries=[
        hp.Count(field="patient", group_by="diagnosis"),
        hp.Mean(field="lab_results.glucose", group_by="age_group"),
        hp.Histogram(field="medications", bins=20),
    ]
)

print(f"Privacy budget remaining: {stats.budget_remaining}")
# Grouped queries return per-group dictionaries
print(stats.results["count:patient|group_by:diagnosis"])
# Warns if budget is getting low
```

#### 4. Synthetic Data Generation

```python
# Generate synthetic patients
synthetic = hp.synthesize(
    deidentified,
    n_patients=10000,
    method="ctgan",                 # Conditional Tabular GAN
    preserve_correlations=True,     # Maintain clinical relationships
    validate=True,                  # Check no synthetic patient matches a real one
)

# Utility report: how good is the synthetic data?
utility = hp.evaluate_utility(synthetic, deidentified)
print(f"Statistical fidelity: {utility.fidelity:.2%}")
print(f"ML utility (AUC preservation): {utility.ml_utility:.2%}")
print(f"Privacy risk (re-identification): {utility.reidentification_risk:.6f}")
```

### HIPAA Compliance Features

| HIPAA Requirement | healthpipe Feature |
|---|---|
| Safe Harbor (18 identifiers) | Multi-layer de-identification engine |
| Minimum Necessary | Configurable field access controls |
| Audit Controls | Comprehensive audit logging |
| Data Integrity | Checksums on all transformations |
| Transmission Security | TLS enforcement on all data sources |
| Breach Notification | Re-identification risk scoring |

### Technical Stack

- **Language**: Python 3.11+
- **FHIR**: `fhir.resources` (Pydantic-based FHIR models)
- **NER**: `spaCy` with `en_core_sci_lg` (clinical NER model)
- **OCR**: `pytesseract` for PDF extraction
- **Differential Privacy**: `opendp` (OpenDP library)
- **Synthetic Data**: `sdv` (Synthetic Data Vault) / `ctgan`
- **Storage**: `duckdb` (analytical queries), `parquet` (file output)
- **Audit**: Structured JSON logging with data lineage

### What Makes This Novel

1. **End-to-end pipeline** — ingest through synthetic generation, not just one piece
2. **Multi-layer de-identification with LLM verification** — catches what regex and NER miss
3. **Differential privacy built in** — formal privacy guarantees, not just "we removed names"
4. **Synthetic data with utility validation** — generates usable fake data and proves it's useful
5. **Directly relevant to vytus.health** — shows you build real health infrastructure

### Repo Structure

```
healthpipe/
├── README.md
├── pyproject.toml
├── src/
│   └── healthpipe/
│       ├── __init__.py
│       ├── ingest/
│       │   ├── fhir.py             # FHIR R4 source
│       │   ├── hl7v2.py            # HL7v2 parser
│       │   ├── csv_mapper.py       # CSV → FHIR mapping
│       │   ├── pdf_ocr.py          # PDF extraction
│       │   └── schema.py           # Internal unified schema
│       ├── deidentify/
│       │   ├── ner.py              # Clinical NER
│       │   ├── patterns.py         # Regex pattern matching
│       │   ├── date_shift.py       # Date shifting
│       │   ├── llm_verify.py       # LLM verification pass
│       │   └── safe_harbor.py      # HIPAA Safe Harbor orchestration
│       ├── privacy/
│       │   ├── differential.py     # Differential privacy mechanisms
│       │   ├── k_anonymity.py      # k-Anonymity / l-Diversity
│       │   └── budget.py           # Privacy budget tracking
│       ├── synthetic/
│       │   ├── generator.py        # Synthetic data generation
│       │   ├── validator.py        # Re-identification risk testing
│       │   └── utility.py          # Utility evaluation
│       ├── audit/
│       │   ├── logger.py           # Audit logging
│       │   ├── lineage.py          # Data lineage tracking
│       │   └── compliance.py       # Compliance report generation
│       └── pipeline.py             # Main orchestration
├── tests/
├── examples/
│   ├── fhir_pipeline.py
│   ├── csv_deidentify.py
│   └── synthetic_generation.py
└── docs/
    ├── hipaa_compliance.md
    ├── de-identification.md
    └── synthetic_data.md
```

### Research References

- HIPAA Privacy Rule (45 CFR §164.514) — Safe Harbor and Expert Determination methods
- "Clinical NLP De-identification: A Survey" (JAMIA, 2024)
- "Synthetic Data Generation for Healthcare: A Review" (Nature Medicine, 2025)
- OpenDP Library documentation (opendp.org)
- NVIDIA healthcare AI expansion (March 2026)
- Synthetic Data Vault documentation (sdv.dev)
