# healthpipe

[![CI](https://github.com/sushaan-k/healthpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/healthpipe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Privacy-preserving clinical data pipeline with HIPAA-compliant de-identification, differential privacy, and synthetic data generation.**

---

## The Problem

Healthcare AI is stuck. Not because the models are bad -- because the **data infrastructure is broken**.

Every health tech startup hits the same wall:

1. Clinical data is messy (HL7v2, FHIR, CSV dumps, PDFs, faxes)
2. Privacy requirements are brutal (HIPAA, state laws, IRB)
3. De-identification is manual or built on brittle regex
4. There is no standard pipeline for ingest -> de-identify -> transform -> analyze
5. Synthetic data for dev/testing does not exist in a usable form

**80% of health AI engineering time is spent on data wrangling.** Most startups build bespoke pipelines that cannot be reused or audited.

## The Solution

`healthpipe` is an open-source framework that provides the full pipeline: **ingest, de-identify, apply differential privacy, generate synthetic data, and audit everything** -- with formal privacy guarantees, not just "we removed the names."

## Quick Start

```bash
pip install healthpipe
```

### Ingest from any source

```python
import asyncio
import healthpipe as hp

async def main():
    dataset = await hp.ingest([
        hp.FHIRSource(url="https://fhir.hospital.org/R4"),
        hp.CSVSource(path="./patient_data.csv", mapping="auto"),
        hp.HL7v2Source(path="./hl7_messages/*.hl7"),
    ])

    print(dataset.patients.count())
    print(dataset.observations.count())

asyncio.run(main())
```

### De-identify with HIPAA Safe Harbor

```python
import asyncio
import healthpipe as hp

async def main():
    # ... (ingest dataset as above) ...

    deidentified = await hp.deidentify(
        dataset,
        method="safe_harbor",
        date_shift=True,
        date_shift_salt="YOUR-SECRET-SALT-HERE",  # required, keep secret
        date_shift_range=(-365, 365),
        llm_verification=True,
        llm_model="claude-haiku-4-5",
    )

    print(deidentified.audit_log)
    # [PHI Removed] PATIENT_NAME "a1b2c3..." -> "[PATIENT_NAME]" (NER, confidence: 0.99)
    # [PHI Removed] SSN "d4e5f6..." -> "[SSN]" (PATTERN, confidence: 1.00)
    # [Date Shifted] offset=142 days (DATE_SHIFT)

asyncio.run(main())
```

### Differentially private statistics

```python
stats = hp.private_stats(
    deidentified,
    epsilon=1.0,
    queries=[
        hp.Count(field="patient", group_by="diagnosis"),
        hp.Mean(field="lab_results.glucose", group_by="age_group"),
        hp.Histogram(field="medications", bins=20),
    ],
)
print(f"Privacy budget remaining: {stats.budget_remaining}")
print(stats.results["count:patient|group_by:diagnosis"])
```

### Synthetic data generation

```python
import asyncio
import healthpipe as hp

async def main():
    # ... (de-identify dataset as above) ...

    synthetic = await hp.synthesize(
        deidentified,
        n_patients=10_000,
        method="gaussian_copula",
        validate=True,
    )

    utility = hp.evaluate_utility(synthetic, deidentified)
    print(f"Statistical fidelity: {utility.fidelity:.2%}")
    print(f"Re-identification risk: {utility.reidentification_risk:.6f}")

asyncio.run(main())
```

## Architecture

```mermaid
graph TD
    A[Data Sources] --> B[Ingest Layer]
    B --> |FHIR R4| C[Unified Schema]
    C --> D[De-identification Engine]

    D --> D1[Layer 1: NER]
    D --> D2[Layer 2: Pattern Matching]
    D --> D3[Layer 3: Date Shifting]
    D --> D4[Layer 4: LLM Verification]

    D1 & D2 & D3 & D4 --> E[Privacy Layer]

    E --> E1[Differential Privacy]
    E --> E2[k-Anonymity / l-Diversity]

    E1 & E2 --> F[Synthetic Generator]
    F --> G[Validation]

    D & E & F & G --> H[Audit & Compliance]

    subgraph Sources
        A1[FHIR R4]
        A2[HL7v2]
        A3[CSV]
        A4[PDF/OCR]
    end

    A1 & A2 & A3 & A4 --> B
```

### De-identification: Four Layers

| Layer | Technique | What It Catches |
|-------|-----------|-----------------|
| 1 | **Named Entity Recognition** | Patient names, locations, organizations |
| 2 | **Pattern Matching** | SSN, phone, email, MRN, IP, ZIP codes |
| 3 | **Date Shifting** | All dates (preserves intervals between events) |
| 4 | **LLM Verification** | Context-dependent identifiers regex misses |

### HIPAA Safe Harbor Coverage

| HIPAA Requirement | healthpipe Feature |
|---|---|
| Safe Harbor (18 identifiers) | Multi-layer de-identification engine |
| Minimum Necessary | Configurable field access controls |
| Audit Controls | Comprehensive JSON audit logging |
| Data Integrity | SHA-256 checksums on all transformations |
| Transmission Security | TLS enforcement on all data sources |
| Breach Notification | Re-identification risk scoring |

## Installation

**Core (no heavy ML dependencies):**

```bash
pip install healthpipe
```

**With optional components:**

```bash
# NLP (spaCy for clinical NER)
pip install healthpipe[nlp]

# OCR (PDF extraction)
pip install healthpipe[ocr]

# Differential privacy (OpenDP)
pip install healthpipe[dp]

# Synthetic data (SDV/CTGAN)
pip install healthpipe[synthetic]

# LLM verification (Anthropic)
pip install healthpipe[llm]

# Everything
pip install healthpipe[all]
```

## CLI

```bash
# Ingest a CSV file
healthpipe ingest ./patients.csv --format csv -o dataset.json

# De-identify
healthpipe deidentify dataset.json -o deidentified.json --audit-log audit.json

# Generate synthetic data
healthpipe synthesize deidentified.json -o synthetic.json --n-patients 5000

# View audit log
healthpipe audit audit.json --format summary
```

## Project Structure

```
src/healthpipe/
    __init__.py          # Public API
    pipeline.py          # Main orchestration
    cli.py               # Click CLI
    exceptions.py        # Custom exception hierarchy
    ingest/
        fhir.py          # FHIR R4 source
        hl7v2.py         # HL7v2 parser
        csv_mapper.py    # CSV -> FHIR mapping
        pdf_ocr.py       # PDF extraction (OCR)
        schema.py        # Unified internal schema
    deidentify/
        ner.py           # Clinical NER (spaCy + fallback)
        patterns.py      # Regex pattern matching
        date_shift.py    # Date shifting
        llm_verify.py    # LLM verification pass
        safe_harbor.py   # HIPAA Safe Harbor orchestration
    privacy/
        differential.py  # Laplace / Gaussian mechanisms
        k_anonymity.py   # k-Anonymity / l-Diversity
        budget.py        # Privacy budget tracking
    synthetic/
        generator.py     # Gaussian copula / CTGAN
        validator.py     # Re-identification risk testing
        utility.py       # Utility evaluation
    audit/
        logger.py        # Structured audit logging
        lineage.py       # Data lineage tracking
        compliance.py    # Compliance report generation
```

## Development

```bash
# Clone and install
git clone https://github.com/sushaan-k/healthpipe.git
cd healthpipe
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=healthpipe

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/healthpipe/
```

## Technical Stack

| Component | Library |
|-----------|---------|
| Data models | `pydantic` v2 |
| FHIR/HTTP | `httpx` (async) |
| Clinical NER | `spaCy` (optional) |
| OCR | `pytesseract` (optional) |
| Differential privacy | `opendp` (optional) |
| Synthetic data | `sdv` / `ctgan` (optional) |
| Numerics | `numpy`, `pandas`, `pyarrow` |
| CLI | `click` |

## Privacy Guarantees

- **Formal differential privacy** via Laplace and Gaussian mechanisms with composable epsilon budget tracking
- **k-Anonymity / l-Diversity** enforcement with automatic generalization and suppression
- **Re-identification risk validation** using Distance to Closest Record (DCR) metrics
- **HIPAA Safe Harbor** compliance with all 18 identifier types addressed (14 auto-detected; 3 require manual review -- see compliance report)
- **Audit trail** with SHA-256 hashed PHI values at construction time (the audit log itself cannot leak data; raw PHI is never stored by default)

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure `pytest`, `ruff check`, and `mypy` pass
5. Open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by [Sushaan Kandukoori](https://github.com/sushaan-k) | [vytus.health](https://vytus.health)
