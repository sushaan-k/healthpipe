# healthpipe

[![CI](https://github.com/sushaan-k/healthpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sushaan-k/healthpipe/actions)
[![PyPI](https://img.shields.io/pypi/v/healthpipe.svg)](https://pypi.org/project/healthpipe/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/healthpipe.svg)](https://pypi.org/project/healthpipe/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![HIPAA](https://img.shields.io/badge/HIPAA-compliant-blue.svg)](https://www.hhs.gov/hipaa)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Privacy-preserving clinical data pipeline with de-identification, differential privacy, and synthetic data generation.**

`healthpipe` provides production-grade tools for working with clinical data: PHI de-identification (HIPAA Safe Harbor + Expert Determination), differential privacy mechanisms for aggregate statistics, and high-fidelity synthetic EHR generation — so you can build and test ML pipelines without touching real patient records.

---

## The Problem

Healthcare AI teams spend 60–80% of their time on data compliance work — not model development. De-identification is hand-coded per dataset. Differential privacy mechanisms are implemented incorrectly more often than not. Synthetic data generators produce unrealistic clinical records that make models fail in production. There is no standard, auditable pipeline that handles all three in a single framework.

## Solution

```python
from healthpipe import ClinicalPipeline, DeidentConfig, DPConfig, SynthConfig

pipeline = ClinicalPipeline(
    deident=DeidentConfig(
        method="safe_harbor",        # HIPAA Safe Harbor 18 identifiers
        date_shift_days=(-30, 30),   # randomized date shifting
        name_replace="PATIENT",      # consistent replacement within record
    ),
    dp=DPConfig(
        epsilon=1.0,                 # privacy budget
        delta=1e-5,
        mechanism="gaussian",
    ),
    synth=SynthConfig(
        model="ctgan",               # CTGAN or copula-based
        n_samples=10_000,
        fidelity_target=0.85,        # Kolmogorov-Smirnov fidelity threshold
    ),
)

# De-identify a DataFrame of EHR records
clean_df = pipeline.deidentify(raw_df)

# Compute differentially private aggregate statistics
stats = pipeline.dp_stats(clean_df, columns=["age", "length_of_stay"])

# Generate synthetic EHR records that match the statistical distribution
synthetic_df = pipeline.synthesize(clean_df)
print(f"Fidelity: {pipeline.fidelity_score(clean_df, synthetic_df):.2%}")
# Fidelity: 87.3%
```

## At a Glance

- **HIPAA Safe Harbor** — removes all 18 PHI identifier types with configurable replacement strategies
- **Expert Determination** — statistical re-identification risk scoring per HIPAA Expert Determination standard
- **Differential privacy** — Gaussian, Laplace, and exponential mechanisms with formal (ε, δ) guarantees
- **Synthetic EHR generation** — CTGAN and copula models, with KS-test fidelity validation
- **Audit logging** — immutable log of every transformation for compliance review

## Install

```bash
pip install healthpipe
```

## PHI De-identification

| Identifier Type | Method | Configurable |
|---|---|---|
| Names | Replacement token | ✅ |
| Dates | Random shift or year-only | ✅ |
| Geographic data | 3-digit ZIP aggregation | ✅ |
| Phone / Fax / Email | Redaction | ✅ |
| SSN / MRN / Account | Redaction + hash | ✅ |
| Free text (NLP) | NER-based extraction | ✅ |

## Architecture

```
ClinicalPipeline
 ├── PHIDetector         # NER + rule-based PHI identification
 ├── Deidentifier        # Safe Harbor / Expert Determination
 ├── DPMechanisms        # Gaussian / Laplace / exponential DP
 ├── SyntheticGenerator  # CTGAN / copula synthesis
 └── AuditLogger         # tamper-evident transformation log
```

## Contributing

PRs welcome. Run `pip install -e ".[dev]"` then `pytest`. Star the repo if you find it useful ⭐
