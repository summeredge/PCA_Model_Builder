## 2026-08-09 - Candidate confirmation payload tests

**Scope**
Project

**Context**
Web candidate confirmation that carries nested operation data.

**Rule**
When an API handler reads a nested operation payload, add an end-to-end test using that exact request shape and real data needed for boundary calculations.

**Rationale**
Testing only a top-level equivalent can hide a frontend/backend payload-layer mismatch.

**Related Files**
- src/pca_model_builder/web.py
- tests/test_web.py
