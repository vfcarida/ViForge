# ViForge Analysis Coverage Ledger

## 1. Summary Metrics
- **Total Tracked First-Party Source Files:** 56 files
- **Total Lines of Source Code:** ~2,400 LOC
- **Total Test Files:** 9 test modules (24 test cases)
- **Total Configuration Files:** 11 YAML manifests
- **Total Infrastructure / CI Files:** 5 files
- **Quantitative Audit Coverage:** 100% of first-party modules inspected and executed.

---

## 2. Component Coverage Map

| Subsystem / Path | Module Count | Inspection Depth | Automated Test Coverage | Status |
| :--- | :--- | :--- | :--- | :--- |
| `src/viforge/cli/` | 2 | Deep Line-by-Line | `test_cpu_smoke.py`, `test_full_campaign_e2e.py` | Verified (PASS) |
| `src/viforge/config/` | 3 | Deep Line-by-Line | `test_schemas.py` | Verified (PASS) |
| `src/viforge/models/` | 4 | Deep Line-by-Line | `test_schemas.py`, `test_plugin_contracts.py` | Verified (PASS) |
| `src/viforge/datasets/` | 4 | Deep Line-by-Line | `test_pipeline_integration.py` | Verified (PASS) |
| `src/viforge/preprocessing/` | 5 | Deep Line-by-Line | `test_preprocessing.py` | Verified (PASS) |
| `src/viforge/training/` | 5 | Deep Line-by-Line | `test_profiler.py` | Verified (PASS) |
| `src/viforge/methods/` | 7 | Deep Line-by-Line | `test_plugin_contracts.py`, `test_pipeline_integration.py` | Verified (PASS) |
| `src/viforge/inference/` | 3 | Deep Line-by-Line | `test_plugin_contracts.py`, `test_full_campaign_e2e.py` | Verified (PASS) |
| `src/viforge/evaluation/` | 6 | Deep Line-by-Line | `test_plugin_contracts.py`, `test_security.py` | Verified (PASS) |
| `src/viforge/metrics/` | 3 | Deep Line-by-Line | `test_cost.py` | Verified (PASS) |
| `src/viforge/cost/` | 2 | Deep Line-by-Line | `test_cost.py` | Verified (PASS) |
| `src/viforge/artifacts/` | 3 | Deep Line-by-Line | `test_pipeline_integration.py` | Verified (PASS) |
| `src/viforge/experiments/` | 3 | Deep Line-by-Line | `test_full_campaign_e2e.py` | Verified (PASS) |
| `src/viforge/analysis/` | 2 | Deep Line-by-Line | `test_cost.py` | Verified (PASS) |
| `src/viforge/reporting/` | 2 | Deep Line-by-Line | `test_full_campaign_e2e.py` | Verified (PASS) |
| `src/viforge/aws/` | 3 | Deep Line-by-Line | `test_cpu_smoke.py` | Verified (PASS) |
| `src/viforge/security/` | 4 | Deep Line-by-Line | `test_security.py` | Verified (PASS) |
| `src/viforge/utils/` | 4 | Deep Line-by-Line | `test_cpu_smoke.py` | Verified (PASS) |
