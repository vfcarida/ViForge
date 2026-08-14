# ViForge Baseline Report

## 1. Execution Environment & Baseline State

- **Date:** 2026-08-14
- **Branch:** `audit-and-enhancement-20260814` (isolated from `main` @ `ba071227ececb60990adcb5178ffd94f4e8d7bfb`)
- **Python Runtime:** 3.12.10 (win32)
- **PyTorch:** 2.12.0+cpu
- **Transformers:** 5.9.0
- **PEFT:** 0.19.1
- **TRL:** 1.5.1
- **Pytest:** 9.0.3

---

## 2. Baseline Command Executions

### `V-001`: Test Suite Execution (`pytest -v`)
- **Command:** `pytest -v`
- **Exit Code:** `0`
- **Result:** **24 passed** out of 24 tests across unit, contract, integration, smoke, and e2e suites.
- **Duration:** 21.81 seconds.

### `V-002`: System Diagnostics (`viforge doctor`)
- **Command:** `viforge doctor`
- **Exit Code:** `0`
- **Result:** CPU physical: 4, logical: 8, RAM total: 15.89 GB, RAM available: 3.12 GB, Disk free: 18.72 GB.

### `V-003`: Static Linting (`ruff check src/ tests/`)
- **Command:** `ruff check src/ tests/`
- **Exit Code:** `1`
- **Result:** Found 45 unused imports across `src/` and `tests/` (`F-001`).

---

## 3. Baseline Audit Findings Matrix

| Finding ID | Title | Category | Severity | Confidence | Observed Evidence | Recommended Remediation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `F-001` | Unused imports across multiple modules | Code Hygiene / Lint | Low | High | `ruff check src/ tests/` output 45 unused imports | Run automated `ruff check --fix` and verify clean import statements |
| `F-002` | Missing `.gitignore` in root directory | Configuration | Medium | High | Transient build/cache `.pyc` and `.egg-info` files entered Git index | Added `.gitignore` and untracked build caches from Git index |
| `F-003` | Generic Exception handling in evaluation sandbox | Security / Reliability | Medium | High | `src/viforge/security/sandbox.py:60` catches generic `Exception` without structured error taxonomy | Add typed sandbox exceptions (`SandboxTimeoutError`, `SandboxExecutionError`, `SandboxMemoryError`) |
| `F-004` | Single-sample fallback in Wilson score calculations | Metrics / Statistical | Low | High | `src/viforge/metrics/statistical.py` defaults to normal approximation on edge cases | Ensure Wilson score interval handles $N=0$ and $N=1$ boundary conditions rigorously |
| `F-005` | Lack of detailed formal architecture and research docs | Documentation | Medium | High | `docs/` folder was missing from initial repository structure | Build complete `docs/` structure with research reports, ADRs, target architecture, and test strategy |
