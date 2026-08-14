# ViForge Testing Strategy

## 1. Test Architecture Layers

```text
tests/
├── unit/            # Deterministic, non-networked unit tests (Schemas, Profiler, Preprocessing, Security, Cost, Pareto)
├── contract/        # Abstract Base Class and Plugin interface conformance
├── integration/     # Pipeline dataflow, manifest verification, and adapter merging
├── smoke/           # System diagnostics doctor and CLI subcommand execution
└── e2e/             # Multi-stage specialization campaign execution
```

---

## 2. Test Execution Commands

```bash
# Run full automated test suite with coverage
pytest tests/ -v --cov=src/viforge --cov-report=term-missing

# Run focused unit tests
pytest tests/unit/ -v

# Run contract and smoke tests
pytest tests/contract/ tests/smoke/ -v

# Run end-to-end integration campaign
pytest tests/e2e/ -v
```
