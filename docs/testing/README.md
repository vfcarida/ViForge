# Testing Guide & Quality Assurance

ViForge enforces a rigorous multi-tier testing strategy ensuring determinism, hardware compatibility, and mathematical correctness across all post-training stages.

---

## Test Architecture Layers

```text
tests/
├── unit/            # Deterministic, non-networked unit tests (Schemas, Profiler, Preprocessing, Security, Cost, Pareto, Synthetic, Distributed, Orchestration)
├── contract/        # Abstract Base Class and Plugin interface conformance (TrainingMethod, EvalHarness)
├── integration/     # Pipeline dataflow, manifest verification, and adapter merging
├── smoke/           # System diagnostics doctor and CLI subcommand execution
└── e2e/             # Multi-stage specialization campaign execution
```

---

## Test Execution Commands

```bash
# Run full unit test suite
python -m pytest tests/unit/ -v

# Run contract, smoke, and integration test suites
python -m pytest tests/contract/ tests/smoke/ tests/integration/ -v

# Run end-to-end integration campaign
python -m pytest tests/e2e/ -v

# Run linting and static analysis
python -m ruff check src/ tests/
```

For more architectural details, see [Test Strategy](TEST_STRATEGY.md).
