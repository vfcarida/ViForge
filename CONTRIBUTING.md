# Contributing to ViForge

Thank you for your interest in contributing to **ViForge**! We welcome contributions ranging from new post-training method plugins to custom benchmark suites and bug fixes.

---

## 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/vfcarida/ViForge.git
cd ViForge

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e .[dev]
```

---

## 2. Coding Standards & PR Guidelines

* **Code Formatting:** We use `ruff` for linting and formatting. Run `ruff check src/ tests/` before submitting a PR.
* **Testing:** All new features must include unit or integration tests under `tests/`. Run `pytest tests/ -v`.
* **Reproducibility:** Seed pinning and zero data leakage standards must be preserved in all evaluation adapters.
* **Typing:** Type annotations are required on all public functions and classes.

---

## 3. Adding a New Training Method Plugin

To add a new specialization method (e.g. `SimPO` or `ORPO`):

1. Inherit from `BaseTrainingMethod` in `src/viforge/methods/`.
2. Implement `prepare_model()` and `execute_stage()`.
3. Register your class with `method_registry.register("your_method_name", YourMethodClass)`.
4. Add a preset config under `configs/methods/`.
5. Add unit tests verifying execution.

---

## 4. Adding a New Evaluation Benchmark

To add a benchmark:

1. Create a class implementing the evaluator protocol with a `.evaluate(inference_backend, output_dir, sampling_params)` method.
2. Register the suite in `src/viforge/evaluation/suites/__init__.py`.
3. Ensure execution is isolated in `ExecutionSandbox`.
