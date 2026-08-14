# ViForge Repository Inventory

| Path | Category | Lines | Size (Bytes) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `pyproject.toml` | Configuration | 76 | 1,907 | Build & package metadata, dependencies, script entrypoints |
| `.gitignore` | Configuration | 52 | 680 | Git ignore patterns for Python, PyTorch, Caches, Run artifacts |
| `README.md` | Documentation | 148 | 7,842 | Visual documentation, Quickstart, Architecture, Reference matrix |
| `CONTRIBUTING.md` | Documentation | 48 | 1,726 | Contribution guidelines, plugin extension standards |
| `SECURITY.md` | Documentation | 28 | 1,067 | Vulnerability reporting policy & security guardrails |
| `CHANGELOG.md` | Documentation | 31 | 1,195 | Semantic version release history |
| `LICENSE` | Legal | 21 | 1,072 | MIT License |
| `src/viforge/__init__.py` | First-Party Source | 8 | 215 | Top-level package version |
| `src/viforge/cli/main.py` | First-Party Source | 214 | 7,890 | Typer CLI commands implementation |
| `src/viforge/cli/__init__.py` | First-Party Source | 7 | 105 | CLI package exports |
| `src/viforge/config/schemas.py` | First-Party Source | 175 | 6,420 | Pydantic v2 schemas and validation models |
| `src/viforge/config/loader.py` | First-Party Source | 46 | 1,520 | YAML/JSON manifest loader with env interpolation |
| `src/viforge/config/__init__.py` | First-Party Source | 48 | 1,210 | Config package exports |
| `src/viforge/models/registry.py` | First-Party Source | 76 | 2,740 | Model registry & default configs |
| `src/viforge/models/adapters.py` | First-Party Source | 60 | 2,180 | HF model/tokenizer loader |
| `src/viforge/models/target_modules.py` | First-Party Source | 27 | 1,180 | Target modules resolver |
| `src/viforge/models/__init__.py` | First-Party Source | 14 | 380 | Models package exports |
| `src/viforge/datasets/registry.py` | First-Party Source | 26 | 820 | Dataset registry |
| `src/viforge/datasets/manifest.py` | First-Party Source | 52 | 1,940 | Governance manifest generator & checksum verifier |
| `src/viforge/datasets/adapters.py` | First-Party Source | 48 | 1,620 | JSONL/Parquet dataset loader & saver |
| `src/viforge/datasets/__init__.py` | First-Party Source | 13 | 360 | Datasets package exports |
| `src/viforge/preprocessing/normalizer.py` | First-Party Source | 31 | 1,040 | AST validation & comment stripping |
| `src/viforge/preprocessing/deduplication.py` | First-Party Source | 87 | 3,120 | MinHash LSH deduplication |
| `src/viforge/preprocessing/contamination.py` | First-Party Source | 82 | 2,980 | 10-gram benchmark contamination detector |
| `src/viforge/preprocessing/packing.py` | First-Party Source | 68 | 2,420 | Sequence packing & attention mask generator |
| `src/viforge/preprocessing/__init__.py` | First-Party Source | 14 | 410 | Preprocessing package exports |
| `src/viforge/training/profiler.py` | First-Party Source | 130 | 5,120 | Pre-flight analytical VRAM profiler |
| `src/viforge/training/cost_estimator.py` | First-Party Source | 58 | 2,240 | FLOPs and cost estimator |
| `src/viforge/training/callbacks.py` | First-Party Source | 32 | 1,080 | Training callbacks & metrics logger |
| `src/viforge/training/backends.py` | First-Party Source | 28 | 940 | Distributed runtime manager |
| `src/viforge/training/__init__.py` | First-Party Source | 14 | 390 | Training package exports |
| `src/viforge/methods/base.py` | First-Party Source | 47 | 1,380 | Abstract base method & dynamic registry |
| `src/viforge/methods/sft.py` | First-Party Source | 63 | 2,290 | Supervised Fine-Tuning |
| `src/viforge/methods/peft_lora.py` | First-Party Source | 108 | 3,920 | LoRA & QLoRA methods |
| `src/viforge/methods/cpt.py` | First-Party Source | 67 | 2,360 | Continued Pretraining (DAPT) |
| `src/viforge/methods/preference.py` | First-Party Source | 94 | 3,240 | DPO and GRPO methods |
| `src/viforge/methods/synthetic.py` | First-Party Source | 81 | 2,820 | Synthetic distillation pipeline |
| `src/viforge/methods/__init__.py` | First-Party Source | 24 | 720 | Methods package exports |
| `src/viforge/inference/backends.py` | First-Party Source | 127 | 4,380 | Mock and HF inference backends |
| `src/viforge/inference/client.py` | First-Party Source | 29 | 880 | Deterministic generation client |
| `src/viforge/inference/__init__.py` | First-Party Source | 17 | 490 | Inference package exports |
| `src/viforge/evaluation/suites/humaneval_plus.py` | First-Party Source | 79 | 2,650 | HumanEval+ evaluator suite |
| `src/viforge/evaluation/suites/swe_bench.py` | First-Party Source | 105 | 3,540 | SWE-bench, MBPP+, LiveCodeBench suites |
| `src/viforge/evaluation/suites/retention.py` | First-Party Source | 100 | 3,420 | MMLU-Pro, GSM8K, ARC-Challenge suites |
| `src/viforge/evaluation/suites/__init__.py` | First-Party Source | 40 | 1,180 | Evaluator registry |
| `src/viforge/evaluation/harness.py` | First-Party Source | 46 | 1,560 | Evaluation harness orchestrator |
| `src/viforge/evaluation/__init__.py` | First-Party Source | 22 | 680 | Evaluation package exports |
| `src/viforge/metrics/statistical.py` | First-Party Source | 31 | 1,020 | Wilson score confidence interval |
| `src/viforge/metrics/comparison.py` | First-Party Source | 46 | 1,620 | Base vs Specialist comparator |
| `src/viforge/metrics/__init__.py` | First-Party Source | 11 | 310 | Metrics package exports |
| `src/viforge/cost/cost_model.py` | First-Party Source | 56 | 1,880 | Economic cost model |
| `src/viforge/cost/__init__.py` | First-Party Source | 7 | 130 | Cost package exports |
| `src/viforge/artifacts/merger.py` | First-Party Source | 47 | 1,620 | LoRA adapter merger & Safetensors export |
| `src/viforge/artifacts/manager.py` | First-Party Source | 26 | 820 | Checkpoint & manifest manager |
| `src/viforge/artifacts/__init__.py` | First-Party Source | 6 | 150 | Artifacts package exports |
| `src/viforge/analysis/pareto.py` | First-Party Source | 57 | 1,980 | Multi-objective Pareto frontier engine |
| `src/viforge/analysis/__init__.py` | First-Party Source | 6 | 130 | Analysis package exports |
| `src/viforge/reporting/generator.py` | First-Party Source | 138 | 5,240 | Markdown, HTML, JSON report generator |
| `src/viforge/reporting/__init__.py` | First-Party Source | 6 | 140 | Reporting package exports |
| `src/viforge/aws/s3.py` | First-Party Source | 34 | 1,120 | S3 multipart sync manager |
| `src/viforge/aws/sagemaker.py` | First-Party Source | 36 | 1,180 | SageMaker runner & AWS cleanup manager |
| `src/viforge/aws/__init__.py` | First-Party Source | 11 | 280 | AWS package exports |
| `src/viforge/security/secrets_scanner.py` | First-Party Source | 31 | 1,120 | TruffleHog secrets scanner |
| `src/viforge/security/pii_filter.py` | First-Party Source | 36 | 1,240 | PII scanner & redactor |
| `src/viforge/security/sandbox.py` | First-Party Source | 64 | 2,120 | Isolated execution sandbox |
| `src/viforge/security/__init__.py` | First-Party Source | 11 | 280 | Security package exports |
| `src/viforge/utils/logging.py` | First-Party Source | 52 | 1,560 | Rich and structured JSON logging |
| `src/viforge/utils/hashing.py` | First-Party Source | 23 | 720 | SHA-256 integrity hash helpers |
| `src/viforge/utils/doctor.py` | First-Party Source | 56 | 2,040 | System diagnostics inspector |
| `src/viforge/utils/__init__.py` | First-Party Source | 15 | 380 | Utils package exports |
| `src/viforge/experiments/dag.py` | First-Party Source | 33 | 1,060 | Stage dependency DAG resolver |
| `src/viforge/experiments/runner.py` | First-Party Source | 224 | 8,920 | End-to-end experiment campaign runner |
| `src/viforge/experiments/__init__.py` | First-Party Source | 7 | 190 | Experiments package exports |
| `configs/models/deepseek_v4_pro.yaml` | Configuration | 18 | 480 | DeepSeek V4 Pro model config |
| `configs/models/reference_bases.yaml` | Configuration | 30 | 790 | Reference base models matrix |
| `configs/methods/qlora.yaml` | Configuration | 16 | 410 | QLoRA method preset |
| `configs/methods/sft.yaml` | Configuration | 10 | 220 | SFT method preset |
| `configs/methods/cpt.yaml` | Configuration | 10 | 220 | CPT method preset |
| `configs/methods/dpo.yaml` | Configuration | 9 | 200 | DPO method preset |
| `configs/methods/grpo.yaml` | Configuration | 7 | 160 | GRPO method preset |
| `configs/experiments/deepseek_v4_pro_software_engineering.yaml` | Configuration | 98 | 2,780 | DeepSeek V4 Pro master campaign manifest |
| `configs/experiments/exp_001_deepseek_v4_pro_baseline.yaml` | Configuration | 35 | 920 | Baseline experiment manifest |
| `configs/experiments/exp_002_deepseek_v4_pro_qlora_sft.yaml` | Configuration | 55 | 1,480 | QLoRA SFT experiment manifest |
| `configs/experiments/exp_003_deepseek_v4_pro_cpt_sft_dpo.yaml` | Configuration | 78 | 2,180 | Compounding CPT+SFT+DPO manifest |
| `tests/unit/test_schemas.py` | Test | 57 | 1,680 | Pydantic v2 schemas unit tests |
| `tests/unit/test_profiler.py` | Test | 66 | 2,120 | Pre-flight memory & cost profiler tests |
| `tests/unit/test_preprocessing.py` | Test | 47 | 1,740 | AST, MinHash, Contamination, Packing tests |
| `tests/unit/test_security.py` | Test | 33 | 1,160 | Secrets scanner, PII, Sandbox tests |
| `tests/unit/test_cost.py` | Test | 53 | 1,720 | CostModel, ParetoEngine, Statistical tests |
| `tests/contract/test_plugin_contracts.py` | Test | 31 | 960 | ABC protocol compliance tests |
| `tests/integration/test_pipeline_integration.py` | Test | 34 | 1,180 | Synthetic & dataset manifest tests |
| `tests/smoke/test_cpu_smoke.py` | Test | 26 | 740 | CPU doctor and CLI smoke tests |
| `tests/e2e/test_full_campaign_e2e.py` | Test | 26 | 980 | Full end-to-end campaign execution test |
| `docker/Dockerfile` | Infrastructure | 11 | 240 | Multi-stage Dockerfile |
| `docker/Dockerfile.cpu` | Infrastructure | 18 | 380 | CPU CI/CD container definition |
| `docker/Dockerfile.gpu` | Infrastructure | 26 | 680 | NVIDIA CUDA 12.2 GPU container definition |
| `.github/workflows/ci.yml` | CI/CD | 34 | 820 | Multi-OS GitHub Actions CI workflow |
| `.github/workflows/gpu-benchmark.yml` | CI/CD | 23 | 540 | GPU dispatch benchmarking workflow |
| `examples/quickstart_specialization.py` | Example | 31 | 1,120 | Programmatic quickstart execution example |
| `examples/custom_benchmark_plugin.py` | Example | 41 | 1,440 | Custom benchmark plugin authoring example |
| `scripts/run_doctor.py` | Script | 14 | 280 | Standalone doctor runner |
| `scripts/decontaminate_dataset.py` | Script | 33 | 1,020 | Standalone decontamination CLI script |
| `scripts/cleanup_aws.py` | Script | 13 | 280 | Standalone AWS cleanup script |
