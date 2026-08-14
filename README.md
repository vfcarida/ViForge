# ViForge — Forging Small Models into Specialists

> **ViForge** is a modular post-training experimentation platform and evaluation framework designed to systematically transform cost-efficient general-purpose language models into high-performing domain experts (starting with **DeepSeek V4 Pro** in **Software Engineering**), quantifying whether specialization justifies its training and operational cost on a multi-objective Pareto frontier.

---

## 1. The Core Research Hypothesis

> **"Can specialization improve the software-engineering capability of a cost-efficient model enough to make it a better capability-per-dollar choice than larger frontier alternatives?"**

ViForge evaluates this question through rigorous empirical Pareto-optimality analysis:

$$\text{Capability-per-Dollar} = \frac{\Delta \mathcal{S}_{\text{domain}} - \lambda \cdot \max(0, -\Delta \mathcal{S}_{\text{general}})}{\mathcal{C}_{\text{train}} + \mathcal{C}_{\text{infer}}(N_{\text{tokens}})}$$

```text
GENERALIST MODEL (e.g. DeepSeek V4 Pro)
       │
       ▼
   DATA / TASK SPECIFICATION (Immutable Manifest, MinHash LSH, Contamination Shield)
       │
       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                                ViForge                                 │
 │                                                                        │
 │  • Pre-Flight Resource Profiler (Analytical VRAM + 15% Headroom)       │
 │  • Supervised Fine-Tuning (SFT)                                        │
 │  • Parameter-Efficient Fine-Tuning (LoRA & QLoRA NF4 4-bit)            │
 │  • Continued Domain Pretraining (CPT / DAPT)                           │
 │  • Preference Optimization (DPO, KTO, GRPO with Unit-Test Rewards)     │
 │  • Synthetic Teacher Distillation & AST Rejection Sampling             │
 │  • Zero-Network Isolated Sandbox Execution                             │
 └────────────────────────────────────────────────────────────────────────┘
       │
       ▼
SPECIALIST MODEL & ZERO-OVERHEAD SAFETENSORS
       │
       ▼
MULTI-OBJECTIVE PARETO FRONTIER EVALUATION
(Domain Quality × Retention × Financial Cost × Latency × Memory)
```

---

## 2. Framework Architecture & Subsystems

```text
viforge/
├── src/
│   └── viforge/
│       ├── cli/                 # Typer CLI (doctor, validate, train, evaluate, compare, etc.)
│       ├── config/              # Pydantic v2 schemas, loaders, validation models
│       ├── models/              # ModelRegistry, HF loaders, target_module resolvers
│       ├── datasets/            # DatasetRegistry, Ingestion adapters, Governance manifests
│       ├── preprocessing/       # Normalization, MinHash LSH, Contamination shield, Sequence packer
│       ├── training/            # Pre-flight VRAM profiler, Cost estimator, Accelerate/FSDP/ZeRO
│       ├── methods/             # SFT, LoRA, QLoRA, CPT, DPO, KTO, GRPO, Synthetic SFT
│       ├── inference/           # Deterministic inference client (Seed pinning, HF/Mock backends)
│       ├── evaluation/          # Sandboxed harness (HumanEval+, MBPP+, LiveCodeBench, SWE-bench)
│       ├── metrics/             # Statistical deltas (Wilson CI, Bootstrap, Significance testing)
│       ├── cost/                # Financial model (Training, Generation, Storage, Inference)
│       ├── artifacts/           # Adapter merger (merge_and_unload), Safetensors exporter
│       ├── experiments/         # Stage DAG resolver, Resumable experiment runner
│       ├── analysis/            # Multi-objective Pareto frontier engine
│       ├── reporting/           # Markdown, HTML, JSON report generator
│       ├── aws/                 # S3 multipart sync, SageMaker runners, Resource cleanup
│       ├── security/            # TruffleHog secrets scanning, PII redaction, Sandboxing
│       └── utils/               # Rich logging, System diagnostics (doctor), SHA-256 hashing
├── configs/
│   ├── models/                  # deepseek_v4_pro.yaml, reference_bases.yaml
│   ├── datasets/                # swe_synthetic_python.yaml, dapt_code_corpus.yaml
│   ├── methods/                 # qlora.yaml, sft.yaml, cpt.yaml, dpo.yaml, grpo.yaml
│   ├── evaluation/              # standard_eval.yaml
│   └── experiments/             # deepseek_v4_pro_software_engineering.yaml
├── tests/
│   ├── unit/                    # Schemas, Profiler, Preprocessing, Security, Cost, Pareto
│   ├── contract/                # Plugin interfaces & ABC protocol tests
│   ├── integration/             # Pipeline integration & manifest integrity
│   ├── smoke/                   # CPU diagnostics & CLI smoke tests
│   └── e2e/                     # Full end-to-end campaign execution
├── docker/                      # Dockerfile, Dockerfile.cpu, Dockerfile.gpu
├── .github/workflows/           # CI/CD and GPU benchmark actions
├── examples/                    # Quickstart & custom benchmark examples
├── scripts/                     # Standalone doctor, decontaminate, and cleanup scripts
└── pyproject.toml
```

---

## 3. CLI Command Suite

```bash
# 1. Inspect environment, PyTorch, CUDA, RAM, and dependencies
viforge doctor

# 2. Validate experiment YAML configurations against schemas
viforge validate configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 3. Prepare, deduplicate, and decontaminate datasets
viforge prepare-data configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 4. Profile VRAM, FLOPs, and estimated AWS costs before allocation
viforge train configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 5. Run baseline model benchmark evaluation
viforge baseline configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 6. Evaluate specialized checkpoints against benchmark suites
viforge evaluate configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 7. Compute statistical deltas & 95% Wilson confidence intervals
viforge compare configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 8. Compute multi-objective Pareto frontier
viforge analyze configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 9. Generate Markdown, HTML, and JSON reports
viforge report configs/experiments/deepseek_v4_pro_software_engineering.yaml

# 10. Execute complete end-to-end specialization campaign
viforge run configs/experiments/deepseek_v4_pro_software_engineering.yaml

# Inspection commands
viforge list-methods
viforge list-datasets
viforge list-evaluators
```

---

## 4. DeepSeek V4 Pro Reference Experiment Campaign

| Stage ID | Specialization Strategy | Target Modules | Est. VRAM / GPU | Est. Cost (AWS) | Primary Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `stage_1_cpt` | Domain-Adaptive Pretraining | All weights | 33.1 GB | \$48.00 | Loss & Domain PPL |
| `stage_2_qlora_sft` | QLoRA 4-bit (NF4, r=64, $\alpha$=128) | `all-linear` (7 projections) | 17.5 GB | \$144.00 | HumanEval+ / SWE-bench |
| `stage_3_dpo` | Direct Preference Optimization | Adapter layers | 36.8 GB | \$48.00 | Preference Margin |

---

## 5. Running Tests

```bash
# Run all unit, contract, integration, smoke, and e2e tests
pytest tests/ -v
```

---

## 6. License

ViForge is released under the **MIT License**.
