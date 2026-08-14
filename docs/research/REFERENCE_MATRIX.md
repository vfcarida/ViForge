# ViForge Research Reference Matrix

| Ref ID | Topic | Source Type | Title | Organization / Authors | Date / Version | Authority Level | Applicable Finding / Planned Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `R-001` | Parameter-Efficient Fine-Tuning | Paper & Repo | *LoRA: Low-Rank Adaptation of Large Language Models* | Hu et al. (Microsoft Research) | ICLR 2022 / PEFT 0.10+ | Tier 1 | `C-001`, `C-002` (LoRA adapter parametrization & rank tuning) |
| `R-002` | 4-bit Quantization | Paper & Repo | *QLoRA: Efficient Finetuning of Quantized LLMs* | Dettmers et al. (UW / Meta) | NeurIPS 2023 / bitsandbytes 0.43+ | Tier 1 | `C-001` (NF4 4-bit quantization & paged optimizers) |
| `R-003` | Direct Preference Optimization | Paper & Repo | *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* | Rafailov et al. (Stanford University) | NeurIPS 2023 / TRL 0.8+ | Tier 1 | `C-003` (DPO loss & beta calibration) |
| `R-004` | Group Relative Policy Optimization | Paper & Technical Report | *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | Shao et al. (DeepSeek-AI) | 2024 / TRL GRPO | Tier 1 | `C-003` (GRPO rollout sampling & unit-test rewards) |
| `R-005` | Code Benchmark Rigor | Paper & Benchmark | *EvalPlus: Rigorous Evaluation of LLM-Generated Code* | Jiawei Liu et al. (UIUC) | NeurIPS 2023 / EvalPlus 0.8+ | Tier 1 | `C-004` (HumanEval+ / MBPP+ test generation) |
| `R-006` | Live Code Benchmark & Contamination | Benchmark & Paper | *LiveCodeBench: Holistic and Contamination-Free Evaluation of LLMs for Code* | Jain et al. | 2024 | Tier 1 | `C-004` (Contamination detection & leakage prevention) |
| `R-007` | Repository-Level Software Engineering | Benchmark & Paper | *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* | Jimenez et al. (Princeton NLP) | ICLR 2024 | Tier 1 | `C-004` (Patch resolution & docker sandbox isolation) |
| `R-008` | General Capability Retention | Benchmark & Paper | *MMLU-Pro: A More Robust and Challenging Multi-Task Benchmark* | Wang et al. (TIGER Lab / Waterloo) | 2024 | Tier 1 | `C-004` (Catastrophic forgetting quantification) |
| `R-009` | MinHash LSH Deduplication | Reference Implementation | *BigCode: StarCoder Data Deduplication & PII Redaction* | BigCode Project (Hugging Face / ServiceNow) | 2023 | Tier 2 | `C-005` (MinHash 128 permutation hashes & TruffleHog scanner) |
| `R-010` | ML System Rigor & Test Score | Paper | *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction* | Breck et al. (Google Research) | IEEE BigData / SE4ML | Tier 1 | `C-006` (Pre-flight resource validation & fail-fast checks) |
| `R-011` | Application & LLM Security | Standard & Guidelines | *OWASP Top 10 for Large Language Model Applications* | OWASP Foundation | 2024 / v1.1 | Tier 1 | `C-007` (Sandboxing, prompt injection & secret leakage prevention) |
| `R-012` | Multi-Objective Pareto Optimization | Research & Text | *Multi-Objective Optimization using Evolutionary Algorithms* | Deb, K. (Wiley / IEEE) | Seminal | Tier 2 | `C-008` (Non-dominated sorting & Pareto front identification) |
