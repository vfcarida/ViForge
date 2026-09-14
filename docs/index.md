# ViForge: Forging Small Models into Domain Specialists

> A Systematic Post-Training, Fine-Tuning & Evaluation Framework for Open-Source Large Language Models.

---

## What is ViForge?

**ViForge** is an open-source, production-grade framework engineered to specialize open-weight foundation models (7B–70B parameters) into high-performing domain experts (e.g. software engineering, mathematics, scientific reasoning) with minimal compute footprint and zero catastrophic forgetting.

```
                  ┌─────────────────────────────────────────┐
                  │       Base Open-Weight Model            │
                  │   (Llama-3, DeepSeek, Qwen-2.5, etc.)   │
                  └────────────────────┬────────────────────┘
                                       │
                         ViForge Specialization Pipeline
                         (CPT → LoRA/QLoRA → DPO/GRPO/KTO)
                                       │
                  ┌────────────────────▼────────────────────┐
                  │       Domain Specialist Model           │
                  │  • Frontier-grade domain capability     │
                  │  • Retained general reasoning           │
                  │  • 80%+ lower inference serving cost    │
                  └─────────────────────────────────────────┘
```

---

## Core Pillars

1. **Multi-Stage Post-Training DAG Engine**:
   Compose Continual Pre-Training (CPT), Parameter-Efficient Fine-Tuning (LoRA / QLoRA), and Preference Alignment (DPO, KTO, GRPO) into a declarative directed acyclic graph.
2. **Pre-Flight Memory Profiler**:
   Compute deterministic, closed-form VRAM requirements across weights, KV cache, activation memory, and optimizer states before training starts.
3. **Automated Data Governance**:
   Banded Locality-Sensitive Hashing (LSH) deduplication, 13-gram benchmark decontamination, and automated credential/PII redaction.
4. **Statistical Rigor**:
   Paired McNemar significance testing, Wilson score 95% confidence intervals, and automated Pareto frontier analysis.
5. **Multi-Platform Deployment**:
   Export directly to GGUF (with Ollama Modelfiles), AWQ post-training quantization, or cloud execution on AWS SageMaker.

---

## Quickstart

```bash
# 1. Install ViForge
pip install -e .

# 2. Run pre-flight health check
viforge doctor

# 3. Choose your execution track:

# Track A: Instant 10-second offline simulation (zero GPU required)
viforge run configs/experiments/deepseek_v4_pro_software_engineering.yaml --mock

# Track B: Live real open-weights fine-tuning (Qwen 2.5 Coder 1.5B Instruct)
viforge run configs/experiments/qwen2.5_coder_1.5b_quickstart.yaml --live
```

---

## Documentation Navigation

- [Architecture Overview](architecture/README.md)
- [Case Studies: DeepSeek V4 Pro × SWE](case-studies/deepseek-v4-pro-swe.md)
- [Testing & Quality Assurance](testing/README.md)
- [Audit & Evolution Plan](audit/README.md)
