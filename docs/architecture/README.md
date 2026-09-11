# ViForge Architectural Overview

ViForge is an enterprise post-training and evaluation framework specifically engineered to forge Small Language Models (SLMs) into domain specialists.

---

## High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion & Pre-Flight
        Config["Experiment Manifest (YAML)"] --> Loader["ConfigLoader / Pydantic v2"]
        Loader --> Profiler["Pre-Flight Resource & Distributed Profiler"]
        Profiler -->|OOM Risk Detected| Remediation["Actionable Remediation Advice"]
        Profiler -->|Feasible| DataPipeline["Data Governance & Synthesis Pipeline"]
    end

    subgraph Data & Synthesis
        DataPipeline --> Scanner["PII & Secret Scanner"]
        Scanner --> Dedup["MinHash LSH Deduplication"]
        Dedup --> Contam["10-gram Contamination Shield"]
        Contam --> Evol["Evol-Instruct Mutation Engine"]
        Evol --> SelfPlay["Self-Play Preference Curating (AST & Length-Normalized)"]
        SelfPlay --> Pack["Sequence Context Packer"]
    end

    subgraph Execution & Orchestration
        Pack --> Orchestrator{"Execution Target"}
        Orchestrator -->|Local / Single-Node| LocalRun["Local PyTorch Engine"]
        Orchestrator -->|HPC Slurm| SlurmGen["Slurm .sbatch Generator (torchrun / Apptainer)"]
        Orchestrator -->|Kubernetes| RayGen["KubeRay RayJob Manifest Generator"]
        Orchestrator -->|AWS Cloud| SageMakerRun["AWS SageMaker Runner (DLC / S3 Sync)"]
    end

    subgraph Training Stages
        LocalRun & SlurmGen & RayGen & SageMakerRun --> DAG["Multi-Stage DAG Resolver"]
        DAG --> CPT["Domain Continued Pretraining (CPT)"]
        DAG --> SFT["Supervised Fine-Tuning (SFT / QLoRA)"]
        DAG --> Align["Reference-Free Preference Alignment (ORPO / SimPO)"]
    end

    subgraph Evaluation & Pareto Frontier
        Align --> EvalHarness["Multi-Benchmark Evaluation Engine"]
        EvalHarness --> Stats["Wilson 95% Confidence Intervals & Significance"]
        Stats --> CostModel["Hardware & Cloud Cost Modeling"]
        CostModel --> ParetoEngine["Multi-Objective Pareto Engine (Non-Dominated Sort)"]
        ParetoEngine --> ModelCard["Hugging Face Model Card & Lineage Generator"]
        ModelCard --> Studio["ViForge Streamlit Studio"]
    end
```

---

## Core Subsystems

| Subsystem | Key Modules | Primary Responsibilities |
| :--- | :--- | :--- |
| **[Distributed Profiler](distributed_training.md)** | `viforge.training.profiler`, `viforge.training.distributed` | Analytical VRAM estimation (DDP, ZeRO-1/2/3, FSDP2, CPU Offload), Accelerate YAML and DeepSpeed JSON auto-generation. |
| **[Synthetic Data & Self-Play](synthetic_data.md)** | `viforge.methods.synthetic`, `viforge.preprocessing.contamination` | Evol-Instruct prompt mutation (deepen, broaden, reasoning, concretize), AST code verification, length-bias mitigation, contamination prevention. |
| **[Cluster Orchestration](orchestration.md)** | `viforge.orchestration.slurm`, `viforge.orchestration.ray`, `viforge.aws.sagemaker` | Slurm HPC `.sbatch` generation, Kubernetes KubeRay `RayJob` YAML generation, and AWS SageMaker managed execution. |
| **[Unified Pareto Engine](unified_pareto.md)** | `viforge.reporting.pareto`, `viforge.reporting.model_card` | Non-dominated sorting, Wilson 95% confidence intervals, Hugging Face Hub standard `README.md` model cards, cryptographic lineage. |
| **[Reference-Free Alignment](../research/reference_free_alignment.md)** | `viforge.methods.orpo`, `viforge.methods.simpo` | Memory-efficient preference alignment without reference model overhead. |
