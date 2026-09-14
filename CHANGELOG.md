# Changelog

All notable changes to **ViForge** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-12

### Added
* Real open-weights live quickstart configuration for `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`configs/experiments/qwen2.5_coder_1.5b_quickstart.yaml`).
* Evol-Instruct synthetic data mutation strategies (`deepen_constraints`, `broaden_domain`, `add_reasoning_steps`, `concretize`) and self-play preference curating for reference-free alignment (ORPO & SimPO).
* Preventive benchmark contamination screening for synthetic datasets.
* Multi-GPU distributed memory profiler modeling DDP, ZeRO-1/2/3, FSDP2, and CPU offloading with automated Accelerate YAML and DeepSpeed JSON exports.
* Enterprise cluster orchestration generators: Slurm HPC multi-node `.sbatch` (torchrun, Apptainer/Singularity) and Kubernetes KubeRay `RayJob` manifests.
* Automated Hugging Face Hub Model Card generator with YAML frontmatter, hyperparameters, Wilson 95% CI deltas, Pareto badges, and ViPym compression recommendations.
* Streamlit Studio web application expanded to 7 interactive tabs.
* Standard Python entry-point plugin architecture (`viforge.methods` and `viforge.evaluators`).
* Automated release workflow with Twine metadata validation and PyPI Trusted Publishing via GitHub OIDC.
* Unit test for real LoRA autograd backward pass and parameter delta verification.

### Changed
* Hardened Mypy static type checking across all 85 source files by re-enabling `arg-type`, `assignment`, `call-arg`, and `misc`.
* Replaced loose type ignores in `sandbox.py`, `callbacks.py`, `backends.py`, and `preparer.py`.

---

## [0.1.0] - 2026-08-14

### Added
* Complete `src/` layout framework with 18 modular subsystems.
* Production Typer CLI with full command suite (`doctor`, `validate`, `prepare-data`, `baseline`, `train`, `evaluate`, `compare`, `analyze`, `report`, `run`, `list-*`).
* Pre-flight VRAM and Cost Profiler with analytical failure prevention and 15% safety headroom.
* Plugin-based Training Taxonomy supporting SFT, LoRA, QLoRA (NF4), CPT (DAPT), DPO, KTO, GRPO, and Synthetic Distillation.
* Data Governance, MinHash LSH deduplication, 10-gram contamination detection, and TruffleHog secrets scanning.
* Isolated Code Execution Sandbox with CPU/Memory limits and zero-network isolation.
* Multi-objective Pareto Frontier Engine optimizing Domain Quality vs General Retention vs Dollar Cost.
* Reference DeepSeek V4 Pro Software Engineering master experiment campaign configuration.
* Automated CI/CD workflows and CPU/GPU Docker containers.
