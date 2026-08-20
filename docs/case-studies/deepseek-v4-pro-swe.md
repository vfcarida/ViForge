# Case Study: DeepSeek V4 Pro × Software Engineering Specialization

> **Status:** Experiment design complete. Infrastructure validated (P01–P12).
> Quantitative results will be populated after first real training run.
> All reproduction steps below are exact and tested against ViForge `main`.

---

## 1. Hypothesis

> *DeepSeek V4 Pro (open-weight, extremely cheap) can be specialized for
> software engineering to close the performance gap with frontier models
> (GPT-4o, Claude 3.5 Sonnet) while maintaining low inference cost and
> acceptable general-capability retention.*

**Why this matters:** At $0.14 / 1M prompt tokens, DeepSeek V4 Pro is **~85×
cheaper** than GPT-4o. If domain specialization can bridge ≥ 50% of the
HumanEval+ gap, the cost-adjusted capability advantage becomes decisive for
engineering teams.

---

## 2. Experiment Design

### 2.1 Base Model

| Property | Value |
|---|---|
| Name | DeepSeek V4 Pro |
| HF Hub ID | `deepseek-ai/deepseek-v4-pro` |
| Fallback | `deepseek-ai/deepseek-coder-6.7b-base` |
| Parameters | 14.5 B total / 2.4 B active (MoE) |
| Context window | 8 192 tokens |
| Pricing | $0.14 / 1M prompt · $0.28 / 1M completion |
| License | MIT |

### 2.2 Techniques Under Comparison

Five techniques are trained in parallel from the same base model and evaluated
on identical benchmarks, enabling a clean Pareto comparison.

| ID | Technique | LoRA rank | Quantization | Min VRAM | Est. time | Est. cost |
|---|---|---|---|---|---|---|
| T1 | Continued Pre-Training (CPT) | — | BF16 | 2 × 40 GB | ~14 h | ~$207 |
| T2 | QLoRA SFT | r = 32 | NF4 4-bit | 1 × 24 GB | ~4 h | ~$30 |
| T3 | LoRA SFT | r = 16 | BF16 | 1 × 40 GB | ~6 h | ~$44 |
| T4 | LoRA SFT | r = 64 | BF16 | 2 × 80 GB | ~8 h | ~$119 |
| T5 | DPO (on T3 checkpoint) | r = 16 | BF16 | 1 × 40 GB | ~3 h | ~$22 |

> **Entry point for 24 GB GPU owners:** Run T2 (QLoRA r=32) first to validate
> the full pipeline before committing GPU-hours to T1/T4.

### 2.3 Training Data (P10 pipeline)

| Technique | Source dataset | Samples | Format |
|---|---|---|---|
| CPT (T1) | `bigcode/starcoderdata` (Python subset) | 50 000 files | `{text}` |
| SFT (T2, T3, T4) | `sahil2801/CodeAlpaca-20k` + `iamtarun/python_code_instructions_18k_alpaca` | 20 000 pairs | `{messages}` |
| DPO (T5) | `argilla/CodeFeedback-Filtered-Instruction` | 10 000 pairs | `{prompt, chosen, rejected}` |

All datasets are downloaded, quality-filtered, MinHash-deduplicated, and
formatted by the `viforge prepare-data` command (see §4 Reproduction Steps).

### 2.4 Evaluation Benchmarks

#### Domain Benchmarks (measure improvement)

| Benchmark | Problems | Metric | Success threshold |
|---|---|---|---|
| **HumanEval+** (bigcode-evaluation-harness) | 164 | pass@1 | ≥ 5% relative gain (Wilson p < 0.05) |
| **SWE-bench Lite** | 300 | resolved% | ≥ 5% relative gain |

#### Retention Benchmarks (verify no catastrophic forgetting)

| Benchmark | Problems | Metric | Failure threshold |
|---|---|---|---|
| **MMLU-Pro** | 1 000 (subset) | accuracy | > 10% degradation = fail |
| **GSM8K** | 1 319 | pass@1 (exact `####` match) | > 10% degradation = fail |
| **ARC-Challenge** | 1 172 | accuracy | > 10% degradation = fail |

### 2.5 Success Criteria

- **Domain:** ≥ 5% relative improvement on HumanEval+ (Wilson 95% CI lower bound > 0)
- **Retention:** All retention benchmarks stay within 10% of baseline (`score / base_score ≥ 0.90`)
- **Statistical:** p < 0.05 on domain improvement (binomial proportion test)
- **Reproducibility:** Same seed (42) + temperature (0.0) → identical results

---

## 3. Experiment Manifest

The full manifest is at:

```
configs/experiments/deepseek_v4_pro_software_engineering.yaml
```

The manifest encodes all 5 training stages as a DAG:

```
stage_1_cpt        (CPT, no dependency)
stage_2_qlora_sft_r32  (QLoRA r=32, no dependency)
stage_3_lora_sft_r16   (LoRA r=16, no dependency)
stage_4_lora_sft_r64   (LoRA r=64, no dependency)
stage_5_dpo            (DPO, depends_on: stage_3_lora_sft_r16)
```

The `ExperimentRunner` resolves execution order topologically, so stages
without dependencies can be parallelized across GPUs by the scheduler.

---

## 4. Reproduction Steps

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/vfcarida/ViForge.git
cd ViForge

# Install all dependencies
pip install -r requirements/dev.lock
pip install -e . --no-deps

# Verify environment
viforge doctor
```

**Required environment variables (optional, for observability):**

```bash
export WANDB_API_KEY="your-key"          # enables W&B metric streaming
export MLFLOW_TRACKING_URI="http://..."  # enables MLflow experiment tracking
export LLAMA_CPP_PATH="/path/to/llama.cpp"  # enables real GGUF export
```

### Step 1: Validate Manifest

```bash
viforge validate configs/experiments/deepseek_v4_pro_software_engineering.yaml
```

Expected output:
```
OK: Valid manifest for experiment deepseek_v4_pro_software_engineering_master
  • Model: DeepSeek V4 Pro (deepseek-ai/deepseek-v4-pro)
  • Stages: 5
  • Benchmarks: 2 domain, 3 retention
```

### Step 2: Prepare Datasets

```bash
# Download, filter, dedup, and format all training data
viforge prepare-data \
  --domain software_engineering \
  --output data/ \
  --max-samples 20000
```

This runs the full P10 pipeline and writes:
- `data/software_engineering_cpt.jsonl` — 50K code files for CPT
- `data/software_engineering_sft.jsonl` — 20K instruction pairs for SFT
- `data/software_engineering_dpo.jsonl` — 10K preference pairs for DPO

### Step 3: Baseline Evaluation

```bash
# Measure base model (no fine-tuning) on all benchmarks
viforge baseline \
  configs/experiments/deepseek_v4_pro_software_engineering.yaml \
  --work-dir runs/
```

This generates `runs/deepseek_v4_pro_software_engineering_master/eval_baseline/`
with per-benchmark JSON reports. **Record these numbers — they are the ground
truth for all delta computations.**

### Step 4: Entry-Point Training — QLoRA r=32 (24 GB GPU)

Run only the QLoRA stage first to validate the pipeline is working before
committing to larger GPU allocations:

```bash
# Edit manifest to pipeline with only stage_2_qlora_sft_r32, then:
viforge run \
  configs/experiments/deepseek_v4_pro_software_engineering.yaml \
  --work-dir runs/ \
  --live
```

> **Note:** `--live` uses the real HuggingFace backend instead of the mock.
> Omit `--live` (default `--mock`) for fast CI/smoke-test runs.

### Step 5: Full Campaign — All Techniques

```bash
# Full 5-technique experiment campaign (~35h on 2× A100-80GB)
viforge run \
  configs/experiments/deepseek_v4_pro_software_engineering.yaml \
  --work-dir runs/ \
  --live
```

The runner automatically:
1. Resolves DAG execution order (stage 5 waits for stage 3)
2. Calls `on_experiment_start()` → streams config to W&B / MLflow
3. Runs `on_stage_start()` / `on_stage_end()` per stage
4. Evaluates the specialized model on all benchmarks
5. Computes statistical deltas and Pareto frontier
6. Generates reports in `runs/.../reports/`

### Step 6: Statistical Comparison

```bash
# View confidence intervals and significance tests
viforge compare \
  configs/experiments/deepseek_v4_pro_software_engineering.yaml \
  --work-dir runs/
```

### Step 7: Pareto Analysis

```bash
# Identify which technique is Pareto-optimal
viforge analyze \
  configs/experiments/deepseek_v4_pro_software_engineering.yaml \
  --work-dir runs/
```

Expected output columns: Model/Stage · Domain Score · Retention · Cost (USD) ·
Cap/Dollar · Pareto Optimal.

### Step 8: Export Best Checkpoint to GGUF

```bash
# Merge adapter into base model first
# (adapter path from stage_3_lora_sft_r16 or whichever stage won Pareto)
ADAPTER=runs/deepseek_v4_pro_software_engineering_master/checkpoints/stage_3_lora_sft_r16/adapter

# Export to GGUF Q4_K_M (requires LLAMA_CPP_PATH set)
viforge export-gguf \
  $ADAPTER \
  --output-dir exports/deepseek-v4-pro-swe-specialist/ \
  --quant-type Q4_K_M \
  --system-prompt "You are a software engineering specialist. Write clean, tested, production-grade code."

# Serve locally with Ollama
ollama create deepseek-v4-pro-swe -f exports/deepseek-v4-pro-swe-specialist/Modelfile
ollama run deepseek-v4-pro-swe
```

---

## 5. Expected Results (Pre-Run Projection)

These are model-based projections from the ViForge Pareto Engine. They will be
replaced with real measured values after the training campaign completes.

> ⚠️ All values below are **projected estimates**, not measured results.

### 5.1 Domain Improvement Projection

| Technique | HumanEval+ (projected) | SWE-bench Lite (projected) | Training Cost |
|---|---|---|---|
| Base (no fine-tuning) | ~35% | ~8% | $0 |
| QLoRA r=32 (T2) | ~43–47% | ~10–12% | $30 |
| LoRA r=16 (T3) | ~45–50% | ~11–13% | $44 |
| LoRA r=64 (T4) | ~47–53% | ~12–15% | $119 |
| CPT + LoRA r=16 (T1→T3) | ~50–56% | ~13–17% | $251 |
| DPO on LoRA r=16 (T5) | ~48–54% | ~12–15% | $66 |

### 5.2 Retention Projection

All techniques are projected to keep retention degradation within the 10%
threshold on MMLU-Pro, GSM8K, and ARC-Challenge, since code-domain CPT/SFT
does not significantly shift factual or mathematical representations.

### 5.3 Pareto Frontier Projection

The expected Pareto-optimal techniques are:
- **QLoRA r=32** — optimal for cost-constrained use (lowest cost / gain ratio)
- **DPO on LoRA r=16** — optimal for quality-constrained use (highest alignment
  at moderate cost)
- **LoRA r=64** — may be dominated by DPO+LoRA unless raw pass@1 is the only
  metric

---

## 6. How to Update This Document with Real Results

After running the campaign, replace §5 with the actual measured values:

```bash
# The runner generates JSON, Markdown, and HTML reports
cat runs/deepseek_v4_pro_software_engineering_master/reports/summary.json | python -m json.tool

# Key fields to extract:
# .baseline_domain_score   → base HumanEval+ pass@1
# .specialized_domain_score → best technique HumanEval+ pass@1
# .domain_gain_pct         → relative improvement
# .retention_delta_pct     → retention change
# .verdict                 → framework conclusion string
# .pareto_frontier[]       → Pareto-optimal techniques
```

Fill in the results table and add:
1. Loss curves (copy from `runs/.../metrics/training_metrics.jsonl`)
2. W&B / MLflow run link
3. Confidence intervals from `viforge compare`
4. Hardware telemetry from stage `*_metrics.json` files

---

## 7. Limitations and Disclosures

- All evaluations use **temperature = 0.0, seed = 42** for full reproducibility.
  Stochastic results (temperature > 0) may differ.
- **No data leakage check** has been performed between training data and HumanEval
  test problems in this initial run. A decontamination scan (`viforge contamination-check`)
  should be added before publishing results.
- **SWE-bench execution** requires Docker sandbox for full isolation. The
  `sandbox_backend: subprocess` setting in the manifest is faster but less
  secure; switch to `docker` for publication-grade results.
- Base model weights (`deepseek-ai/deepseek-v4-pro`) availability is subject to
  DeepSeek's distribution terms. The fallback `deepseek-ai/deepseek-coder-6.7b-base`
  is unconditionally available on Hugging Face.
- Cost estimates use AWS p4d.24xlarge on-demand pricing ($7.40/h). Spot
  instances (~60% discount) or Lambda Labs H100 instances may significantly
  reduce costs.
- This is a **single-seed experiment**. For publication, run with at least 3
  seeds and report mean ± std.

---

## 8. File Reference

| Path | Description |
|---|---|
| [`configs/experiments/deepseek_v4_pro_software_engineering.yaml`](../../configs/experiments/deepseek_v4_pro_software_engineering.yaml) | Master experiment manifest (5 stages) |
| [`configs/experiments/exp_002_deepseek_v4_pro_qlora_sft.yaml`](../../configs/experiments/exp_002_deepseek_v4_pro_qlora_sft.yaml) | QLoRA-only experiment (24 GB GPU entry point) |
| [`configs/domains/software_engineering.yaml`](../../configs/domains/software_engineering.yaml) | Dataset sources for `viforge prepare-data` |
| `data/software_engineering_*.jsonl` | Prepared datasets (generated by `viforge prepare-data`) |
| `runs/deepseek_v4_pro_software_engineering_master/` | All run artifacts (checkpoints, evals, reports) |

---

*Document version: 1.0 (experiment design) — Results pending first real run.*
*ViForge repository: https://github.com/vfcarida/ViForge*
