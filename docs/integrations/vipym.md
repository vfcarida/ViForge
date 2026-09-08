# ViPym Downstream Compression Integration

ViForge integrates natively with **ViPym** (High-Performance LLM Compression & Inference Engine) to bridge **post-training domain specialization** with **downstream model compression and serving optimization**.

While ViForge specializes models through Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), and Alignment (DPO, GRPO, KTO), ViPym compresses the resulting specialized checkpoints using modern post-training quantization, activation smoothing, logit distillation, and KV-cache compression.

```
+-------------------------------------------------------------------------------+
|                             ViForge Post-Training                             |
|  [Base Model] ---> [CPT Domain Adapt] ---> [LoRA/QLoRA SFT] ---> [DPO/GRPO]   |
|                                                                     |         |
|                                                           [Merge Checkpoint]  |
+---------------------------------------------------------------------+---------+
                                                                      |
                                                          viforge export-vipym
                                                                      v
+-------------------------------------------------------------------------------+
|                         ViPym Downstream Compression                          |
|   SmoothQuant W8A8  |  AutoRound W4A16  |  Logit Distill  |  FP8 KV Cache     |
|                                                                     |         |
|                                                            [Deploy Engine]    |
|                                                        (vLLM / TensorRT-LLM)  |
+-------------------------------------------------------------------------------+
```

---

## Supported Compression Recipes

ViForge provides a catalog of pre-configured, production-validated ViPym recipes:

| Recipe Identifier | Method | Scheme | Description | Typical Size Reduction | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `smoothquant_w8a8` | `smoothquant` | `W8A8` | Activation-aware 8-bit quantization with migration multiplier ($\alpha=0.85$) | ~50% VRAM reduction | Ampere, Ada, Hopper |
| `autoround_w4a16` | `autoround` | `W4A16` | Sign Gradient Descent rounding for 4-bit weight compression | ~75% VRAM reduction | Any GPU (T4, A10G, L4, A100) |
| `awq_w4a16` | `awq` | `W4A16` | Activation-aware 4-bit weight-only quantization | ~75% VRAM reduction | vLLM / HuggingFace |
| `gptq_w4a16` | `gptq` | `W4A16` | Second-order Taylor series error compensation | ~75% VRAM reduction | AutoGPTQ / vLLM |
| `distill_logit` | `distill_logit` | `DISTILL_STUDENT`| Teacher-student soft target logit distillation | Target student model size | Low-latency edge & CPU/GPU |
| `fp8_kv` | `kv_cache_fp8` | `FP8` | 8-bit floating point key-value cache compression | 50% KV cache memory | Ada & Hopper architectures |
| `spinquant` | `spinquant` | `W4A16` | Random orthogonal rotations to suppress activation outliers | ~75% VRAM reduction | Outlier-heavy architectures |

---

## CLI Usage

### 1. Export ViForge Models to ViPym Manifests

You can export directly from a merged checkpoint directory or from a ViForge `experiment.yaml` manifest.

```bash
# Export from a merged checkpoint
viforge export-vipym ./merged_specialist --recipe smoothquant_w8a8 --output configs/vipym_smoothquant.yaml

# Export from a ViForge experiment manifest (automatically maps evaluation suites)
viforge export-vipym configs/deepseek_coder.yaml --recipe autoround_w4a16 --output configs/vipym_autoround.yaml

# Distillation recipe: specify the target student model
viforge export-vipym ./merged_specialist --recipe distill_logit --student-model Qwen/Qwen2.5-Coder-1.5B --output configs/vipym_distill.yaml
```

### 2. Execute ViPym Compression

Run compression pipelines either in **deterministic simulation mode** (fast CI/testing) or in **live production mode**:

```bash
# Fast mock simulation (validates configuration and produces simulated metrics)
viforge compress-vipym configs/vipym_smoothquant.yaml --mock

# Live execution using the installed ViPym engine
viforge compress-vipym configs/vipym_smoothquant.yaml --live

# End-to-end: directly pass a ViForge model directory and execute compression
viforge compress-vipym ./merged_specialist --recipe autoround_w4a16 --work-dir runs/vipym_autoround --mock
```

---

## Python Programmatic API

### Exporting Configurations

```python
from viforge.integrations.vipym import ViPymExporter, CompressionRecipe

# Export from model directory
yaml_path = ViPymExporter.export_from_model_dir(
    model_dir="./merged_specialist",
    recipe=CompressionRecipe.SMOOTHQUANT_W8A8,
    output_yaml_path="configs/vipym_export.yaml",
    calibration_dataset="wikitext",
    calibration_samples=512,
)

# Export from ViForge manifest with benchmark mapping
yaml_path = ViPymExporter.export_from_manifest(
    manifest_or_path="configs/experiment.yaml",
    recipe=CompressionRecipe.AUTOROUND_W4A16,
    output_yaml_path="configs/vipym_autoround.yaml",
)
```

### Executing Compression

```python
from viforge.integrations.vipym import ViPymRunner

# Execute compression (mock or live)
summary = ViPymRunner.run_compression(
    vipym_config_path="configs/vipym_export.yaml",
    work_dir="runs/vipym",
    mock=True,
)

print(f"Status: {summary['status']}")
print(f"Size Reduction: {summary['size_reduction_pct']}%")
print(f"Speedup: {summary['latency_speedup']}x")
```

---

## Architectural Guarantees & Schema Compliance

1. **Zero Hard Dependency**: ViForge does not require `vipym` to be installed for post-training, evaluation, or standard artifact export (GGUF, AWQ). ViPym integration functions as an optional peer extension.
2. **Pydantic Validation**: When `vipym` is installed in the active Python environment, `ViPymExporter` validates all exported configurations against `vipym.core.config.ViPymExperimentConfig` before writing to disk.
3. **Automated Evaluation Mapping**: ViForge evaluation benchmarks (e.g., `humaneval_plus`, `swe_bench_lite`, `gsm8k`, `arc_challenge`) are translated to ViPym evaluation suite identifiers (`humaneval`, `swe_bench`, `gsm8k`, `arc`).
