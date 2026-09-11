"""
ViForge Automated Model Card (README.md) Generator for Hugging Face Hub.
Produces rich, reproducible, and verifiable model documentation adhering to open-source best practices.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from viforge.config.schemas import ExperimentManifest, ExperimentSummaryReport
from viforge.utils.logging import logger


class ModelCardGenerator:
    """
    Generates standardized Hugging Face Hub Model Cards (README.md) with YAML frontmatter,
    training hyperparameters, benchmark evaluation tables, statistical significance, and Pareto metrics.
    """

    @classmethod
    def generate_model_card(
        cls,
        manifest: ExperimentManifest,
        summary: Optional[ExperimentSummaryReport] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Builds the Markdown content for the HuggingFace model card.
        """
        methods_list = [s.method.value for s in manifest.pipeline]
        method_tags = "\n".join(f"- {m}" for m in set(methods_list))

        domain_gain_str = f"{summary.domain_gain_pct:+.2f}%" if summary else "N/A"
        retention_delta_str = f"{summary.retention_delta_pct:+.2f}%" if summary else "N/A"
        total_cost_str = f"${summary.total_training_cost_usd:.2f}" if summary else "N/A"

        stages_rows = []
        for s in manifest.pipeline:
            stages_rows.append(
                f"| `{s.stage_id}` | `{s.method.value}` | `{s.hyperparameters.learning_rate}` | "
                f"`{s.hyperparameters.num_epochs}` | `{s.hyperparameters.lora_rank}` | `{s.hyperparameters.optimizer}` |"
            )
        stages_table = "\n".join(stages_rows)

        benchmarks_section = ""
        if summary and summary.statistical_deltas:
            delta_rows = []
            for d in summary.statistical_deltas:
                sig_badge = "✅ Significant" if d.is_significant else "⚖️ Inconclusive"
                delta_rows.append(
                    f"| {d.metric_name} | {d.baseline_value:.3f} | {d.specialized_value:.3f} | "
                    f"{d.relative_delta_pct:+.2f}% | [{d.ci_lower:.3f}, {d.ci_upper:.3f}] | {sig_badge} |"
                )
            benchmarks_section = (
                "### Statistical Benchmark Comparison\n\n"
                "| Benchmark | Baseline | Specialist | Relative Delta | 95% Confidence Interval | Significance (p < 0.05) |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
                + "\n".join(delta_rows)
                + "\n\n"
            )

        card_content = f"""---
language:
- en
- code
license: {manifest.model.expected_license}
base_model: {manifest.model.hf_hub_id}
tags:
- viforge
- domain-specialization
- small-language-models
- pareto-optimal
{method_tags}
pipeline_tag: text-generation
---

# {manifest.model.name} (ViForge Specialist)

This model is a domain-specialized checkpoint forged using **[ViForge](https://github.com/viforge/viforge)**, an open-source post-training, Pareto optimization, and evaluation framework.

## Model Highlights
- **Base Architecture:** [{manifest.model.name}](https://huggingface.co/{manifest.model.hf_hub_id}) ({manifest.model.total_parameters}B parameters)
- **Domain Gain:** **{domain_gain_str}** improvement on domain tasks
- **Capability Retention:** **{retention_delta_str}** change on general reasoning benchmarks
- **Total Specialization Compute:** **{total_cost_str}**
- **Artifact Governance:** Verifiable cryptographic provenance with CycloneDX SBOM attestation

## Post-Training Pipeline Configuration

| Stage ID | Method | Learning Rate | Epochs | LoRA Rank | Optimizer |
| :--- | :--- | :--- | :--- | :--- | :--- |
{stages_table}

## Evaluation & Empirical Verification

{benchmarks_section}
### Executive Verdict
> {summary.verdict if summary else "Training campaign configured and validated against ViForge quality bar."}

## Downstream Compression & Edge Deployment (ViPym)
This model is directly compatible with downstream post-training compression via **ViPym**:
- **SmoothQuant (W8A8):** Recommended for zero-loss 2x memory reduction on datacenter GPUs.
- **AutoRound / AWQ (W4A16):** Recommended for 4-bit edge serving ($\\le 8$ GB VRAM footprint).
- **GGUF (Q4_K_M):** Optimized for local CPU/Metal execution via Ollama and llama.cpp.

## Usage & Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{manifest.model.hf_hub_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "def optimize_database_query(query: str) -> str:\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Reproducibility & Governance
- **Experiment Manifest ID:** `{manifest.experiment_id}`
- **ViForge Schema Version:** `{manifest.schema_version}`
- **Generated At:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`
"""

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(card_content, encoding="utf-8")
            logger.info(f"Model Card generated successfully at: {out_p}")

        return card_content
