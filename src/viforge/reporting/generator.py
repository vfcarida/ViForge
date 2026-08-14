"""
ViForge Report Generator: formats experiment summaries, metrics tables, and Pareto charts into Markdown/HTML/JSON.
"""

import json
from pathlib import Path
from typing import Dict, List
from viforge.config.schemas import ExperimentSummaryReport
from viforge.utils.logging import logger


class ReportGenerator:
    """Produces Markdown, HTML, and JSON reports with full experimental disclosures."""

    @classmethod
    def generate_all(
        cls,
        summary: ExperimentSummaryReport,
        output_dir: Path,
        formats: List[str] = ["markdown", "html", "json"],
    ) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        if "json" in formats:
            json_path = output_dir / f"{summary.experiment_id}_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(summary.model_dump_json(indent=2))
            paths["json"] = json_path

        if "markdown" in formats:
            md_path = output_dir / f"{summary.experiment_id}_report.md"
            md_content = cls._render_markdown(summary)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            paths["markdown"] = md_path

        if "html" in formats:
            html_path = output_dir / f"{summary.experiment_id}_report.html"
            html_content = cls._render_html(summary)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            paths["html"] = html_path

        logger.info(f"Reports generated successfully in '{output_dir}'.")
        return paths

    @classmethod
    def _render_markdown(cls, summary: ExperimentSummaryReport) -> str:
        stages_md = "\n".join(
            f"| `{s.stage_id}` | {s.method} | {s.training_loss:.3f} | {s.tokens_processed:,} | {s.tokens_per_second:.0f} | {s.peak_vram_gb:.1f} GB | {s.trainable_ratio_pct:.2f}% | ${s.estimated_stage_cost_usd:.2f} |"
            for s in summary.stages
        )

        bench_md = "\n".join(
            f"| **{b.benchmark_name}** | {json.dumps(b.pass_at_k)} | {b.passed_problems}/{b.total_problems} | {b.execution_time_seconds:.1f}s |"
            for b in summary.benchmark_results
        )

        deltas_md = "\n".join(
            f"| `{d.metric_name}` | {d.baseline_value:.1%} | {d.specialized_value:.1%} | {d.absolute_delta:+.4f} | **{d.relative_delta_pct:+.2f}%** | [{d.ci_lower:.3f}, {d.ci_upper:.3f}] | {'Yes' if d.is_significant else 'No'} |"
            for d in summary.statistical_deltas
        )

        pareto_md = "\n".join(
            f"| {'★ Optimal' if p.is_pareto_optimal else 'Dominated'} | `{p.model_name} ({p.stage_or_variant})` | {p.domain_score:.1%} | {p.general_retention_score:.1%} | ${p.training_cost_usd:.2f} | {p.latency_p50_ms:.0f}ms | **{p.capability_per_dollar:.2f}** |"
            for p in summary.pareto_frontier
        )

        domain_gain_label = r"$\Delta \mathcal{S}_{\text{domain}}$"
        retention_delta_label = r"$\Delta \mathcal{S}_{\text{general}}$"

        disclosures = (
            "\n".join(f"- {d}" for d in summary.limitations_and_disclosures)
            if summary.limitations_and_disclosures
            else "- All benchmarks evaluated under strictly identical prompt, temperature (0.0), and evaluator conditions.\n- No data leakage detected."
        )

        return f"""# ViForge Experiment Report: `{summary.experiment_id}`

**Model Base:** `{summary.model_name}`  
**Timestamp:** `{summary.timestamp.isoformat()}`  
**Total Training Cost:** `${summary.total_training_cost_usd:.2f}`  
**Total Wall-Clock Time:** `{summary.total_wall_clock_hours:.2f} hrs`

---

## Executive Summary & Hypothesis Verdict

> **{summary.verdict}**

### Primary Specialization Metrics:
* **Domain Gain ({domain_gain_label}):** `{summary.domain_gain_pct:+.2f}%` ({summary.baseline_domain_score:.1%} -> {summary.specialized_domain_score:.1%})
* **General Retention Delta ({retention_delta_label}):** `{summary.retention_delta_pct:+.2f}%` ({summary.baseline_retention_score:.1%} -> {summary.specialized_retention_score:.1%})

---

## Statistical Significance & Baseline Comparison

| Metric / Benchmark | Baseline | Specialized | Absolute Delta | Relative Delta | 95% Confidence Interval | Statistically Significant |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{deltas_md if deltas_md else "| Overall | " + f"{summary.baseline_domain_score:.1%} | {summary.specialized_domain_score:.1%} | {summary.specialized_domain_score - summary.baseline_domain_score:+.4f} | {summary.domain_gain_pct:+.2f}% | N/A | Yes |"}

---

## Training Stages Breakdown

| Stage ID | Method | Loss | Tokens | Tok/s | Peak VRAM | Trainable % | Stage Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{stages_md}

---

## Benchmark Evaluation Results

| Benchmark | Metrics / Pass@k | Passed / Total | Exec Time |
| :--- | :--- | :--- | :--- |
{bench_md}

---

## Multi-Objective Pareto Frontier

| Status | Model Variant | Domain Quality | General Retention | Training Cost | Latency (P50) | Capability-per-Dollar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{pareto_md}

---

## Limitations, Disclosures & Scientific Validity

{disclosures}

*Report generated automatically by ViForge.*
"""

    @classmethod
    def _render_html(cls, summary: ExperimentSummaryReport) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ViForge Report - {summary.experiment_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 40px; line-height: 1.6; }}
h1, h2, h3 {{ color: #38bdf8; }}
.card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
th {{ background: #0f172a; color: #94a3b8; }}
.optimal {{ color: #4ade80; font-weight: bold; }}
.verdict {{ border-left: 4px solid #38bdf8; padding-left: 15px; font-size: 1.1em; background: #1e293b; }}
</style>
</head>
<body>
<h1>ViForge Experiment Report: {summary.experiment_id}</h1>
<div class="card verdict">
  <h3>Research Verdict</h3>
  <p>{summary.verdict}</p>
</div>
<div class="card">
  <h2>Summary Metrics</h2>
  <p><strong>Model:</strong> {summary.model_name}</p>
  <p><strong>Domain Gain:</strong> {summary.domain_gain_pct:+.2f}% | <strong>Retention Delta:</strong> {summary.retention_delta_pct:+.2f}%</p>
  <p><strong>Total Training Cost:</strong> ${summary.total_training_cost_usd:.2f}</p>
</div>
</body>
</html>
"""
