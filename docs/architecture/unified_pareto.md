# Unified Pareto Analysis & Interactive Studio

ViForge provides a **5-Dimensional Unified Pareto Optimization Engine** and an **Interactive Dashboard Studio** bridging post-training domain specialization with downstream serving compression.

---

## 1. The 5-Dimensional Pareto Optimization Framework

Traditional evaluation frameworks only look at training accuracy vs compute cost. ViForge expands this into a complete **end-to-end multi-objective Pareto optimization**:

$$\text{Evaluate } \vec{F}(\theta) = \big[ \text{DomainScore}(\theta), \text{Retention}(\theta), -\text{Cost}(\theta), -\text{VRAM}(\theta), -\text{Latency}(\theta) \big]$$

A candidate model variant $A$ dominates variant $B$ ($A \succ B$) if and only if $A$ is no worse than $B$ on all 5 criteria, and strictly superior on at least one:

1. **Domain Capability Score** (Maximize) — Benchmark accuracy on targeted specialization tasks (e.g., SWE-bench, HumanEval+).
2. **General Capability Retention** (Maximize) — Preservation of general reasoning and world knowledge (e.g., MMLU, GSM8K, ARC).
3. **Total Compute Cost ($)** (Minimize) — Cumulative amortized spend across CPT, SFT, DPO/GRPO, and quantization calibration.
4. **Serving VRAM Footprint (GB)** (Minimize) — GPU memory required to host and serve the model in production (vLLM / TensorRT-LLM).
5. **Inference Latency P50 (ms)** (Minimize) — Time-per-token generation latency.

```
       [Base Model] (e.g. DeepSeek-Coder 7B)
            │  Domain: 0.500 | VRAM: 28.0 GB | Latency: 120 ms | Cost: $0
            ▼
       [ViForge Specialist] (CPT + SFT + DPO)
            │  Domain: 0.680 (+36%) | VRAM: 28.1 GB | Latency: 124 ms | Cost: $24.50
            ├──────────────────────────┬──────────────────────────┬──────────────────────────┐
            ▼                          ▼                          ▼                          ▼
     [SmoothQuant W8A8]        [AutoRound W4A16]          [FP8 KV Cache]       [Logit Distillation]
      Domain: 0.675 (-0.7%)     Domain: 0.665 (-2.2%)     Domain: 0.678 (-0.3%) Domain: 0.629 (-7.5%)
      VRAM: 14.2 GB (50% save)  VRAM: 7.2 GB (74% save)   VRAM: 18.0 GB (36%)   VRAM: 3.5 GB (87% save)
      Latency: 63 ms (1.95x)    Latency: 85 ms (1.45x)    Latency: 98 ms (1.2x) Latency: 38 ms (3.2x)
      Total Cost: $24.70        Total Cost: $25.10        Total Cost: $24.55    Total Cost: $31.00
```

---

## 2. Canonical Deployment Sweet Spots

The engine automatically detects the optimal operational points on the Pareto frontier:

| Sweet Spot | Recommendation Criteria | Example Target Hardware |
| :--- | :--- | :--- |
| **Max Domain Capability** | Highest accuracy regardless of VRAM | 8× H100 or A100 Data Center Clusters |
| **Edge Sweet Spot ($\le 8\text{ GB}$)** | Highest capability fitting in $\le 8.5\text{ GB}$ | RTX 4060, T4, L4, or Workstation GPUs |
| **Ultra-Throughput** | Lowest latency generation ($< 40\text{ ms}$) | High-concurrency agent loops and live IDE completions |
| **Maximum Efficiency Index** | Highest $(Domain \times Throughput) / (VRAM \times Cost)$ | Best ROI for production cloud deployments |

---

## 3. CLI Usage

### Unified Pareto Analysis
```bash
# Analyze experiment manifest and compute full Base -> Specialist -> Compressed frontier
viforge pareto-unified configs/experiments/exp_003_deepseek_v4_pro_cpt_sft_dpo.yaml

# Customize expected domain gain, retention delta, and export directory
viforge pareto-unified configs/experiments/exp_003_deepseek_v4_pro_cpt_sft_dpo.yaml \
  --domain-gain 36.0 \
  --retention-delta 0.0 \
  --cost 24.50 \
  --output-dir reports/pareto
```

### Launch Interactive Studio Dashboard
```bash
# Launch interactive Streamlit Studio on localhost:8501
viforge ui

# Specify custom port or bind to external host
viforge ui --port 8080 --host 0.0.0.0 --no-browser
```

---

## 4. Interactive Studio Features

The ViForge Studio web interface provides:
- **Campaign Overview:** Live model progression and key KPI cards.
- **Unified Pareto Explorer:** 2D and 3D Plotly scatter plots with VRAM sliders, Pareto filters, and sweet spot callouts.
- **Statistical Rigor:** Paired McNemar significance matrices and Wilson 95% score confidence intervals.
- **Visual Experiment Designer:** Interactive YAML assembly with real-time pre-flight VRAM and cost estimators.
- **ViPym Compression Hub:** Point-and-click execution of downstream recipes (`smoothquant_w8a8`, `autoround_w4a16`, etc.) in simulated mock or live mode.
- **System Doctor:** Real-time hardware and dependency health diagnostics.

---

## 5. Cross-Model Multi-Campaign Benchmark Comparison

When evaluating multiple base model architectures (e.g. Qwen2.5-Coder-7B vs Llama-3.1-8B vs DeepSeek-Coder-1.3B) or comparing competing specialization recipes, ViForge provides cross-model Pareto comparison via `MultiCampaignParetoComparator`:

### CLI Multi-Campaign Comparison
```bash
# Compare multiple campaign Pareto reports on a unified global frontier
viforge compare-campaigns \
  runs/qwen7b/unified_pareto.json \
  runs/llama8b/unified_pareto.json \
  runs/deepseek1b/unified_pareto.json \
  --output-dir reports/cross_model_pareto
```

The command outputs:
1. A summary table comparing model architectures, peak accuracy, minimum VRAM, and number of globally non-dominated variants.
2. `multi_model_pareto.json`: Structured benchmark metrics, campaign summaries, and global sweet spot recommendations.
3. `multi_model_pareto.html`: Interactive Dark Glassmorphism Plotly chart with color-coded model traces and gold star callouts for globally optimal configurations.

### Studio Multi-Model Campaign Overlay
In **ViForge Studio** Tab 2 (**🎯 Unified Pareto Frontier**), the **Cross-Model Campaign Benchmark Comparison** expander allows toggling model architectures simultaneously to visualize trade-offs directly in browser.

