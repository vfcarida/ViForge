# ViForge Experiment Report: `exp_002_deepseek_v4_pro_qlora_sft`

**Model Base:** `DeepSeek V4 Pro`  
**Timestamp:** `2026-08-14T18:11:39.140119+00:00`  
**Total Training Cost:** `$0.03`  
**Total Wall-Clock Time:** `0.00 hrs`

---

## Executive Summary & Hypothesis Verdict

> **HYPOTHESIS CONFIRMED: Specialization delivered a +100.0% domain improvement while retaining general capability (+0.0%), yielding a superior capability-per-dollar metric (249.98) at $0.03 total training cost.**

### Primary Specialization Metrics:
* **Domain Gain ($\Delta \mathcal{S}_{\text{domain}}$):** `+100.00%` (50.0% -> 100.0%)
* **General Retention Delta ($\Delta \mathcal{S}_{\text{general}}$):** `+0.00%` (75.0% -> 75.0%)

---

## Training Stages Breakdown

| Stage ID | Method | Loss | Tokens | Tok/s | Peak VRAM | Trainable % | Stage Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `stage_1_qlora_sft` | qlora | 0.825 | 204,800,000 | 2400 | 28.4 GB | 0.10% | $0.03 |

---

## Benchmark Evaluation Results

| Benchmark | Metrics / Pass@k | Passed / Total | Exec Time |
| :--- | :--- | :--- | :--- |
| **humaneval_plus** | {"pass@1": 1.0} | 3/3 | 0.6s |
| **swe_bench_lite** | {"resolved@1": 1.0} | 2/2 | 0.0s |
| **mmlu_pro** | {"acc": 0.5} | 1/2 | 0.0s |
| **gsm8k** | {"acc": 1.0} | 2/2 | 0.0s |

---

## Multi-Objective Pareto Frontier

| Status | Model Variant | Domain Quality | General Retention | Training Cost | Latency (P50) | Capability-per-Dollar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ★ Optimal | `DeepSeek V4 Pro (Base Baseline)` | 50.0% | 75.0% | $0.00 | 120ms | **0.00** |
| ★ Optimal | `DeepSeek V4 Pro (Specialized (qlora))` | 100.0% | 75.0% | $0.03 | 124ms | **249.98** |

*Report generated automatically by ViForge.*
