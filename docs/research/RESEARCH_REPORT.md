# ViForge Technical Research Report

## 1. Domain & Theoretical Foundations

### A. Parameter-Efficient Post-Training (LoRA & QLoRA)
Fine-tuning full parameter sets on frontier architectures (14B+ parameters) incurs significant compute and VRAM overhead:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A), \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}$$
Where $r \ll \min(d, k)$ (typically $r \in \{32, 64\}$) reduces trainable parameters by $\sim 99.9\%$. In QLoRA, base weights $W_0$ are stored in NormalFloat4 (NF4) with double quantization, while gradients are accumulated in BF16/FP16, enabling 14B model specialization on single 24GB–40GB GPUs.

### B. Continual Pretraining (DAPT) & Catastrophic Forgetting
Domain-Adaptive Pretraining (DAPT) adapts token distributions to domain corpora (e.g. AST syntax, GitHub repositories). However, prolonged training risks catastrophic forgetting of general reasoning capabilities (MMLU-Pro, GSM8K). ViForge balances this via dual evaluation:
$$\text{Score}_{\text{composite}} = \Delta \mathcal{S}_{\text{domain}} - \lambda \cdot \max(0, -\Delta \mathcal{S}_{\text{general}})$$

### C. Preference Alignment (DPO & GRPO)
- **Direct Preference Optimization (DPO):** Optimizes policy $\pi_\theta$ directly over pairwise preferences $(y_w \succ y_l)$ without training a standalone reward model:
  $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
- **Group Relative Policy Optimization (GRPO):** Computes group baseline advantages across $G$ candidate completions per prompt, eliminating the critic network and using deterministic unit-test pass/fail rewards.

---

## 2. Evaluation & Contamination Shielding

To guarantee scientific validity, ViForge incorporates:
1. **EvalPlus Test Augmentation:** Expanding HumanEval and MBPP test suites by $80\times$ via mutation and constraint synthesis.
2. **10-gram Sliding Window Leakage Checking:** Rejecting training samples with $>5\%$ verbatim shingle overlap against evaluation benchmarks.
3. **Wilson Score Confidence Intervals:**
   $$\hat{p} \pm z_{1-\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \bigg/ \left(1 + \frac{z^2}{n}\right)$$
