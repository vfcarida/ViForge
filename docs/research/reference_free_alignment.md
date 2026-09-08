# Reference-Free Alignment: ORPO & SimPO

Post-training preference alignment has historically relied on **Direct Preference Optimization (DPO)**, which requires simultaneously maintaining an active policy $\pi_\theta$ and a frozen reference model $\pi_{\text{ref}}$ in memory. While mathematically elegant, this doubles the base model weight VRAM overhead ($2\times$) and can induce length exploitation (*verbosity bias*).

ViForge provides first-class support for **Next-Generation Reference-Free Alignment**:
1. **ORPO** (*Odds Ratio Preference Optimization*) — Monolithic SFT + Odds Ratio alignment.
2. **SimPO** (*Simple Preference Optimization*) — Length-normalized sequence likelihood with target reward margin $\gamma$.

---

## 1. Architectural & Mathematical Foundations

### 1.1 ORPO: Odds Ratio Preference Optimization

Introduced by Hong et al. (2024), ORPO unifies supervised instruction tuning (NLL loss) with pairwise preference penalties into a single monolithic loss function without requiring an explicit reference policy:

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w \mid x) + \lambda \cdot \mathcal{L}_{\text{OR}}(y_w, y_l \mid x)$$

Where the odds ratio between preferred completion $y_w$ and dispreferred completion $y_l$ is defined as:

$$\text{odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)}$$

$$\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)} \right)$$

- $\lambda$ (`orpo_alpha`, default $0.1$): Balance weight regulating the odds ratio penalty relative to standard cross-entropy.
- **Key Advantage:** Trains SFT and preference discrimination in a single phase without needing pre-aligned checkpoints or double VRAM allocation.

---

### 1.2 SimPO: Simple Preference Optimization

Introduced by Meng et al. (2024), SimPO aligns the policy directly with the target generation metric by using length-normalized average log-likelihood and an explicit target reward margin $\gamma$:

$$\mathcal{L}_{\text{SimPO}} = -\log \sigma \left( \frac{\beta}{|y_w|} \log P_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log P_\theta(y_l \mid x) - \gamma \right)$$

Where:
- $\beta$ (`beta`, default $2.0$): Implicit reward scaling temperature.
- $|y_w|, |y_l|$: Token lengths of winning and losing responses, explicitly preventing verbosity bias.
- $\gamma$ (`simpo_gamma`, default $0.5$): Target margin ensuring winning completions maintain a strict margin over losing completions.
- **Key Advantage:** Superior benchmark accuracy across AlpacaEval 2 and MT-Bench while eliminating length bias and cutting VRAM in half.

---

## 2. Quantitative Comparison: Alignment Methods in ViForge

| Dimension | DPO | KTO | ORPO | SimPO |
| :--- | :--- | :--- | :--- | :--- |
| **Reference Model Required?** | **YES** ($\pi_{\text{ref}}$ in VRAM) | **YES** ($\pi_{\text{ref}}$ in VRAM) | **NO** (Reference-Free) | **NO** (Reference-Free) |
| **VRAM Weight Multiplier** | $1.8\times - 2.0\times$ | $1.8\times - 2.0\times$ | **$1.0\times$** | **$1.0\times$** |
| **Data Format** | Pairwise (`chosen`, `rejected`) | Binary (`prompt`, `label`) | Pairwise (`chosen`, `rejected`) | Pairwise (`chosen`, `rejected`) |
| **SFT Warmup Required?** | Yes | Yes | **No** (Monolithic) | Recommended |
| **Verbosity Bias Mitigation** | Low | Low | Medium | **High** (Explicit Length Norm) |
| **Typical Target Hardware** | A100 80GB / H100 | A100 80GB / H100 | **RTX 4090 / A10G / L4** | **RTX 4090 / A10G / L4** |

---

## 3. Configuration & YAML Specification

### ORPO Stage Example
```yaml
pipeline:
  - stage_id: stage_02_orpo
    method: orpo
    dataset:
      id: ultrafeedback_binarized
      split: train
    hyperparameters:
      learning_rate: 1.0e-5
      orpo_alpha: 0.1
      max_seq_len: 2048
      per_device_batch_size: 2
      gradient_accumulation_steps: 8
      gradient_checkpointing: true
      quantization: nf4
```

### SimPO Stage Example
```yaml
pipeline:
  - stage_id: stage_02_simpo
    method: simpo
    dataset:
      id: ultrafeedback_binarized
      split: train
    hyperparameters:
      learning_rate: 5.0e-6
      beta: 2.0
      simpo_gamma: 0.5
      max_seq_len: 2048
      per_device_batch_size: 2
      gradient_accumulation_steps: 8
      gradient_checkpointing: true
      quantization: nf4
```
