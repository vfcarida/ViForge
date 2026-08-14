# ADR-002: Multi-Objective Pareto Frontier and Cost Modeling

## Context
A model cannot be considered successful merely because target domain metrics improve if catastrophic forgetting occurs or training/inference costs exceed those of larger frontier models.

## Decision
1. Compute multi-objective non-dominated Pareto frontiers across 5 dimensions: Domain Quality, General Retention, Training Cost, Latency, and Memory.
2. Formalize the Capability-per-Dollar index:
   $$\text{Capability-per-Dollar} = \frac{\Delta \mathcal{S}_{\text{domain}} - \lambda \cdot \max(0, -\Delta \mathcal{S}_{\text{general}})}{\mathcal{C}_{\text{train}} + \mathcal{C}_{\text{infer}}(N_{\text{tokens}})}$$

## Consequences
- Produces rigorous, mathematically grounded research verdicts answering whether specialization is economically justified.
