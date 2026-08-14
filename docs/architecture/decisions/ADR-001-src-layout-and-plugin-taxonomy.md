# ADR-001: src/ Layout and Dynamic Plugin Architecture

## Context
ViForge needs to support an extensible taxonomy of post-training methods (SFT, LoRA, QLoRA, CPT, DPO, GRPO) and benchmark evaluators without requiring changes to the core orchestrator or CLI.

## Decision
1. Organize the codebase under a standard `src/` layout (`src/viforge/`).
2. Implement dynamic registries (`MethodRegistry`, `EvaluatorRegistry`, `InferenceBackendRegistry`) with base abstract protocols.
3. Use Pydantic v2 schemas for all configuration contracts.

## Consequences
- Clean packaging and clear separation between source and tests.
- Contributors can author custom methods or benchmarks by implementing the interface and registering their class.
