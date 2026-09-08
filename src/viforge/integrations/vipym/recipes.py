"""
ViForge x ViPym Pre-Packaged Compression Recipes.
"""

from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, Field


class CompressionRecipe(str, Enum):
    """Supported standardized downstream compression recipes."""

    SMOOTHQUANT_W8A8 = "smoothquant_w8a8"
    AUTOROUND_W4A16 = "autoround_w4a16"
    AWQ_W4A16 = "awq_w4a16"
    GPTQ_W4A16 = "gptq_w4a16"
    DISTILL_LOGIT = "distill_logit"
    FP8_KV = "fp8_kv"
    SPINQUANT = "spinquant"


class RecipeDefinition(BaseModel):
    """Metadata and parameters defining a ViPym compression recipe."""

    recipe: CompressionRecipe
    method: str = Field(..., description="ViPym compression method name")
    scheme: str = Field(..., description="ViPym compression scheme (e.g. W8A8, W4A16)")
    description: str
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    recommended_for: str


RECIPE_CATALOG: Dict[CompressionRecipe, RecipeDefinition] = {
    CompressionRecipe.SMOOTHQUANT_W8A8: RecipeDefinition(
        recipe=CompressionRecipe.SMOOTHQUANT_W8A8,
        method="smoothquant",
        scheme="W8A8",
        description="SmoothQuant W8A8 activation-weight quantization for high-throughput low-latency serving.",
        default_parameters={"alpha": 0.5, "per_channel": True},
        recommended_for="Inference serving on modern server GPUs with INT8 Tensor Cores (e.g. A100, H100).",
    ),
    CompressionRecipe.AUTOROUND_W4A16: RecipeDefinition(
        recipe=CompressionRecipe.AUTOROUND_W4A16,
        method="autoround",
        scheme="W4A16",
        description="AutoRound gradient-free optimization for 4-bit weight-only quantization.",
        default_parameters={"bits": 4, "group_size": 128, "iters": 200},
        recommended_for="Low-VRAM edge devices and local developer workstations (e.g. RTX 4090, Apple Silicon).",
    ),
    CompressionRecipe.AWQ_W4A16: RecipeDefinition(
        recipe=CompressionRecipe.AWQ_W4A16,
        method="awq",
        scheme="W4A16",
        description="Activation-aware Weight Quantization preserving salient weight channels.",
        default_parameters={"bits": 4, "group_size": 128},
        recommended_for="Fast vLLM and SGLang deployment with minimal perplexity degradation.",
    ),
    CompressionRecipe.GPTQ_W4A16: RecipeDefinition(
        recipe=CompressionRecipe.GPTQ_W4A16,
        method="gptq",
        scheme="W4A16",
        description="One-shot second-order error compensation quantization.",
        default_parameters={"bits": 4, "group_size": 128, "act_order": True},
        recommended_for="Cross-platform inference runtimes supporting ExLlamaV2 and HuggingFace TGI.",
    ),
    CompressionRecipe.DISTILL_LOGIT: RecipeDefinition(
        recipe=CompressionRecipe.DISTILL_LOGIT,
        method="distill_logit",
        scheme="DISTILL_STUDENT",
        description="Teacher-Student logit distillation transferring specialist capability to a compact base model.",
        default_parameters={"temperature": 2.0, "alpha": 0.5},
        recommended_for="Compressing large specialist models (e.g. 7B/14B) into ultra-fast student models (e.g. 1.5B/3B).",
    ),
    CompressionRecipe.FP8_KV: RecipeDefinition(
        recipe=CompressionRecipe.FP8_KV,
        method="kv_cache_fp8",
        scheme="FP8",
        description="FP8 quantization for dynamic attention KV cache.",
        default_parameters={"kv_cache_dtype": "fp8"},
        recommended_for="High-concurrency long-context serving scenarios requiring 2x larger KV cache capacity.",
    ),
    CompressionRecipe.SPINQUANT: RecipeDefinition(
        recipe=CompressionRecipe.SPINQUANT,
        method="spinquant",
        scheme="W4A16",
        description="Random Hadamard matrix rotation to suppress activation outliers before quantization.",
        default_parameters={"bits": 4, "group_size": 128},
        recommended_for="Extreme 4-bit quantization on architectures with pronounced activation outliers.",
    ),
}
