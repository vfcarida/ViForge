"""
ViForge Model Registry: manages known base models, licenses, and architecture parameters.
"""

from typing import Dict, List, Optional
from viforge.config.schemas import ModelConfig, ModelType, BasePricingConfig
from viforge.models.adapters import HuggingFaceModelAdapter
from viforge.utils.logging import logger


class ModelRegistry:
    """Central registry of reference base models and model card metadata."""

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._register_defaults()

    def _register_defaults(self):
        # DeepSeek V4 Pro
        self.register(
            ModelConfig(
                name="DeepSeek V4 Pro",
                hf_hub_id="deepseek-ai/deepseek-v4-pro",
                revision="main",
                model_type=ModelType.CAUSAL_LM,
                total_parameters=14.5,
                active_parameters=14.5,
                context_window=8192,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                expected_license="MIT",
                pricing=BasePricingConfig(prompt_usd_per_1m=0.14, completion_usd_per_1m=0.28),
                trust_remote_code=False,
            )
        )
        # DeepSeek Coder 6.7B
        self.register(
            ModelConfig(
                name="DeepSeek-Coder-6.7B-Instruct",
                hf_hub_id="deepseek-ai/deepseek-coder-6.7b-instruct",
                revision="main",
                model_type=ModelType.CAUSAL_LM,
                total_parameters=6.7,
                context_window=16384,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                expected_license="deepseek-license",
                pricing=BasePricingConfig(prompt_usd_per_1m=0.10, completion_usd_per_1m=0.20),
            )
        )
        # Qwen 2.5 Coder 7B
        self.register(
            ModelConfig(
                name="Qwen2.5-Coder-7B",
                hf_hub_id="Qwen/Qwen2.5-Coder-7B",
                revision="main",
                model_type=ModelType.CAUSAL_LM,
                total_parameters=7.6,
                context_window=32768,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                expected_license="Apache-2.0",
                pricing=BasePricingConfig(prompt_usd_per_1m=0.12, completion_usd_per_1m=0.24),
            )
        )

    def register(self, config: ModelConfig) -> None:
        self._models[config.name.lower()] = config
        self._models[config.hf_hub_id.lower()] = config
        logger.debug(f"Registered model: {config.name}")

    def get_config(self, identifier: str) -> ModelConfig:
        key = identifier.lower()
        if key in self._models:
            return self._models[key]
        # Return fallback configuration if unknown
        return ModelConfig(
            name=identifier,
            hf_hub_id=identifier,
            total_parameters=14.0,
            expected_license="Custom/Unknown",
        )

    def get_adapter(self, identifier: str) -> HuggingFaceModelAdapter:
        config = self.get_config(identifier)
        return HuggingFaceModelAdapter(config)

    def list_all(self) -> List[ModelConfig]:
        seen = set()
        result = []
        for conf in self._models.values():
            if conf.hf_hub_id not in seen:
                seen.add(conf.hf_hub_id)
                result.append(conf)
        return result


model_registry = ModelRegistry()
