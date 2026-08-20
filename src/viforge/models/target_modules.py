"""
ViForge Target Modules Resolver for PEFT/LoRA architectures.
"""

from typing import List, Union


class TargetModuleResolver:
    """Resolves linear layer target module names for known and generic LLM architectures."""

    ARCH_MODULES = {
        "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
    }

    @classmethod
    def resolve(
        cls, model_name_or_hub_id: str, requested_targets: Union[str, List[str], None]
    ) -> List[str]:
        if isinstance(requested_targets, list) and requested_targets:
            return requested_targets

        if requested_targets == "all-linear" or requested_targets is None:
            name_lower = model_name_or_hub_id.lower()
            for key, modules in cls.ARCH_MODULES.items():
                if key in name_lower:
                    return modules
            # Default fallback for modern standard causal transformers
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        if isinstance(requested_targets, str):
            return [requested_targets]
        return ["q_proj", "v_proj"]
