"""
ViForge Model Adapter for HuggingFace Transformers.
"""

from typing import Any, List, Optional
from viforge.config.schemas import ModelConfig
from viforge.models.target_modules import TargetModuleResolver
from viforge.utils.logging import logger


class HuggingFaceModelAdapter:
    """Standard adapter for loading and inspecting Hugging Face base models."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def load_base_model(self, quantization: Optional[str] = None, device_map: str = "auto") -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise RuntimeError("PyTorch and Transformers are required to load models.")

        quant_config = None
        torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        if quantization == "nf4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif quantization == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(
            f"Loading base model '{self.model_config.hf_hub_id}' (revision: {self.model_config.revision}, quant: {quantization})..."
        )
        return AutoModelForCausalLM.from_pretrained(
            self.model_config.hf_hub_id,
            revision=self.model_config.revision,
            quantization_config=quant_config,
            torch_dtype=torch_dtype if quant_config is None else None,
            device_map=device_map,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.hf_hub_id,
            revision=self.model_config.revision,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def get_target_modules(self, requested: Optional[str] = None) -> List[str]:
        return TargetModuleResolver.resolve(
            self.model_config.hf_hub_id, requested or self.model_config.target_modules
        )
