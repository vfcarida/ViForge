"""
ViForge High-Throughput & Reproducible Inference Backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type
from viforge.config.schemas import SamplingParams
from viforge.utils.logging import logger


class BaseInferenceBackend(ABC):
    """Abstract interface for generation backends."""

    @abstractmethod
    def load_model(self, model_path_or_id: str, adapter_path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        pass


class MockInferenceBackend(BaseInferenceBackend):
    """Deterministic Mock Backend for rapid testing, CI, and smoke tests."""

    def __init__(self, fixed_completion: Optional[str] = None):
        self.model_loaded = False
        self.adapter_path = None
        self.fixed_completion = fixed_completion

    def load_model(self, model_path_or_id: str, adapter_path: Optional[str] = None) -> None:
        self.model_loaded = True
        self.adapter_path = adapter_path
        logger.info(f"MockInferenceBackend: loaded '{model_path_or_id}' (adapter: {adapter_path}).")

    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        completions = []
        is_specialized = self.adapter_path is not None

        for prompt in prompts:
            if self.fixed_completion:
                completions.append(self.fixed_completion)
            elif "has_close_elements" in prompt:
                if is_specialized:
                    completions.append(
                        "    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"
                    )
                else:
                    completions.append("    return False\n")
            elif "separate_paren_groups" in prompt:
                if is_specialized:
                    completions.append("    return ['()', '(())', '(()())']\n")
                else:
                    completions.append("    return []\n")
            elif "truncate_number" in prompt:
                if is_specialized:
                    completions.append("    return number - int(number)\n")
                else:
                    completions.append("    return 0.0\n")
            elif "MMLU" in prompt or "mitochondria" in prompt:
                completions.append("Answer: (B) ATP generation")
            else:
                completions.append("def solution():\n    return True\n")
        return completions


class HuggingFaceInferenceBackend(BaseInferenceBackend):
    """HuggingFace AutoModelForCausalLM generation backend."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self, model_path_or_id: str, adapter_path: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info(f"HFInferenceBackend: loading model from '{model_path_or_id}'...")
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            torch_dtype=dtype,
            device_map=self.device,
        )

        if adapter_path and Path(adapter_path).exists():
            logger.info(f"HFInferenceBackend: attaching adapter from '{adapter_path}'...")
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            self.model = base_model

        self.model.eval()

    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        import torch
        from transformers import set_seed

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        set_seed(sampling_params.seed)
        completions = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            do_sample = sampling_params.temperature > 0.0

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=sampling_params.max_new_tokens,
                    do_sample=do_sample,
                    temperature=max(1e-4, sampling_params.temperature) if do_sample else None,
                    top_p=sampling_params.top_p if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            completions.append(text)

        return completions


class InferenceBackendRegistry:
    def __init__(self):
        self._backends: Dict[str, Type[BaseInferenceBackend]] = {
            "mock": MockInferenceBackend,
            "huggingface": HuggingFaceInferenceBackend,
        }

    def register(self, name: str, backend_cls: Type[BaseInferenceBackend]) -> None:
        self._backends[name.lower()] = backend_cls

    def get(self, name: str) -> BaseInferenceBackend:
        key = name.lower()
        if key not in self._backends:
            raise KeyError(
                f"Inference backend '{name}' not found. Available: {list(self._backends.keys())}"
            )
        return self._backends[key]()


backend_registry = InferenceBackendRegistry()
