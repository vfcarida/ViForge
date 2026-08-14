"""
ViForge Inference Client: guarantees reproducible generation across baseline and specialist runs.
"""

from typing import List, Optional
from viforge.config.schemas import SamplingParams
from viforge.inference.backends import BaseInferenceBackend


class InferenceClient:
    """Unified client ensuring reproducible, deterministic generation conditions."""

    def __init__(self, backend: BaseInferenceBackend):
        self.backend = backend

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int = 42,
        stop_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
            stop_tokens=stop_tokens or ["<|endoftext|>", "</s>", "```\n"],
        )
        return self.backend.generate(prompts, sampling)
