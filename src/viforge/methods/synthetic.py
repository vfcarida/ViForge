"""
ViForge Synthetic Task Generation, Verification, and Rejection Sampling Pipeline.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from viforge.methods.sft import SFTMethod
from viforge.methods.base import method_registry
from viforge.preprocessing.normalizer import CodeNormalizer
from viforge.utils.logging import logger


class SyntheticDataPipeline:
    """
    Orchestrates synthetic task generation, AST syntax verification, and rejection sampling.
    """

    def __init__(
        self,
        teacher_model_id: str = "claude-3-5-sonnet-20241022",
        generation_temperature: float = 0.7,
        min_code_lines: int = 2,
    ):
        self.teacher_model_id = teacher_model_id
        self.temperature = generation_temperature
        self.min_code_lines = min_code_lines

    @staticmethod
    def extract_code_block(text: str) -> str:
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        return text.strip()

    def verify_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        response_text = example.get("response", "") or example.get("completion", "") or example.get("code", "")
        code = self.extract_code_block(str(response_text))
        is_valid, err_msg = CodeNormalizer.validate_ast(code)
        num_lines = len(code.splitlines())

        return {
            "is_valid_syntax": is_valid,
            "syntax_error": err_msg,
            "code_lines": num_lines,
            "is_accepted": is_valid and num_lines >= self.min_code_lines,
        }

    def process_and_filter_candidates(
        self,
        raw_candidates: List[Dict[str, Any]],
        system_prompt: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        accepted = []
        rejected = 0

        for cand in raw_candidates:
            v_res = self.verify_example(cand)
            if v_res["is_accepted"]:
                enriched = dict(cand)
                enriched["provenance"] = {
                    "teacher_model": self.teacher_model_id,
                    "generation_time": datetime.now(timezone.utc).isoformat(),
                    "generation_temperature": self.temperature,
                    "system_prompt_snippet": system_prompt[:100] + "...",
                    "verification_passed": True,
                    "syntax_checked": True,
                }
                accepted.append(enriched)
            else:
                rejected += 1

        stats = {
            "total_candidates": len(raw_candidates),
            "accepted_samples": len(accepted),
            "rejected_samples": rejected,
            "acceptance_rate_pct": round((len(accepted) / max(1, len(raw_candidates))) * 100.0, 2),
            "teacher_model": self.teacher_model_id,
        }
        logger.info(
            f"Synthetic data filtering: {len(accepted)}/{len(raw_candidates)} passed verification "
            f"({stats['acceptance_rate_pct']}% acceptance)."
        )
        return accepted, stats


class SyntheticSFTMethod(SFTMethod):
    """Specialized SFT stage on verified teacher synthetic data."""

    @property
    def method_name(self) -> str:
        return "synthetic_sft"


method_registry.register("synthetic_sft", SyntheticSFTMethod)
