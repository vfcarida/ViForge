"""
ViForge HumanEval+ Evaluation Suite Plugin.
Loads benchmark problems from Hugging Face (`evalplus/humanevalplus` or `openai_humaneval`) with offline fallback.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from viforge.config.schemas import BenchmarkResult, SamplingParams
from viforge.inference.backends import BaseInferenceBackend
from viforge.security.sandbox import ExecutionSandbox
from viforge.utils.logging import logger


class HumanEvalPlusSuite:
    """HumanEval and HumanEval+ pass@1 benchmark evaluator."""

    @property
    def name(self) -> str:
        return "humaneval_plus"

    def __init__(self):
        self.sandbox = ExecutionSandbox(default_timeout_sec=10)

    def _fallback_toy_problems(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    """\n',
                "test": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n",
                "entry_point": "has_close_elements",
            },
            {
                "task_id": "HumanEval/1",
                "prompt": 'def separate_paren_groups(paren_string: str) -> list[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses.\n    """\n',
                "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
                "entry_point": "separate_paren_groups",
            },
            {
                "task_id": "HumanEval/2",
                "prompt": 'def truncate_number(number: float) -> float:\n    """ Given a positive floating point number, it can be decomposed into an integer part and decimals.\n    """\n',
                "test": "assert truncate_number(3.5) == 0.5\n",
                "entry_point": "truncate_number",
            },
        ]

    def load_problems(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load problems from Hugging Face dataset (evalplus/humanevalplus or openai_humaneval).
        Falls back gracefully to curated toy problems when offline.
        """
        problems: List[Dict[str, Any]] = []
        try:
            from datasets import load_dataset

            try:
                ds = load_dataset("evalplus/humanevalplus", split="test")
            except Exception:
                ds = load_dataset("openai_humaneval", split="test")

            for row in ds:
                entry = {
                    "task_id": row.get("task_id", f"HumanEval/{len(problems)}"),
                    "prompt": row.get("prompt", ""),
                    "test": row.get("test", ""),
                    "entry_point": row.get("entry_point", ""),
                }
                problems.append(entry)
            logger.info(f"Loaded {len(problems)} HumanEval+ problems from HuggingFace.")
        except Exception as e:
            logger.debug(f"HumanEval+ online dataset loading fallback: {e}")
            problems = self._fallback_toy_problems()

        return problems[:limit] if limit else problems

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 10,
        limit: Optional[int] = None,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        test_problems = self.load_problems(limit=limit)
        prompts = [p["prompt"] for p in test_problems]
        completions = inference_backend.generate(prompts, sampling_params)

        passed = 0
        failed = 0

        for prob, comp in zip(test_problems, completions):
            full_code = prob["prompt"] + comp
            res = self.sandbox.execute_snippet(
                code_string=full_code,
                test_assertions=prob["test"],
                timeout_seconds=timeout_seconds,
            )
            if res["passed"]:
                passed += 1
            else:
                failed += 1

        total = len(test_problems)
        pass_at_1 = passed / max(1, total)
        elapsed = time.time() - start_time

        logger.info(f"HumanEval+ evaluation: {passed}/{total} passed (Pass@1: {pass_at_1:.2%}).")

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"pass@1": round(pass_at_1, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=failed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"pass_rate_pct": round(pass_at_1 * 100.0, 2)},
        )
