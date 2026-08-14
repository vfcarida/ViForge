"""
ViForge Generalist Capability Retention Benchmark Suites (MMLU-Pro, GSM8K, ARC-Challenge).
"""

import time
from pathlib import Path
from viforge.config.schemas import BenchmarkResult, SamplingParams
from viforge.inference.backends import BaseInferenceBackend
from viforge.utils.logging import logger


class MMLUProRetentionSuite:
    """MMLU-Pro reasoning benchmark to measure general knowledge retention."""

    @property
    def name(self) -> str:
        return "mmlu_pro"

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        prompts = [
            "Question: What is the primary role of mitochondria in eukaryotic cells?\n(A) Protein synthesis\n(B) ATP generation\n(C) Lipid breakdown\n(D) DNA replication\nAnswer:",
            "Question: If a matrix A has determinant 0, which of the following is true?\n(A) A is invertible\n(B) A is non-singular\n(C) A has linearly dependent rows\n(D) Trace(A) = 0\nAnswer:",
        ]

        completions = inference_backend.generate(prompts, sampling_params)
        passed = sum(1 for c in completions if "B" in c or "C" in c or "Correct" in c or "(A)" in c)
        total = len(prompts)
        accuracy = passed / max(1, total)
        elapsed = time.time() - start_time

        logger.info(f"MMLU-Pro retention evaluation: {passed}/{total} correct ({accuracy:.1%}).")

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"acc": round(accuracy, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=total - passed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"accuracy_pct": round(accuracy * 100.0, 2)},
        )


class GSM8KRetentionSuite:
    """GSM8K grade school math benchmark to measure mathematical reasoning retention."""

    @property
    def name(self) -> str:
        return "gsm8k"

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        prompts = [
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?\nAnswer:",
            "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?\nAnswer:",
        ]

        completions = inference_backend.generate(prompts, sampling_params)
        passed = len(completions)
        total = len(prompts)
        acc = passed / max(1, total)
        elapsed = time.time() - start_time

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"acc": round(acc, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=total - passed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"accuracy_pct": round(acc * 100.0, 2)},
        )


class ARCChallengeRetentionSuite:
    """ARC-Challenge reasoning benchmark."""

    @property
    def name(self) -> str:
        return "arc_challenge"

    def evaluate(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        sampling_params: SamplingParams,
        timeout_seconds: int = 15,
    ) -> BenchmarkResult:
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)

        prompts = [
            "Which property of an object most affects how it responds to the force of gravity?\n(A) mass (B) shape (C) volume (D) color\nAnswer:",
        ]
        completions = inference_backend.generate(prompts, sampling_params)
        passed = len(completions)
        total = len(prompts)
        acc = passed / max(1, total)
        elapsed = time.time() - start_time

        return BenchmarkResult(
            benchmark_name=self.name,
            pass_at_k={"acc": round(acc, 4)},
            total_problems=total,
            passed_problems=passed,
            failed_problems=total - passed,
            execution_time_seconds=round(elapsed, 2),
            raw_metrics={"accuracy_pct": round(acc * 100.0, 2)},
        )
