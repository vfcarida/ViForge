"""
ViForge Evaluation Harness: deterministic orchestrator comparing baseline vs specialized models.
"""

from pathlib import Path
from typing import List, Tuple
from viforge.config.schemas import BenchmarkResult, EvaluationConfig
from viforge.evaluation.suites import evaluator_registry
from viforge.inference.backends import BaseInferenceBackend
from viforge.utils.logging import logger


class EvaluationHarness:
    """
    Orchestrates domain benchmarks and general retention suites under identical conditions.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run_all(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
    ) -> Tuple[List[BenchmarkResult], List[BenchmarkResult]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        domain_results: List[BenchmarkResult] = []
        retention_results: List[BenchmarkResult] = []

        logger.info(
            f"EvaluationHarness: running {len(self.config.domain_benchmarks)} domain and "
            f"{len(self.config.general_retention_benchmarks)} retention benchmarks."
        )

        for b_item in self.config.domain_benchmarks:
            suite = evaluator_registry.get(b_item.name)
            res = suite.evaluate(
                inference_backend=inference_backend,
                output_dir=output_dir / "domain" / b_item.name,
                sampling_params=self.config.sampling,
                timeout_seconds=b_item.timeout_seconds,
            )
            domain_results.append(res)

        for r_item in self.config.general_retention_benchmarks:
            suite = evaluator_registry.get(r_item.name)
            res = suite.evaluate(
                inference_backend=inference_backend,
                output_dir=output_dir / "retention" / r_item.name,
                sampling_params=self.config.sampling,
                timeout_seconds=r_item.timeout_seconds,
            )
            retention_results.append(res)

        return domain_results, retention_results
