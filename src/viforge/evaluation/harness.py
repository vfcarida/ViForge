"""
ViForge Evaluation Harness: deterministic orchestrator comparing baseline vs specialized models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from viforge.config.schemas import (
    BenchmarkItemConfig,
    BenchmarkResult,
    EvaluationConfig,
    RetentionResult,
)
from viforge.evaluation.suites import evaluator_registry
from viforge.inference.backends import BaseInferenceBackend
from viforge.utils.logging import logger


def check_retention(
    baseline_results: Union[List[BenchmarkResult], Dict[str, float]],
    specialized_results: Union[List[BenchmarkResult], Dict[str, float]],
    threshold: float = 0.10,
) -> List[RetentionResult]:
    """
    Check if specialization degraded general retention capabilities beyond acceptable threshold.
    threshold: maximum allowed relative degradation (e.g., 0.10 = max 10% drop).
    """
    base_map: Dict[str, float] = {}
    if isinstance(baseline_results, dict):
        base_map = dict(baseline_results)
    else:
        for r in baseline_results:
            score = (
                list(r.pass_at_k.values())[0]
                if r.pass_at_k
                else (r.passed_problems / max(1, r.total_problems))
            )
            base_map[r.benchmark_name] = score

    spec_map: Dict[str, float] = {}
    if isinstance(specialized_results, dict):
        spec_map = dict(specialized_results)
    else:
        for r in specialized_results:
            score = (
                list(r.pass_at_k.values())[0]
                if r.pass_at_k
                else (r.passed_problems / max(1, r.total_problems))
            )
            spec_map[r.benchmark_name] = score

    results: List[RetentionResult] = []
    for b_name, base_s in base_map.items():
        spec_s = spec_map.get(b_name, 0.0)
        drop = max(0.0, base_s - spec_s)
        rel_drop = drop / max(1e-5, base_s) if base_s > 0 else 0.0
        within = rel_drop <= threshold

        results.append(
            RetentionResult(
                benchmark=b_name,
                base_score=round(base_s, 4),
                specialized_score=round(spec_s, 4),
                degradation=round(rel_drop, 4),
                within_threshold=within,
            )
        )
    return results


class EvaluationHarness:
    """
    Orchestrates domain benchmarks and general retention suites under identical conditions.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _normalize_items(
        self, items: List[Union[str, BenchmarkItemConfig]]
    ) -> List[BenchmarkItemConfig]:
        normalized = []
        for item in items:
            if isinstance(item, str):
                normalized.append(BenchmarkItemConfig(name=item))
            elif isinstance(item, BenchmarkItemConfig):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(BenchmarkItemConfig(**item))
        return normalized

    def run_all(
        self,
        inference_backend: BaseInferenceBackend,
        output_dir: Path,
        limit: Optional[int] = None,
    ) -> Tuple[List[BenchmarkResult], List[BenchmarkResult]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        domain_results: List[BenchmarkResult] = []
        retention_results: List[BenchmarkResult] = []

        max_p = limit if limit is not None else self.config.max_problems

        domain_items = self._normalize_items(self.config.domain_benchmarks)
        retention_items = self._normalize_items(
            self.config.retention_benchmarks
            if self.config.retention_benchmarks is not None
            else (self.config.general_retention_benchmarks or [])
        )

        logger.info(
            f"EvaluationHarness: running {len(domain_items)} domain and "
            f"{len(retention_items)} retention benchmarks (max_problems={max_p})."
        )

        for b_item in domain_items:
            suite = evaluator_registry.get(b_item.name)
            res = suite.evaluate(
                inference_backend=inference_backend,
                output_dir=output_dir / "domain" / b_item.name,
                sampling_params=self.config.sampling,
                timeout_seconds=b_item.timeout_seconds,
                limit=max_p,
            )
            domain_results.append(res)

        for r_item in retention_items:
            suite = evaluator_registry.get(r_item.name)
            res = suite.evaluate(
                inference_backend=inference_backend,
                output_dir=output_dir / "retention" / r_item.name,
                sampling_params=self.config.sampling,
                timeout_seconds=r_item.timeout_seconds,
                limit=max_p,
            )
            retention_results.append(res)

        return domain_results, retention_results
