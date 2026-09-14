"""
ViForge Evaluation Benchmark Suites Registry.
"""

from typing import Dict, Type
from viforge.evaluation.suites.humaneval_plus import HumanEvalPlusSuite
from viforge.evaluation.suites.swe_bench import SWEBenchSuite, MBPPPlusSuite, LiveCodeBenchSuite
from viforge.evaluation.suites.retention import (
    MMLUProRetentionSuite,
    GSM8KRetentionSuite,
    ARCChallengeRetentionSuite,
)


class EvaluatorRegistry:
    """Registry for evaluation benchmarks and general retention suites."""

    def __init__(self):
        self._suites: Dict[str, Type] = {
            "humaneval_plus": HumanEvalPlusSuite,
            "swe_bench_lite": SWEBenchSuite,
            "mbpp_plus": MBPPPlusSuite,
            "livecodebench": LiveCodeBenchSuite,
            "mmlu_pro": MMLUProRetentionSuite,
            "gsm8k": GSM8KRetentionSuite,
            "arc_challenge": ARCChallengeRetentionSuite,
        }
        self.discover_entrypoint_plugins()

    def register(self, name: str, suite_cls: Type) -> None:
        self._suites[name.lower()] = suite_cls

    def discover_entrypoint_plugins(self) -> int:
        """
        Discovers and registers third-party benchmark plugins via standard Python entry points:
        group = 'viforge.evaluators'
        """
        import importlib.metadata

        count = 0
        try:
            entry_points = importlib.metadata.entry_points()
            eps = (
                entry_points.select(group="viforge.evaluators")
                if hasattr(entry_points, "select")
                else entry_points.get("viforge.evaluators", [])  # type: ignore[attr-defined]
            )
            for ep in eps:
                try:
                    suite_cls = ep.load()
                    self.register(ep.name, suite_cls)
                    count += 1
                except Exception:
                    pass
        except Exception:
            pass
        return count

    def get(self, name: str):
        key = name.lower()
        if key not in self._suites:
            raise KeyError(
                f"Evaluation benchmark '{name}' not found. Available: {list(self._suites.keys())}"
            )
        return self._suites[key]()

    def list_all(self) -> list[str]:
        return list(self._suites.keys())


evaluator_registry = EvaluatorRegistry()

__all__ = [
    "HumanEvalPlusSuite",
    "SWEBenchSuite",
    "MBPPPlusSuite",
    "LiveCodeBenchSuite",
    "MMLUProRetentionSuite",
    "GSM8KRetentionSuite",
    "ARCChallengeRetentionSuite",
    "EvaluatorRegistry",
    "evaluator_registry",
]
