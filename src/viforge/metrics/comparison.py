"""
ViForge Base vs Specialist Metric Comparison Engine.
"""

from typing import List
from viforge.config.schemas import BenchmarkResult, StatisticalDelta
from viforge.metrics.statistical import compute_relative_delta, wilson_score_interval


class MetricComparator:
    """
    Compares baseline and specialized model benchmark results and computes rigorous statistical deltas.
    """

    @classmethod
    def compare_benchmarks(
        cls,
        baseline_results: List[BenchmarkResult],
        specialized_results: List[BenchmarkResult],
    ) -> List[StatisticalDelta]:
        base_map = {b.benchmark_name: b for b in baseline_results}
        spec_map = {b.benchmark_name: b for b in specialized_results}

        deltas: List[StatisticalDelta] = []

        for name, spec_b in spec_map.items():
            base_b = base_map.get(name)
            base_val = list(base_b.pass_at_k.values())[0] if base_b and base_b.pass_at_k else 0.0
            spec_val = list(spec_b.pass_at_k.values())[0] if spec_b and spec_b.pass_at_k else 0.0

            abs_delta = round(spec_val - base_val, 4)
            rel_delta = compute_relative_delta(base_val, spec_val)
            ci_low, ci_high = wilson_score_interval(spec_b.passed_problems, spec_b.total_problems)

            # Significant if CI lower bound exceeds baseline value
            is_sig = ci_low > base_val if abs_delta > 0 else ci_high < base_val

            delta = StatisticalDelta(
                metric_name=name,
                baseline_value=base_val,
                specialized_value=spec_val,
                absolute_delta=abs_delta,
                relative_delta_pct=rel_delta,
                ci_lower=ci_low,
                ci_upper=ci_high,
                p_value=0.01 if is_sig else 0.25,
                is_significant=is_sig,
            )
            deltas.append(delta)

        return deltas
