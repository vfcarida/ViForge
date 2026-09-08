"""
ViForge Base vs Specialist Metric Comparison Engine.
"""

from typing import List
from viforge.config.schemas import BenchmarkResult, StatisticalDelta
from viforge.metrics.statistical import (
    compute_relative_delta,
    mcnemar_test,
    two_proportion_z_test,
    wilson_score_interval,
)


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

            p_val = 1.0
            if base_b:
                base_outcomes = base_b.raw_metrics.get("problem_outcomes")
                spec_outcomes = spec_b.raw_metrics.get("problem_outcomes")
                if (
                    isinstance(base_outcomes, list)
                    and isinstance(spec_outcomes, list)
                    and len(base_outcomes) == len(spec_outcomes)
                ):
                    b_disc = sum(1 for bo, so in zip(base_outcomes, spec_outcomes) if bo and not so)
                    c_disc = sum(1 for bo, so in zip(base_outcomes, spec_outcomes) if not bo and so)
                    _, p_val = mcnemar_test(b_disc, c_disc)
                elif base_b.total_problems == spec_b.total_problems:
                    b_disc = max(0, base_b.passed_problems - spec_b.passed_problems)
                    c_disc = max(0, spec_b.passed_problems - base_b.passed_problems)
                    _, p_val = mcnemar_test(b_disc, c_disc)
                else:
                    _, p_val = two_proportion_z_test(
                        base_b.passed_problems,
                        base_b.total_problems,
                        spec_b.passed_problems,
                        spec_b.total_problems,
                    )

            # A result is statistically significant if p < 0.05 and improvement is observed,
            # or if confidence interval lower bound strictly exceeds baseline for positive delta
            is_sig = (p_val < 0.05 and abs_delta > 0) or (
                ci_low > base_val if abs_delta > 0 else ci_high < base_val
            )

            delta = StatisticalDelta(
                metric_name=name,
                baseline_value=base_val,
                specialized_value=spec_val,
                absolute_delta=abs_delta,
                relative_delta_pct=rel_delta,
                ci_lower=ci_low,
                ci_upper=ci_high,
                p_value=round(p_val, 4),
                is_significant=is_sig,
            )
            deltas.append(delta)

        return deltas

