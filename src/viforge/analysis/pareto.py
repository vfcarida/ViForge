"""
ViForge Multi-Objective Pareto Frontier Analysis Engine.
"""

from typing import List
from viforge.config.schemas import ParetoPoint
from viforge.utils.logging import logger


class ParetoEngine:
    """
    Computes non-dominated Pareto-optimal models across quality, retention, cost, latency, and memory.
    """

    @classmethod
    def identify_pareto_frontier(cls, points: List[ParetoPoint]) -> List[ParetoPoint]:
        if not points:
            return []

        results = []
        for i, pt_a in enumerate(points):
            is_dominated = False
            for j, pt_b in enumerate(points):
                if i == j:
                    continue

                b_better_or_equal = (
                    pt_b.domain_score >= pt_a.domain_score
                    and pt_b.general_retention_score >= pt_a.general_retention_score
                    and pt_b.training_cost_usd <= pt_a.training_cost_usd
                    and pt_b.latency_p50_ms <= pt_a.latency_p50_ms
                    and pt_b.memory_footprint_gb <= pt_a.memory_footprint_gb
                )

                b_strictly_better = (
                    pt_b.domain_score > pt_a.domain_score
                    or pt_b.general_retention_score > pt_a.general_retention_score
                    or pt_b.training_cost_usd < pt_a.training_cost_usd
                    or pt_b.latency_p50_ms < pt_a.latency_p50_ms
                    or pt_b.memory_footprint_gb < pt_a.memory_footprint_gb
                )

                if b_better_or_equal and b_strictly_better:
                    is_dominated = True
                    break

            pt_copy = pt_a.model_copy()
            pt_copy.is_pareto_optimal = not is_dominated
            results.append(pt_copy)

        optimal_count = sum(1 for p in results if p.is_pareto_optimal)
        logger.info(f"Pareto analysis: {optimal_count}/{len(results)} points on the optimal frontier.")
        return results
