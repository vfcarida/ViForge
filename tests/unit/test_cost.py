"""
Unit tests for ViForge Cost Model, Pareto Frontier, and Statistical Metrics.
"""

from viforge.cost.cost_model import CostModel
from viforge.config.schemas import BasePricingConfig, ParetoPoint
from viforge.analysis.pareto import ParetoEngine
from viforge.metrics.statistical import wilson_score_interval, compute_relative_delta


def test_cost_model():
    train_cost = CostModel.calculate_training_cost(wall_clock_seconds=3600, hourly_rate=7.40)
    assert train_cost == 7.40

    data_cost = CostModel.calculate_data_generation_cost(num_queries=1000)
    assert data_cost > 0

    pricing = BasePricingConfig(prompt_usd_per_1m=0.14, completion_usd_per_1m=0.28)
    infer_cost = CostModel.calculate_inference_cost(1_000_000, 1_000_000, pricing)
    assert round(infer_cost, 2) == 0.42

    cap_usd = CostModel.compute_capability_per_dollar(
        domain_gain=25.0, retention_delta=0.0, total_experiment_cost=50.0
    )
    assert cap_usd > 0


def test_pareto_engine():
    points = [
        ParetoPoint(
            model_name="M1",
            stage_or_variant="Variant A",
            domain_score=0.80,
            general_retention_score=0.75,
            training_cost_usd=10.0,
            inference_cost_per_1m=0.20,
            latency_p50_ms=100.0,
            memory_footprint_gb=16.0,
            capability_per_dollar=50.0,
        ),
        ParetoPoint(
            model_name="M2",
            stage_or_variant="Variant B (Dominated)",
            domain_score=0.70,
            general_retention_score=0.70,
            training_cost_usd=20.0,
            inference_cost_per_1m=0.30,
            latency_p50_ms=150.0,
            memory_footprint_gb=32.0,
            capability_per_dollar=20.0,
        ),
    ]
    frontier = ParetoEngine.identify_pareto_frontier(points)
    assert frontier[0].is_pareto_optimal is True
    assert frontier[1].is_pareto_optimal is False


def test_statistical_metrics():
    low, high = wilson_score_interval(80, 100)
    assert 0.70 < low < 0.80
    assert 0.80 < high < 0.90

    # Boundary conditions
    low_0, high_0 = wilson_score_interval(0, 0)
    assert low_0 == 0.0 and high_0 == 0.0

    low_zero, high_zero = wilson_score_interval(0, 10)
    assert low_zero == 0.0 and high_zero > 0.0

    low_full, high_full = wilson_score_interval(10, 10)
    assert low_full < 1.0 and high_full == 1.0

    rel_delta = compute_relative_delta(0.50, 0.75)
    assert rel_delta == 50.0

    rel_delta_zero = compute_relative_delta(0.0, 0.50)
    assert rel_delta_zero == 100.0
