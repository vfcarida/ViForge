"""
Unit tests for ViForge Cost Model, Pareto Frontier, and Statistical Metrics.
"""

import pytest

from viforge.analysis.pareto import ParetoEngine
from viforge.config.schemas import BasePricingConfig, ParetoPoint
from viforge.cost.cost_model import CostModel
from viforge.metrics.statistical import compute_relative_delta, wilson_score_interval


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_pass_at_k_estimation():
    from viforge.metrics.statistical import compute_pass_at_k_distribution, estimate_pass_at_k

    # Test c=0 -> 0.0
    assert estimate_pass_at_k(n=10, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=10, c=0, k=5) == 0.0

    # Test c=n -> 1.0
    assert estimate_pass_at_k(n=10, c=10, k=1) == 1.0
    assert estimate_pass_at_k(n=10, c=10, k=5) == 1.0

    # Test n=10, c=1, k=1 -> 1/10 = 0.10
    assert estimate_pass_at_k(n=10, c=1, k=1) == 0.10

    # Test n=10, c=5, k=1 -> 0.50
    assert estimate_pass_at_k(n=10, c=5, k=1) == 0.50

    # Test n - c < k -> 1.0 (e.g. n=10, c=8, k=5 -> 10-8=2 < 5)
    assert estimate_pass_at_k(n=10, c=8, k=5) == 1.0

    # Test ValueError when n < k
    with pytest.raises(ValueError, match="Total samples n"):
        estimate_pass_at_k(n=3, c=1, k=5)

    # Test distribution computation across multiple problems
    problem_results = [
        [True, False, False, False, False],  # problem 1: c=1/5
        [True, True, True, True, True],  # problem 2: c=5/5
        [False, False, False, False, False],  # problem 3: c=0/5
    ]
    dist = compute_pass_at_k_distribution(problem_results, k_list=[1, 5])
    assert "pass@1" in dist
    assert "pass@5" in dist
    assert 0.0 < dist["pass@1"] < 1.0
    assert 0.0 < dist["pass@5"] <= 1.0
