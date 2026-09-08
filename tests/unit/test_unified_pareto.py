"""
Unit tests for ViForge Unified Pareto Analysis Engine.
"""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from viforge.analysis.unified_pareto import (
    SweetSpotRecommendation,
    UnifiedParetoEngine,
    UnifiedParetoPoint,
)
from viforge.cli.main import app


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_calculate_metrics():
    """Verify calculated metrics (gain, throughput, cap/dollar, efficiency)."""
    metrics = UnifiedParetoEngine.calculate_metrics(
        domain_score=0.68,
        general_retention_score=0.65,
        total_cost_usd=24.50,
        serving_memory_gb=28.1,
        serving_latency_ms=124.0,
        base_domain=0.50,
        base_retention=0.65,
    )

    assert metrics["domain_gain_pct"] == 36.0
    assert metrics["retention_delta_pct"] == 0.0
    assert metrics["serving_throughput_tok_s"] > 0
    assert metrics["capability_per_dollar"] > 0
    assert metrics["efficiency_index"] > 0


def test_build_unified_points():
    """Verify construction of complete lifecycle cohort."""
    points = UnifiedParetoEngine.build_unified_points(
        model_name="DeepSeek-Coder-7B",
        base_domain_score=0.50,
        base_retention_score=0.65,
        specialized_domain_score=0.68,
        specialized_retention_score=0.65,
        training_cost_usd=24.50,
    )

    assert len(points) >= 5
    types = [p.variant_type for p in points]
    assert "base" in types
    assert "specialist" in types
    assert any("smoothquant" in t for t in types)
    assert any("autoround" in t for t in types)

    # Check non-domination flag
    assert any(p.is_pareto_optimal for p in points)


def test_identify_pareto_frontier():
    """Verify 5D non-domination algorithm."""
    pt_dominated = UnifiedParetoPoint(
        model_name="TestModel",
        variant_type="variant_a",
        variant_label="Inferior Variant",
        domain_score=0.40,
        general_retention_score=0.50,
        total_cost_usd=50.0,
        serving_memory_gb=40.0,
        serving_latency_ms=200.0,
    )
    pt_superior = UnifiedParetoPoint(
        model_name="TestModel",
        variant_type="variant_b",
        variant_label="Superior Variant",
        domain_score=0.70,
        general_retention_score=0.70,
        total_cost_usd=10.0,
        serving_memory_gb=10.0,
        serving_latency_ms=50.0,
    )

    frontier = UnifiedParetoEngine.identify_pareto_frontier([pt_dominated, pt_superior])
    assert len(frontier) == 2
    by_type = {p.variant_type: p.is_pareto_optimal for p in frontier}
    assert by_type["variant_b"] is True
    assert by_type["variant_a"] is False


def test_find_sweet_spots():
    """Verify detection of canonical deployment sweet spots."""
    points = UnifiedParetoEngine.build_unified_points(
        model_name="DeepSeek-Coder-7B",
    )
    spots = UnifiedParetoEngine.find_sweet_spots(points)
    assert len(spots) >= 3

    categories = [s.category for s in spots]
    assert any("Max Domain Capability" in c for c in categories)
    assert any("Edge" in c for c in categories)
    assert any("Ultra-Throughput" in c for c in categories)


def test_export_json_and_html(tmp_path: Path):
    """Test exporting Pareto data to JSON and standalone HTML."""
    points = UnifiedParetoEngine.build_unified_points(model_name="TestModel")
    json_path = UnifiedParetoEngine.export_json(points, tmp_path / "pareto.json")
    html_path = UnifiedParetoEngine.generate_html_chart(points, tmp_path / "pareto.html")

    assert json_path.exists()
    assert html_path.exists()
    assert json_path.stat().st_size > 100
    assert html_path.stat().st_size > 100


def test_cli_pareto_unified(cli_runner, tmp_path: Path):
    """Test CLI pareto-unified command."""
    manifest_path = Path("configs/experiments/exp_003_deepseek_v4_pro_cpt_sft_dpo.yaml")
    out_dir = tmp_path / "reports_test"

    result = cli_runner.invoke(
        app,
        [
            "pareto-unified",
            str(manifest_path),
            "--domain-gain",
            "36.0",
            "--cost",
            "24.50",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Unified Pareto Frontier" in result.output
    assert "Sweet-Spot Recommendations" in result.output
    assert (out_dir / "unified_pareto.json").exists()
    assert (out_dir / "unified_pareto.html").exists()
