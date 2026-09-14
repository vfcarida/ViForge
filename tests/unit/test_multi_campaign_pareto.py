"""
Unit tests for MultiCampaignParetoComparator and CLI compare-campaigns command.
"""

from pathlib import Path
from typer.testing import CliRunner

from viforge.analysis.unified_pareto import (
    MultiCampaignParetoComparator,
    UnifiedParetoEngine,
    UnifiedParetoPoint,
)
from viforge.cli.main import app

runner = CliRunner()


def test_compute_global_frontier():
    """Test multi-objective Pareto dominance calculation across multiple model campaigns."""
    pts_model_a = [
        UnifiedParetoPoint(
            model_name="Model-A-7B",
            variant_type="base",
            variant_label="Model-A-Base",
            domain_score=0.50,
            general_retention_score=0.65,
            serving_memory_gb=14.0,
            serving_latency_ms=45.0,
            total_cost_usd=0.0,
        ),
        UnifiedParetoPoint(
            model_name="Model-A-7B",
            variant_type="specialist",
            variant_label="Model-A-Specialist",
            domain_score=0.72,
            general_retention_score=0.64,
            serving_memory_gb=14.0,
            serving_latency_ms=45.0,
            total_cost_usd=25.0,
        ),
    ]

    pts_model_b = [
        UnifiedParetoPoint(
            model_name="Model-B-1.3B",
            variant_type="specialist",
            variant_label="Model-B-Edge-Specialist",
            domain_score=0.55,
            general_retention_score=0.58,
            serving_memory_gb=3.0,
            serving_latency_ms=18.0,
            total_cost_usd=8.0,
        ),
        UnifiedParetoPoint(
            model_name="Model-B-1.3B",
            variant_type="dominated_variant",
            variant_label="Model-B-Poor",
            domain_score=0.40,
            general_retention_score=0.50,
            serving_memory_gb=16.0,  # worse memory than Model A and lower score
            serving_latency_ms=50.0,
            total_cost_usd=30.0,
        ),
    ]

    campaigns = {"Model-A": pts_model_a, "Model-B": pts_model_b}
    global_pts = MultiCampaignParetoComparator.compute_global_frontier(campaigns)

    opt_labels = {p.variant_label for p in global_pts if p.is_pareto_optimal}

    # Model-A-Specialist has highest accuracy (0.72) -> optimal
    assert "Model-A-Specialist" in opt_labels
    # Model-B-Edge-Specialist has lowest VRAM (3.0 GB) and latency (18 ms) -> optimal
    assert "Model-B-Edge-Specialist" in opt_labels
    # Model-B-Poor is strictly dominated by Model-A-Base and Model-A-Specialist
    assert "Model-B-Poor" not in opt_labels


def test_export_multi_model_json_and_html(tmp_path: Path):
    """Test exporting cross-model benchmark comparison to JSON and Plotly HTML."""
    pts_a = UnifiedParetoEngine.build_unified_points(
        model_name="Qwen2.5-Coder-7B",
        base_domain_score=0.50,
        base_retention_score=0.65,
        specialized_domain_score=0.68,
        specialized_retention_score=0.65,
        training_cost_usd=24.50,
    )
    pts_b = UnifiedParetoEngine.build_unified_points(
        model_name="Llama-3.1-8B",
        base_domain_score=0.46,
        base_retention_score=0.68,
        specialized_domain_score=0.61,
        specialized_retention_score=0.67,
        training_cost_usd=29.00,
    )

    campaigns = {"Qwen2.5-Coder-7B": pts_a, "Llama-3.1-8B": pts_b}
    global_pts = MultiCampaignParetoComparator.compute_global_frontier(campaigns)

    json_out = tmp_path / "multi_pareto.json"
    html_out = tmp_path / "multi_pareto.html"

    MultiCampaignParetoComparator.export_multi_model_json(campaigns, global_pts, json_out)
    MultiCampaignParetoComparator.generate_multi_model_html_chart(campaigns, global_pts, html_out)

    assert json_out.exists()
    assert html_out.exists()
    assert '"campaign_count": 2' in json_out.read_text(encoding="utf-8")
    assert "Qwen2.5-Coder-7B" in html_out.read_text(encoding="utf-8")


def test_cli_compare_campaigns(tmp_path: Path):
    """Test CLI command viforge compare-campaigns with multiple JSON files."""
    pts_1 = UnifiedParetoEngine.build_unified_points(
        model_name="Model-Alpha",
        base_domain_score=0.50,
        base_retention_score=0.65,
        specialized_domain_score=0.70,
        specialized_retention_score=0.65,
        training_cost_usd=20.0,
    )
    pts_2 = UnifiedParetoEngine.build_unified_points(
        model_name="Model-Beta",
        base_domain_score=0.40,
        base_retention_score=0.60,
        specialized_domain_score=0.58,
        specialized_retention_score=0.59,
        training_cost_usd=10.0,
    )

    f1 = tmp_path / "campaign1.json"
    f2 = tmp_path / "campaign2.json"
    UnifiedParetoEngine.export_json(pts_1, f1)
    UnifiedParetoEngine.export_json(pts_2, f2)

    out_dir = tmp_path / "comparison_output"
    res = runner.invoke(
        app,
        ["compare-campaigns", str(f1), str(f2), "--output-dir", str(out_dir)],
    )

    assert res.exit_code == 0
    assert "Cross-Model Multi-Campaign Pareto Benchmark" in res.output
    assert (out_dir / "multi_model_pareto.json").exists()
    assert (out_dir / "multi_model_pareto.html").exists()
