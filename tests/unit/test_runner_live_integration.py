"""
Integration tests for ExperimentRunner end-to-end execution, preflight checks,
live model execution wiring, and strict mode compliance.
"""

from pathlib import Path
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from viforge.config.loader import ConfigLoader
from viforge.config.schemas import ExperimentSummaryReport
from viforge.experiments.runner import ExperimentRunner, StrictExecutionError


def _create_tiny_llama() -> LlamaForCausalLM:
    """Create a minimal in-memory causal LM for fast verification."""
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(config)


@pytest.fixture
def baseline_manifest_path() -> Path:
    p = Path("configs/experiments/exp_001_deepseek_v4_pro_baseline.yaml")
    if not p.exists():
        pytest.skip("Baseline experiment configuration not found.")
    return p


def test_experiment_runner_preflight_checks(baseline_manifest_path: Path, tmp_path: Path):
    runner = ExperimentRunner.from_yaml(baseline_manifest_path, work_dir=tmp_path)
    preflight = runner.run_preflight_checks()

    assert "stages" in preflight
    assert len(preflight["stages"]) == len(runner.manifest.pipeline)
    for stage_prof in preflight["stages"]:
        assert "stage_id" in stage_prof
        assert "vram" in stage_prof
        assert "cost" in stage_prof


def test_experiment_runner_execute_mock_end_to_end(baseline_manifest_path: Path, tmp_path: Path):
    manifest = ConfigLoader.load_manifest(baseline_manifest_path)
    # Use minimal evaluation problems for rapid execution
    manifest.evaluation.max_problems = 1
    manifest.evaluation.domain_benchmarks = manifest.evaluation.domain_benchmarks[:1]
    manifest.evaluation.general_retention_benchmarks = (
        manifest.evaluation.general_retention_benchmarks[:1]
    )
    runner = ExperimentRunner(manifest, work_dir=tmp_path)

    report = runner.execute(backend_type="mock")
    assert isinstance(report, ExperimentSummaryReport)
    assert report.experiment_id == manifest.experiment_id
    assert len(report.stages) == len(manifest.pipeline)
    assert len(report.benchmark_results) > 0
    assert len(report.pareto_frontier) > 0
    assert report.total_training_cost_usd >= 0.0

    # Verify generated report artifacts on disk
    reports_dir = tmp_path / manifest.experiment_id / "reports"
    assert (reports_dir / f"{manifest.experiment_id}_report.json").exists()
    assert (reports_dir / f"{manifest.experiment_id}_report.md").exists()


def test_experiment_runner_run_training_stages_live(baseline_manifest_path: Path, tmp_path: Path):
    manifest = ConfigLoader.load_manifest(baseline_manifest_path)
    runner = ExperimentRunner(manifest, work_dir=tmp_path)
    tiny_model = _create_tiny_llama()

    # Run training stages with live tiny model
    stage_results = runner.run_training_stages(model=tiny_model, live=True)
    assert len(stage_results) == len(manifest.pipeline)
    for stage_id, stage_data in stage_results.items():
        metrics = stage_data["metrics"]
        assert metrics.stage_id == stage_id
        assert metrics.wall_clock_seconds > 0.0


def test_experiment_runner_strict_mode_rejects_mock(baseline_manifest_path: Path, tmp_path: Path):
    runner = ExperimentRunner.from_yaml(baseline_manifest_path, work_dir=tmp_path)
    with pytest.raises(StrictExecutionError):
        runner.execute(backend_type="mock", strict=True)
