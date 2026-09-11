"""
Unit tests for Automated Hugging Face Model Card Generator and Artifact Lineage Tracker.
"""

from pathlib import Path
import pytest

from viforge.artifacts.manager import ArtifactManager
from viforge.config.schemas import (
    BenchmarkResult,
    ExperimentManifest,
    ExperimentSummaryReport,
    HardwareConfig,
    ModelConfig,
    StatisticalDelta,
    TrainingStageConfig,
)
from viforge.reporting.model_card import ModelCardGenerator


@pytest.fixture
def sample_manifest():
    return ExperimentManifest(
        experiment_id="test_model_card_exp",
        model=ModelConfig(
            name="DeepSeek V4 Pro Test",
            hf_hub_id="deepseek-ai/DeepSeek-V4-Pro",
            total_parameters=14.0,
            expected_license="Apache-2.0",
        ),
        hardware=HardwareConfig(num_gpus=4, accelerator="NVIDIA_A100_80GB"),
        pipeline=[
            TrainingStageConfig(
                stage_id="stage_sft",
                method="sft",
                dataset={"id": "ds_sft", "split": "train"},
            ),
            TrainingStageConfig(
                stage_id="stage_orpo",
                method="orpo",
                dataset={"id": "ds_orpo", "split": "train"},
            ),
        ],
    )


@pytest.fixture
def sample_summary():
    return ExperimentSummaryReport(
        experiment_id="test_model_card_exp",
        model_name="DeepSeek V4 Pro Test",
        baseline_domain_score=0.50,
        specialized_domain_score=0.68,
        domain_gain_pct=36.0,
        baseline_retention_score=0.65,
        specialized_retention_score=0.65,
        retention_delta_pct=0.0,
        total_training_cost_usd=24.50,
        total_wall_clock_hours=3.2,
        verdict="HYPOTHESIS CONFIRMED: Substantial domain gain achieved with zero retention degradation.",
        statistical_deltas=[
            StatisticalDelta(
                metric_name="humaneval_plus",
                baseline_value=0.50,
                specialized_value=0.68,
                absolute_delta=0.18,
                relative_delta_pct=36.0,
                ci_lower=0.62,
                ci_upper=0.74,
                p_value=0.002,
                is_significant=True,
            )
        ],
    )


@pytest.mark.unit
def test_model_card_content_and_yaml_frontmatter(sample_manifest, sample_summary):
    card = ModelCardGenerator.generate_model_card(sample_manifest, summary=sample_summary)
    assert "---" in card
    assert "base_model: deepseek-ai/DeepSeek-V4-Pro" in card
    assert "license: Apache-2.0" in card
    assert "DeepSeek V4 Pro Test (ViForge Specialist)" in card
    assert "+36.00%" in card
    assert "Statistical Benchmark Comparison" in card
    assert "humaneval_plus" in card
    assert "✅ Significant" in card
    assert "HYPOTHESIS CONFIRMED" in card


@pytest.mark.unit
def test_artifact_manager_lineage_and_card(sample_manifest, sample_summary, tmp_path: Path):
    mgr = ArtifactManager(tmp_path)
    card_path = mgr.generate_model_card(sample_manifest, summary=sample_summary)

    assert card_path.exists()
    assert (tmp_path / sample_manifest.experiment_id / "lineage.jsonl").exists()

    lineage = mgr.get_lineage(sample_manifest.experiment_id)
    assert len(lineage) >= 1
    assert lineage[0]["artifact_type"] == "model_card"
