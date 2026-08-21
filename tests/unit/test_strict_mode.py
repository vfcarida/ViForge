"""
Unit tests for ViForge Strict Execution Mode.
"""

from pathlib import Path
import pytest
from viforge.experiments.runner import ExperimentRunner, StrictExecutionError
from viforge.config.schemas import (
    ExperimentManifest,
    HardwareConfig,
    ModelConfig,
    TrainingStageConfig,
)


def _make_sample_manifest(strict: bool = False) -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id="test-strict-mode-exp",
        model=ModelConfig(
            name="TestModel",
            hf_hub_id="test/model",
            total_parameters=7.0,
        ),
        hardware=HardwareConfig(accelerator="CPU", num_gpus=1),
        pipeline=[
            TrainingStageConfig(
                stage_id="stage_1",
                method="sft",
                dataset={"id": "ds1", "split": "train"},
            )
        ],
        strict_mode=strict,
    )


def test_strict_mode_raises_on_mock_baseline(tmp_path: Path):
    """Test that strict mode raises StrictExecutionError when mock backend is used."""
    manifest = _make_sample_manifest(strict=True)
    runner = ExperimentRunner(manifest, work_dir=tmp_path)

    with pytest.raises(StrictExecutionError, match="Strict execution mode is enabled"):
        runner.run_baseline_evaluation(backend_type="mock")


def test_strict_mode_raises_on_mock_specialized(tmp_path: Path):
    """Test that strict mode raises StrictExecutionError when mock specialized eval is run."""
    manifest = _make_sample_manifest(strict=True)
    runner = ExperimentRunner(manifest, work_dir=tmp_path)

    with pytest.raises(StrictExecutionError, match="Strict execution mode is enabled"):
        runner.run_specialized_evaluation(backend_type="mock")


def test_strict_mode_raises_on_execute(tmp_path: Path):
    """Test that execute(strict=True) raises StrictExecutionError on mock backend."""
    manifest = _make_sample_manifest(strict=False)
    runner = ExperimentRunner(manifest, work_dir=tmp_path)

    with pytest.raises(StrictExecutionError, match="cannot execute campaign with mock backend"):
        runner.execute(backend_type="mock", strict=True)


def test_non_strict_mode_succeeds_on_mock(tmp_path: Path):
    """Test that non-strict mode runs mock baseline evaluation without error."""
    manifest = _make_sample_manifest(strict=False)
    runner = ExperimentRunner(manifest, work_dir=tmp_path)

    results = runner.run_baseline_evaluation(backend_type="mock", limit=2)
    assert len(results) > 0
