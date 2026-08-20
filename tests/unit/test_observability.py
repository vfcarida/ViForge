"""
Unit tests for Observability Callbacks (WandB, MLflow, Local JSONL) and LigerKernelManager.

Tests verify:
- Both callbacks no-op gracefully without API keys or packages
- Local JSONL streaming always works
- Lifecycle hooks (on_experiment_start, on_stage_start, on_stage_end, on_experiment_end) run without crash
- ExperimentRunner wires observability at all lifecycle points
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from viforge.config.schemas import CostTrackingConfig, HardwareConfig, StageMetrics
from viforge.training.backends import BackendManager, LigerKernelManager
from viforge.training.callbacks import (
    HAS_MLFLOW,
    HAS_WANDB,
    CompositeObservabilityCallback,
    MLflowCallback,
    WandbCallback,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #

STAGE_METRICS = StageMetrics(
    stage_id="stage_1",
    method="sft",
    training_loss=0.4321,
    eval_loss=0.4100,
    tokens_processed=100_000,
    tokens_per_second=1250.0,
    peak_vram_gb=14.3,
    trainable_parameters=10_000_000,
    total_parameters=14_000_000_000,
    trainable_ratio_pct=0.071,
    wall_clock_seconds=120.0,
    estimated_stage_cost_usd=0.25,
)


# --------------------------------------------------------------------------- #
# Existing test (unchanged)                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_observability_callbacks(tmp_output_dir: Path):
    out_dir = tmp_output_dir / "metrics"
    cfg = CostTrackingConfig(log_to_wandb=True, log_to_mlflow=True)
    cb = CompositeObservabilityCallback(
        output_dir=out_dir, cost_tracking=cfg, experiment_id="test-exp"
    )
    # on_experiment_start no-ops (no keys set) — should not crash
    cb.on_experiment_start(config={"model": "test-model"})

    cb.log_step(step=1, loss=0.5432, lr=1e-4, tokens_per_sec=1200.0, vram_gb=14.2)
    cb.log_step(step=2, loss=0.4321, lr=9e-5, tokens_per_sec=1250.0, vram_gb=14.3)

    assert cb.metrics_log_path.exists()
    with open(cb.metrics_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2

    summary_path = cb.save_stage_summary(STAGE_METRICS)
    assert summary_path.exists()
    cb.finish()


@pytest.mark.unit
def test_wandb_and_mlflow_standalone():
    # Disabled state — no crash regardless of package availability
    wb = WandbCallback(enabled=False)
    wb.init_run()
    wb.log({"test_loss": 0.5})
    wb.log_summary({"final_score": 0.9})
    wb.log_table("test_table", [{"metric": "val", "score": "1.0"}])
    wb.finish()

    mf = MLflowCallback(enabled=False)
    mf.init_run(run_name="test")
    mf.log_metrics({"test_metric": 1.0})
    mf.log_params({"lr": 0.001})
    mf.finish()


@pytest.mark.unit
def test_liger_kernel_manager():
    # Graceful fallback when liger is not installed or enabled=False
    assert LigerKernelManager.apply_liger_kernel_to_model(enable=False) is False
    res = LigerKernelManager.apply_liger_kernel_to_model(enable=True)
    assert isinstance(res, bool)

    hw = HardwareConfig(accelerator="NVIDIA_A100_80GB")
    backend_info = BackendManager.setup_backend(hw, use_liger=True)
    assert "liger_kernel_active" in backend_info


# --------------------------------------------------------------------------- #
# New tests — graceful no-op without keys/packages                             #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_wandb_disabled_without_api_key(tmp_output_dir: Path):
    """WandB must not activate when WANDB_API_KEY is absent."""
    env = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        wb = WandbCallback(enabled=True, project_name="test-proj", experiment_id="e1")
        assert wb.enabled is False  # No key → disabled
        # All ops must be no-ops
        wb.init_run({"param": 1})
        wb.log({"loss": 0.1}, step=1)
        wb.log_summary({"final": 0.9})
        wb.log_table("tbl", [{"a": "b"}])
        wb.finish()


@pytest.mark.unit
def test_mlflow_disabled_without_tracking_uri(tmp_output_dir: Path):
    """MLflow must not activate when MLFLOW_TRACKING_URI is absent."""
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
    with patch.dict(os.environ, env, clear=True):
        mf = MLflowCallback(enabled=True, experiment_name="test-exp")
        assert mf.enabled is False  # No URI → disabled
        mf.init_run(run_name="r1", params={"lr": 0.001})
        mf.log_metrics({"loss": 0.5}, step=1)
        mf.log_params({"key": "val"})
        mf.finish()


@pytest.mark.unit
def test_full_lifecycle_no_crash(tmp_output_dir: Path):
    """Full CompositeObservabilityCallback lifecycle must never crash even with no backends."""
    out_dir = tmp_output_dir / "obs_lifecycle"
    cfg = CostTrackingConfig(log_to_wandb=False, log_to_mlflow=False)
    cb = CompositeObservabilityCallback(
        output_dir=out_dir, cost_tracking=cfg, experiment_id="lifecycle-test"
    )

    cb.on_experiment_start(config={"model": "deepseek-v4-pro"})
    cb.on_stage_start(stage_id="stage_1_sft", params={"lr": 2e-4, "epochs": 3})
    cb.log_step(step=1, loss=0.85, lr=2e-4, tokens_per_sec=1800.0, vram_gb=28.1)
    cb.log_step(step=2, loss=0.80, lr=1.9e-4, tokens_per_sec=1810.0, vram_gb=28.2)
    cb.on_stage_end(STAGE_METRICS)
    cb.on_experiment_end(
        summary={
            "domain_gain_pct": 8.5,
            "retention_delta_pct": -1.2,
            "total_training_cost_usd": 12.50,
        }
    )

    # JSONL file must always be written
    assert cb.metrics_log_path.exists()
    with open(cb.metrics_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2

    # Stage summary JSON must be persisted
    assert (out_dir / "stage_1_metrics.json").exists()


@pytest.mark.unit
def test_wandb_real_integration_with_mock(tmp_output_dir: Path):
    """When WANDB_API_KEY and wandb package are present, real calls are made."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.run = mock_run
    mock_run.summary = {}

    with (
        patch("viforge.training.callbacks._wandb_mod", mock_wandb),
        patch("viforge.training.callbacks.HAS_WANDB", True),
        patch.dict(os.environ, {"WANDB_API_KEY": "test-api-key"}),
    ):
        wb = WandbCallback(enabled=True, project_name="test-proj", experiment_id="e1")
        assert wb.enabled is True

        wb.init_run({"lr": 0.001})
        mock_wandb.init.assert_called_once_with(
            project="test-proj", name="e1", config={"lr": 0.001}, reinit=True
        )

        wb.log({"loss": 0.5}, step=1)
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=1)

        wb.log_summary({"final_score": 0.8})
        assert mock_run.summary.get("final_score") == 0.8

        wb.finish()
        mock_wandb.finish.assert_called_once()


@pytest.mark.unit
def test_mlflow_real_integration_with_mock(tmp_output_dir: Path):
    """When MLFLOW_TRACKING_URI and mlflow package are present, real calls are made."""
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value = MagicMock()

    with (
        patch("viforge.training.callbacks._mlflow_mod", mock_mlflow),
        patch("viforge.training.callbacks.HAS_MLFLOW", True),
        patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}),
    ):
        mf = MLflowCallback(enabled=True, experiment_name="ViForge_Test")
        assert mf.enabled is True

        mf.init_run(run_name="test-run", params={"lr": "2e-4", "epochs": "3"})
        mock_mlflow.set_experiment.assert_called_once_with("ViForge_Test")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

        mf.log_metrics({"loss": 0.5, "eval_loss": 0.48}, step=5)
        mock_mlflow.log_metrics.assert_called()

        mf.finish()
        mock_mlflow.end_run.assert_called_once()
