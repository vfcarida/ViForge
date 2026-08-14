"""
Unit tests for Observability Callbacks (WandB, MLflow, Local JSONL) and LigerKernelManager.
"""

import tempfile
from pathlib import Path
from viforge.config.schemas import CostTrackingConfig, StageMetrics
from viforge.training.callbacks import (
    WandbCallback,
    MLflowCallback,
    CompositeObservabilityCallback,
)
from viforge.training.backends import LigerKernelManager, BackendManager
from viforge.config.schemas import HardwareConfig


def test_observability_callbacks():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "metrics"
        cfg = CostTrackingConfig(log_to_wandb=True, log_to_mlflow=True)
        cb = CompositeObservabilityCallback(output_dir=out_dir, cost_tracking=cfg, experiment_id="test-exp")

        cb.log_step(step=1, loss=0.5432, lr=1e-4, tokens_per_sec=1200.0, vram_gb=14.2)
        cb.log_step(step=2, loss=0.4321, lr=9e-5, tokens_per_sec=1250.0, vram_gb=14.3)

        assert cb.metrics_log_path.exists()
        with open(cb.metrics_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2

        stage_metrics = StageMetrics(
            stage_id="stage_1",
            method="sft",
            training_loss=0.4321,
            eval_loss=0.4100,
            tokens_processed=100000,
            tokens_per_second=1250.0,
            peak_vram_gb=14.3,
            trainable_parameters=10000000,
            total_parameters=14000000000,
            trainable_ratio_pct=0.071,
            wall_clock_seconds=120.0,
            estimated_stage_cost_usd=0.25,
        )
        summary_path = cb.save_stage_summary(stage_metrics)
        assert summary_path.exists()
        cb.finish()


def test_wandb_and_mlflow_standalone():
    wb = WandbCallback(enabled=False)
    wb.init_run()
    wb.log({"test_loss": 0.5})
    wb.finish()

    mf = MLflowCallback(enabled=False)
    mf.init_run(run_name="test")
    mf.log_metrics({"test_metric": 1.0})
    mf.finish()


def test_liger_kernel_manager():
    # Graceful fallback when liger is not installed or enabled=False
    assert LigerKernelManager.apply_liger_kernel_to_model(enable=False) is False
    res = LigerKernelManager.apply_liger_kernel_to_model(enable=True)
    assert isinstance(res, bool)

    hw = HardwareConfig(accelerator="NVIDIA_A100_80GB")
    backend_info = BackendManager.setup_backend(hw, use_liger=True)
    assert "liger_kernel_active" in backend_info
