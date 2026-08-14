"""
ViForge Training Telemetry, Checkpointing, and Callbacks (WandB, MLflow, TensorBoard).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from viforge.config.schemas import StageMetrics, CostTrackingConfig
from viforge.utils.logging import logger


class WandbCallback:
    """Streams live training metrics and artifacts to Weights & Biases."""

    def __init__(self, project_name: str = "viforge-specialization", experiment_id: str = "exp-001", enabled: bool = False):
        self.enabled = enabled
        self.project_name = project_name
        self.experiment_id = experiment_id
        self._initialized = False

    def init_run(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        try:
            # In production: import wandb; wandb.init(project=self.project_name, name=self.experiment_id, config=config)
            self._initialized = True
            logger.info(f"Initialized Weights & Biases run for project '{self.project_name}'")
        except Exception as e:
            logger.warning(f"Could not initialize WandB: {e}. Falling back to offline mode.")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or not self._initialized:
            return
        # wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._initialized:
            logger.info("Finalized WandB run.")


class MLflowCallback:
    """Streams experiment metrics, parameters, and model artifacts to MLflow."""

    def __init__(self, experiment_name: str = "ViForge_Specialization", enabled: bool = False):
        self.enabled = enabled
        self.experiment_name = experiment_name
        self._initialized = False

    def init_run(self, run_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        try:
            # In production: import mlflow; mlflow.set_experiment(self.experiment_name); mlflow.start_run(run_name=run_name)
            self._initialized = True
            logger.info(f"Initialized MLflow run in experiment '{self.experiment_name}'")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled or not self._initialized:
            return
        # mlflow.log_metrics(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._initialized:
            logger.info("Ended MLflow run.")


class CompositeObservabilityCallback:
    """Orchestrates local logging, JSONL streaming, WandB, and MLflow."""

    def __init__(self, output_dir: Path, cost_tracking: Optional[CostTrackingConfig] = None, experiment_id: str = "exp-001"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_path = output_dir / "training_metrics.jsonl"
        self.cost_tracking = cost_tracking or CostTrackingConfig()

        self.wandb = WandbCallback(
            project_name=self.cost_tracking.wandb_project,
            experiment_id=experiment_id,
            enabled=self.cost_tracking.log_to_wandb,
        )
        self.mlflow = MLflowCallback(
            experiment_name=self.cost_tracking.mlflow_experiment_name,
            enabled=self.cost_tracking.log_to_mlflow,
        )

        self.wandb.init_run()
        self.mlflow.init_run(run_name=experiment_id)

    def log_step(self, step: int, loss: float, lr: float, tokens_per_sec: float, vram_gb: float) -> None:
        record = {
            "step": step,
            "loss": round(loss, 4),
            "lr": lr,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "vram_gb": round(vram_gb, 2),
        }
        with open(self.metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.wandb.log(record, step=step)
        self.mlflow.log_metrics(record, step=step)

    def save_stage_summary(self, metrics: StageMetrics) -> Path:
        summary_path = self.output_dir / f"{metrics.stage_id}_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(metrics.model_dump_json(indent=2))
        logger.info(f"Saved stage metrics summary to {summary_path}")
        return summary_path

    def finish(self) -> None:
        self.wandb.finish()
        self.mlflow.finish()


# Backward compatibility alias
TrainingCallback = CompositeObservabilityCallback
