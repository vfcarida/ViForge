"""
ViForge Training Telemetry, Checkpointing, and Callbacks (WandB, MLflow, TensorBoard).

WandB and MLflow integrations activate when:
  - WandB: WANDB_API_KEY env var is set AND wandb package is installed
  - MLflow: MLFLOW_TRACKING_URI env var is set AND mlflow package is installed

Both fail gracefully (no-op) if keys/packages are absent.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from viforge.config.schemas import CostTrackingConfig, StageMetrics
from viforge.utils.logging import logger

# --------------------------------------------------------------------------- #
# Optional package detection                                                    #
# --------------------------------------------------------------------------- #

try:
    import wandb as _wandb_mod

    HAS_WANDB = True
except ImportError:
    _wandb_mod = None  # type: ignore[assignment]
    HAS_WANDB = False

try:
    import mlflow as _mlflow_mod

    HAS_MLFLOW = True
except ImportError:
    _mlflow_mod = None  # type: ignore[assignment]
    HAS_MLFLOW = False


# --------------------------------------------------------------------------- #
# WandB Callback                                                                #
# --------------------------------------------------------------------------- #


class WandbCallback:
    """Streams live training metrics and artifacts to Weights & Biases.

    Activates only when:
    - ``enabled=True`` is passed (controlled by CostTrackingConfig.log_to_wandb)
    - ``wandb`` package is installed
    - ``WANDB_API_KEY`` environment variable is set
    """

    def __init__(
        self,
        project_name: str = "viforge-specialization",
        experiment_id: str = "exp-001",
        enabled: bool = False,
    ):
        self.project_name = project_name
        self.experiment_id = experiment_id
        self._initialized = False

        api_key_present = bool(os.environ.get("WANDB_API_KEY"))
        self.enabled = enabled and HAS_WANDB and api_key_present

        if enabled and not HAS_WANDB:
            logger.warning(
                "WandB logging requested but 'wandb' package is not installed. "
                "Falling back to offline mode. Install with: pip install wandb"
            )
        elif enabled and not api_key_present:
            logger.warning(
                "WandB logging requested but WANDB_API_KEY is not set. "
                "Falling back to offline mode."
            )

    def init_run(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        try:
            _wandb_mod.init(
                project=self.project_name,
                name=self.experiment_id,
                config=config or {},
                reinit=True,
            )
            self._initialized = True
            logger.info(
                f"WandB: initialized run '{self.experiment_id}' in project '{self.project_name}'."
            )
        except Exception as e:
            logger.warning(f"WandB: could not initialize run: {e}. Falling back to offline mode.")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or not self._initialized:
            return
        try:
            _wandb_mod.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"WandB: failed to log metrics at step {step}: {e}")

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log final experiment-level summary values."""
        if not self.enabled or not self._initialized:
            return
        try:
            for key, value in summary.items():
                _wandb_mod.run.summary[key] = value  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(f"WandB: failed to log summary: {e}")

    def log_table(self, table_name: str, data: list[Dict[str, Any]]) -> None:
        """Log a comparison table artifact."""
        if not self.enabled or not self._initialized or not data:
            return
        try:
            columns = list(data[0].keys())
            rows = [list(row.values()) for row in data]
            table = _wandb_mod.Table(columns=columns, data=rows)
            _wandb_mod.log({table_name: table})
        except Exception as e:
            logger.warning(f"WandB: failed to log table '{table_name}': {e}")

    def finish(self) -> None:
        if not self.enabled or not self._initialized:
            return
        try:
            _wandb_mod.finish()
            logger.info("WandB: finalized run.")
        except Exception as e:
            logger.warning(f"WandB: error finishing run: {e}")


# --------------------------------------------------------------------------- #
# MLflow Callback                                                               #
# --------------------------------------------------------------------------- #


class MLflowCallback:
    """Streams experiment metrics, parameters, and model artifacts to MLflow.

    Activates only when:
    - ``enabled=True`` is passed (controlled by CostTrackingConfig.log_to_mlflow)
    - ``mlflow`` package is installed
    - ``MLFLOW_TRACKING_URI`` environment variable is set
    """

    def __init__(
        self,
        experiment_name: str = "ViForge_Specialization",
        enabled: bool = False,
    ):
        self.experiment_name = experiment_name
        self._initialized = False
        self._run = None

        tracking_uri_present = bool(os.environ.get("MLFLOW_TRACKING_URI"))
        self.enabled = enabled and HAS_MLFLOW and tracking_uri_present

        if enabled and not HAS_MLFLOW:
            logger.warning(
                "MLflow logging requested but 'mlflow' package is not installed. "
                "Falling back to offline mode. Install with: pip install mlflow"
            )
        elif enabled and not tracking_uri_present:
            logger.warning(
                "MLflow logging requested but MLFLOW_TRACKING_URI is not set. "
                "Falling back to offline mode."
            )

    def init_run(self, run_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        try:
            _mlflow_mod.set_experiment(self.experiment_name)
            self._run = _mlflow_mod.start_run(run_name=run_name)
            if params:
                flat_params = {str(k): str(v) for k, v in params.items() if v is not None}
                _mlflow_mod.log_params(flat_params)
            self._initialized = True
            logger.info(f"MLflow: started run '{run_name}' in experiment '{self.experiment_name}'.")
        except Exception as e:
            logger.warning(f"MLflow: could not initialize run: {e}. Falling back to offline mode.")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled or not self._initialized:
            return
        try:
            # MLflow only accepts numeric metrics; filter out non-float values
            numeric_metrics = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric_metrics:
                _mlflow_mod.log_metrics(numeric_metrics, step=step)
        except Exception as e:
            logger.warning(f"MLflow: failed to log metrics at step {step}: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment-level parameters."""
        if not self.enabled or not self._initialized:
            return
        try:
            flat_params = {str(k): str(v) for k, v in params.items() if v is not None}
            _mlflow_mod.log_params(flat_params)
        except Exception as e:
            logger.warning(f"MLflow: failed to log params: {e}")

    def finish(self) -> None:
        if not self.enabled or not self._initialized:
            return
        try:
            _mlflow_mod.end_run()
            logger.info("MLflow: ended run.")
        except Exception as e:
            logger.warning(f"MLflow: error ending run: {e}")


# --------------------------------------------------------------------------- #
# Composite Orchestrator                                                        #
# --------------------------------------------------------------------------- #


class CompositeObservabilityCallback:
    """Orchestrates local JSONL streaming, WandB, and MLflow observability backends.

    Lifecycle hooks:
    - ``on_experiment_start(config)``  — called when experiment begins
    - ``on_stage_start(stage_id, params)`` — called before each training stage
    - ``log_step(step, loss, lr, ...)``  — called per training step
    - ``on_stage_end(metrics)``        — called after each training stage
    - ``on_experiment_end(summary)``   — called when experiment completes
    - ``finish()``                     — finalizes all backends
    """

    def __init__(
        self,
        output_dir: Path,
        cost_tracking: Optional[CostTrackingConfig] = None,
        experiment_id: str = "exp-001",
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_path = output_dir / "training_metrics.jsonl"
        self.cost_tracking = cost_tracking or CostTrackingConfig()
        self.experiment_id = experiment_id

        self.wandb = WandbCallback(
            project_name=self.cost_tracking.wandb_project,
            experiment_id=experiment_id,
            enabled=self.cost_tracking.log_to_wandb,
        )
        self.mlflow = MLflowCallback(
            experiment_name=self.cost_tracking.mlflow_experiment_name,
            enabled=self.cost_tracking.log_to_mlflow,
        )

    # ---------------------------------------------------------------------- #
    # Lifecycle Hooks                                                          #
    # ---------------------------------------------------------------------- #

    def on_experiment_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of the experiment; initializes observability backends."""
        self.wandb.init_run(config=config)
        self.mlflow.init_run(
            run_name=self.experiment_id,
            params=config,
        )

    def on_stage_start(self, stage_id: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Called before a training stage begins; logs stage-level params."""
        logger.info(f"ObservabilityCallback: stage '{stage_id}' started.")
        if params:
            stage_params = {f"stage_{stage_id}/{k}": v for k, v in params.items()}
            self.mlflow.log_params(stage_params)
            self.wandb.log({"stage": stage_id, **(params or {})})

    def log_step(
        self, step: int, loss: float, lr: float, tokens_per_sec: float, vram_gb: float
    ) -> None:
        """Called per training step; streams metrics to all active backends."""
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

    def on_stage_end(self, metrics: StageMetrics) -> None:
        """Called after a training stage; logs stage-level summary metrics."""
        self.save_stage_summary(metrics)

        stage_summary = {
            f"stage/{metrics.stage_id}/training_loss": metrics.training_loss,
            f"stage/{metrics.stage_id}/eval_loss": metrics.eval_loss,
            f"stage/{metrics.stage_id}/peak_vram_gb": metrics.peak_vram_gb,
            f"stage/{metrics.stage_id}/tokens_per_second": metrics.tokens_per_second,
            f"stage/{metrics.stage_id}/wall_clock_seconds": metrics.wall_clock_seconds,
            f"stage/{metrics.stage_id}/cost_usd": metrics.estimated_stage_cost_usd,
        }
        self.wandb.log(stage_summary)
        self.mlflow.log_metrics(stage_summary)

    def on_experiment_end(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Called at experiment completion; logs final comparison metrics and table."""
        if summary:
            final_metrics = {
                k: v
                for k, v in summary.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            self.wandb.log_summary(summary)
            self.mlflow.log_metrics(final_metrics)

            # Log comparison table to WandB
            table_rows = [
                {k: str(v) for k, v in summary.items() if not isinstance(v, (list, dict))}
            ]
            self.wandb.log_table("experiment_summary", table_rows)

        self.finish()

    def save_stage_summary(self, metrics: StageMetrics) -> Path:
        """Persist stage metrics as a JSON artifact."""
        summary_path = self.output_dir / f"{metrics.stage_id}_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(metrics.model_dump_json(indent=2))
        logger.info(f"Saved stage metrics summary to {summary_path}")
        return summary_path

    def finish(self) -> None:
        """Finalize all observability backends."""
        self.wandb.finish()
        self.mlflow.finish()


# Backward compatibility alias
TrainingCallback = CompositeObservabilityCallback
