"""
ViForge Training Telemetry, Checkpointing, and Callbacks.
"""

import json
from pathlib import Path
from viforge.config.schemas import StageMetrics
from viforge.utils.logging import logger


class TrainingCallback:
    """Manages metrics streaming, checkpoint manifests, and resume tracking."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_path = output_dir / "training_metrics.jsonl"

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

    def save_stage_summary(self, metrics: StageMetrics) -> Path:
        summary_path = self.output_dir / f"{metrics.stage_id}_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(metrics.model_dump_json(indent=2))
        logger.info(f"Saved stage metrics summary to {summary_path}")
        return summary_path
