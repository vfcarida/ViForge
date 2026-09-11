from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from viforge.config.schemas import ExperimentManifest, ExperimentSummaryReport
from viforge.reporting.model_card import ModelCardGenerator
from viforge.utils.logging import logger


class ArtifactManager:
    """Manages saving, loading, tracking experiment checkpoints, artifact lineage, and metadata."""

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest(self, experiment_id: str, manifest_data: Dict[str, Any]) -> Path:
        manifest_path = self.root_dir / experiment_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, default=str)
        logger.info(f"Saved artifact manifest to {manifest_path}")
        return manifest_path

    def get_stage_checkpoint_dir(self, experiment_id: str, stage_id: str) -> Path:
        p = self.root_dir / experiment_id / "checkpoints" / stage_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def record_lineage(
        self,
        experiment_id: str,
        stage_id: str,
        artifact_type: str,
        artifact_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record artifact creation event in persistent experiment lineage log."""
        lineage_file = self.root_dir / experiment_id / "lineage.jsonl"
        lineage_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "stage_id": stage_id,
            "artifact_type": artifact_type,
            "artifact_path": str(artifact_path),
            "metadata": metadata or {},
        }
        with open(lineage_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.debug(f"Recorded lineage event for {artifact_type} ({stage_id})")
        return entry

    def get_lineage(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Retrieve full lineage history for an experiment run."""
        lineage_file = self.root_dir / experiment_id / "lineage.jsonl"
        if not lineage_file.exists():
            return []
        entries = []
        with open(lineage_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def generate_model_card(
        self,
        manifest: ExperimentManifest,
        summary: Optional[ExperimentSummaryReport] = None,
        output_filename: str = "README.md",
    ) -> Path:
        """Generates a Hugging Face Model Card inside the experiment root directory."""
        dest = self.root_dir / manifest.experiment_id / output_filename
        ModelCardGenerator.generate_model_card(manifest, summary=summary, output_path=dest)
        self.record_lineage(
            experiment_id=manifest.experiment_id,
            stage_id="reporting",
            artifact_type="model_card",
            artifact_path=dest,
        )
        return dest

