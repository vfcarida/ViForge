"""
ViForge Artifact Manager: manages checkpoints, manifests, and adapter registries.
"""

import json
from pathlib import Path
from typing import Any, Dict
from viforge.utils.logging import logger


class ArtifactManager:
    """Manages saving, loading, and tracking experiment checkpoints and metadata."""

    def __init__(self, root_dir: Path):
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
