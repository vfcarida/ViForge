"""
ViForge Dataset Registry: manages registered datasets and governance manifests.
"""

from typing import Dict, List, Optional
from viforge.config.schemas import DatasetGovernanceManifest
from viforge.utils.logging import logger


class DatasetRegistry:
    """Central registry of registered datasets and immutable governance metadata."""

    def __init__(self):
        self._manifests: Dict[str, DatasetGovernanceManifest] = {}

    def register(self, manifest: DatasetGovernanceManifest) -> None:
        self._manifests[manifest.dataset_id.lower()] = manifest
        logger.debug(f"Registered dataset manifest: {manifest.dataset_id}")

    def get_manifest(self, dataset_id: str) -> Optional[DatasetGovernanceManifest]:
        return self._manifests.get(dataset_id.lower())

    def list_all(self) -> List[DatasetGovernanceManifest]:
        return list(self._manifests.values())


dataset_registry = DatasetRegistry()
