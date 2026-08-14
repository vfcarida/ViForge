"""
ViForge Datasets module: ManifestManager, DatasetAdapter, DatasetRegistry.
"""

from viforge.datasets.manifest import ManifestManager
from viforge.datasets.adapters import DatasetAdapter
from viforge.datasets.registry import DatasetRegistry, dataset_registry

__all__ = [
    "ManifestManager",
    "DatasetAdapter",
    "DatasetRegistry",
    "dataset_registry",
]
