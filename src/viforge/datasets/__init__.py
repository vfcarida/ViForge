"""
ViForge Datasets module: ManifestManager, DatasetAdapter, DatasetRegistry.
"""

from viforge.datasets.manifest import ManifestManager
from viforge.datasets.adapters import DatasetAdapter
from viforge.datasets.registry import DatasetRegistry, dataset_registry
from viforge.datasets.filters import BaseQualityFilter, CodeQualityFilter, TextQualityFilter
from viforge.datasets.preparer import DatasetPreparer, load_domain_preset

__all__ = [
    "ManifestManager",
    "DatasetAdapter",
    "DatasetRegistry",
    "dataset_registry",
    "BaseQualityFilter",
    "CodeQualityFilter",
    "TextQualityFilter",
    "DatasetPreparer",
    "load_domain_preset",
]
