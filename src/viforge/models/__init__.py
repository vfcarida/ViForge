"""
ViForge Models module: Registry, TargetModuleResolver, HuggingFaceModelAdapter.
"""

from viforge.models.target_modules import TargetModuleResolver
from viforge.models.adapters import HuggingFaceModelAdapter
from viforge.models.registry import ModelRegistry, model_registry

__all__ = [
    "TargetModuleResolver",
    "HuggingFaceModelAdapter",
    "ModelRegistry",
    "model_registry",
]
