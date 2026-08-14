"""
ViForge Base Training Method Interface and Dynamic Plugin Registry.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type
from viforge.config.schemas import StageMetrics, TrainingStageConfig
from viforge.utils.logging import logger


class BaseTrainingMethod(ABC):
    """Abstract interface for all post-training specialization methods."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass

    @abstractmethod
    def prepare_model(self, model: Any, stage_config: TrainingStageConfig) -> Any:
        pass

    @abstractmethod
    def execute_stage(
        self,
        model: Any,
        tokenizer: Any,
        train_data_path: Path,
        eval_data_path: Optional[Path],
        stage_config: TrainingStageConfig,
        output_dir: Path,
    ) -> StageMetrics:
        pass


class MethodRegistry:
    """Plugin manager for dynamic discovery of training methods."""

    def __init__(self):
        self._methods: Dict[str, Type[BaseTrainingMethod]] = {}

    def register(self, name: str, method_cls: Type[BaseTrainingMethod]) -> None:
        self._methods[name.lower()] = method_cls
        logger.debug(f"Registered method plugin: {name}")

    def get(self, name: str) -> BaseTrainingMethod:
        key = name.lower()
        if key not in self._methods:
            raise KeyError(f"Training method '{name}' not registered. Available: {list(self._methods.keys())}")
        return self._methods[key]()

    def list_all(self) -> list[str]:
        return list(self._methods.keys())


method_registry = MethodRegistry()
