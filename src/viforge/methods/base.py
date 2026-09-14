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
        self.discover_entrypoint_plugins()

    def register(self, name: str, method_cls: Type[BaseTrainingMethod]) -> None:
        self._methods[name.lower()] = method_cls
        logger.debug(f"Registered method plugin: {name}")

    def discover_entrypoint_plugins(self) -> int:
        """
        Discovers and registers third-party method plugins via standard Python entry points:
        group = 'viforge.methods'
        """
        import importlib.metadata

        count = 0
        try:
            entry_points = importlib.metadata.entry_points()
            eps = (
                entry_points.select(group="viforge.methods")
                if hasattr(entry_points, "select")
                else entry_points.get("viforge.methods", [])  # type: ignore[attr-defined]
            )
            for ep in eps:
                try:
                    plugin_cls = ep.load()
                    if isinstance(plugin_cls, type) and issubclass(plugin_cls, BaseTrainingMethod):
                        self.register(ep.name, plugin_cls)
                        count += 1
                        logger.info(f"Loaded third-party method plugin '{ep.name}' via entry point.")
                    else:
                        logger.warning(
                            f"Entry point '{ep.name}' is not a subclass of BaseTrainingMethod; skipped."
                        )
                except Exception as e:
                    logger.warning(f"Failed to load entry point method '{ep.name}': {e}")
        except Exception as e:
            logger.debug(f"Plugin discovery encountered an issue: {e}")
        return count

    def get(self, name: str) -> BaseTrainingMethod:
        key = name.lower()
        if key not in self._methods:
            raise KeyError(
                f"Training method '{name}' not registered. Available: {list(self._methods.keys())}"
            )
        return self._methods[key]()

    def list_all(self) -> list[str]:
        return list(self._methods.keys())


method_registry = MethodRegistry()
