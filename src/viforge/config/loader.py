"""
ViForge Configuration Loader and Validator.
"""

import os
import re
from pathlib import Path
import yaml
from viforge.config.schemas import ExperimentManifest, ModelConfig
from viforge.utils.logging import logger


class ConfigLoader:
    """Loads and validates YAML/JSON experiment configurations."""

    @classmethod
    def interpolate_env_vars(cls, content: str) -> str:
        """Replace ${ENV_VAR} or ${ENV_VAR:default} patterns."""
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")

        def replacer(match):
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_val)

        return pattern.sub(replacer, content)

    @classmethod
    def load_manifest(cls, path: Path) -> ExperimentManifest:
        """Load, interpolate, and validate an ExperimentManifest."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        interpolated = cls.interpolate_env_vars(raw)
        data = yaml.safe_load(interpolated)
        manifest = ExperimentManifest.model_validate(data)
        logger.debug(f"Loaded and validated manifest: {manifest.experiment_id}")
        return manifest

    @classmethod
    def load_model_config(cls, path: Path) -> ModelConfig:
        """Load and validate a standalone ModelConfig."""
        if not path.exists():
            raise FileNotFoundError(f"Model config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        data = yaml.safe_load(cls.interpolate_env_vars(raw))
        return ModelConfig.model_validate(data)
