"""
Shared pytest fixtures and configuration for ViForge test suite.
"""

from pathlib import Path

import pytest

from viforge.config.loader import ConfigLoader
from viforge.config.schemas import ExperimentManifest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_config_path(project_root) -> Path:
    """Return the path to a sample experiment YAML configuration."""
    return project_root / "configs" / "experiments" / "deepseek_v4_pro_software_engineering.yaml"


@pytest.fixture(scope="session")
def sample_manifest(sample_config_path) -> ExperimentManifest:
    """Return a parsed sample ExperimentManifest."""
    loader = ConfigLoader()
    return loader.load(sample_config_path)


@pytest.fixture
def mock_backend():
    """Return a fresh instance of MockInferenceBackend."""
    from viforge.inference.backends import MockInferenceBackend

    return MockInferenceBackend()


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Return a newly created output directory within pytest's temporary path."""
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output
