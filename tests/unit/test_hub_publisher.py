"""
Unit tests for Hugging Face Hub direct publishing integration and CLI push-to-hub command.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from viforge.artifacts.hub import HuggingFaceHubPublisher, validate_repo_id
from viforge.cli.main import app

runner = CliRunner()


def test_validate_repo_id():
    """Test repo ID validation for standard Hugging Face Hub format."""
    assert validate_repo_id("viforge-org/qwen2.5-coder") is True
    assert validate_repo_id("user123/my-specialist.v1") is True
    assert validate_repo_id("standalone-model-name") is True
    assert validate_repo_id("org_name/model_name") is True

    assert validate_repo_id("") is False
    assert validate_repo_id("invalid/repo/with/too/many/slashes") is False
    assert validate_repo_id("invalid name with spaces/repo") is False
    assert validate_repo_id("repo@name/invalid") is False


def test_upload_model_validates_directory(tmp_path: Path):
    """Test that upload_model raises FileNotFoundError if model_dir does not exist."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        HuggingFaceHubPublisher.upload_model(
            model_dir=non_existent,
            repo_id="test-org/model",
        )


def test_upload_model_validates_repo_id(tmp_path: Path):
    """Test that upload_model raises ValueError if repo_id is invalid."""
    model_dir = tmp_path / "valid_dir"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="Invalid Hugging Face repo_id format"):
        HuggingFaceHubPublisher.upload_model(
            model_dir=model_dir,
            repo_id="invalid/too/many/slashes",
        )


def test_upload_model_successful_mocked(tmp_path: Path):
    """Test successful model upload with mocked HfApi."""
    model_dir = tmp_path / "specialist_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    mock_api = MagicMock()

    with patch("viforge.artifacts.hub.HfApi", return_value=mock_api):
        url = HuggingFaceHubPublisher.upload_model(
            model_dir=model_dir,
            repo_id="viforge-org/qwen-coder-specialist",
            token="hf_mock_token_12345",
            private=True,
            commit_message="Test commit",
        )

        assert url == "https://huggingface.co/viforge-org/qwen-coder-specialist"
        mock_api.create_repo.assert_called_once_with(
            repo_id="viforge-org/qwen-coder-specialist",
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        mock_api.upload_folder.assert_called_once_with(
            folder_path=str(model_dir),
            repo_id="viforge-org/qwen-coder-specialist",
            repo_type="model",
            commit_message="Test commit",
        )


def test_upload_model_with_model_card(tmp_path: Path):
    """Test upload_model copies model card to README.md before uploading."""
    model_dir = tmp_path / "model_weights"
    model_dir.mkdir()
    model_card = tmp_path / "external_model_card.md"
    model_card.write_text("# Model Card for Specialist", encoding="utf-8")

    mock_api = MagicMock()

    with patch("viforge.artifacts.hub.HfApi", return_value=mock_api):
        url = HuggingFaceHubPublisher.upload_model(
            model_dir=model_dir,
            repo_id="viforge-org/model-card-test",
            model_card_path=model_card,
        )

        assert url == "https://huggingface.co/viforge-org/model-card-test"
        readme_dest = model_dir / "README.md"
        assert readme_dest.exists()
        assert readme_dest.read_text(encoding="utf-8") == "# Model Card for Specialist"


def test_cli_push_to_hub_successful(tmp_path: Path):
    """Test CLI push-to-hub command executes successfully when mocked."""
    model_dir = tmp_path / "cli_model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with patch(
        "viforge.artifacts.hub.HuggingFaceHubPublisher.upload_model",
        return_value="https://huggingface.co/my-org/my-specialist",
    ) as mock_upload:
        result = runner.invoke(
            app,
            [
                "push-to-hub",
                str(model_dir),
                "my-org/my-specialist",
                "--token",
                "hf_cli_token_999",
                "--private",
                "--commit-message",
                "Release v1.0",
            ],
        )

        assert result.exit_code == 0
        assert "Model published successfully to Hugging Face Hub" in result.output
        assert "https://huggingface.co/my-org/my-specialist" in result.output
        mock_upload.assert_called_once()
