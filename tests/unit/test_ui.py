"""
Unit tests for ViForge Interactive Studio (UI) launcher.
"""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from viforge.cli.main import app
from viforge.ui import DASHBOARD_APP_PATH


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_dashboard_file_exists():
    """Verify that the Streamlit app script exists and is readable."""
    assert DASHBOARD_APP_PATH.exists()
    assert DASHBOARD_APP_PATH.is_file()
    content = DASHBOARD_APP_PATH.read_text(encoding="utf-8")
    assert "st.set_page_config" in content
    assert "UnifiedParetoEngine" in content
    assert "ViPymExporter" in content


def test_cli_ui_help(cli_runner):
    """Verify that viforge ui --help displays usage and options."""
    result = cli_runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "Launch the ViForge Interactive Studio" in result.output
    assert "--port" in result.output
    assert "--host" in result.output
