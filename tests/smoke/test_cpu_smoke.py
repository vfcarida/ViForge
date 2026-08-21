"""
CPU Smoke tests for CLI commands and Doctor diagnostics.
"""

import pytest
from typer.testing import CliRunner

from viforge.cli.main import app
from viforge.utils.doctor import SystemDoctor

runner = CliRunner()


@pytest.mark.smoke
def test_doctor_diagnose():
    diag = SystemDoctor.diagnose()
    assert "python_version" in diag
    assert "cpu_count_physical" in diag
    assert diag["ram_total_gb"] > 0


@pytest.mark.smoke
def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ViForge System Diagnostics" in result.stdout


@pytest.mark.smoke
def test_cli_list_commands():
    res_m = runner.invoke(app, ["list-methods"])
    assert res_m.exit_code == 0
    assert "qlora" in res_m.stdout

    res_e = runner.invoke(app, ["list-evaluators"])
    assert res_e.exit_code == 0
    assert "humaneval_plus" in res_e.stdout


@pytest.mark.smoke
def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "ViForge" in res.stdout
    assert "doctor" in res.stdout
    assert "run" in res.stdout


@pytest.mark.smoke
def test_cli_prepare_data_help():
    res = runner.invoke(app, ["prepare-data", "--help"])
    assert res.exit_code == 0
    assert "preset" in res.stdout


@pytest.mark.smoke
def test_cli_generate_sbom(tmp_path):
    out_sbom = tmp_path / "sbom.json"
    res = runner.invoke(app, ["generate-sbom", "--output", str(out_sbom)])
    assert res.exit_code == 0
    assert out_sbom.exists()
    assert "CycloneDX SBOM generated" in res.stdout


@pytest.mark.smoke
def test_cli_merge_adapters_help():
    res = runner.invoke(app, ["merge-adapters", "--help"])
    assert res.exit_code == 0
    assert "base_model_id" in res.stdout
