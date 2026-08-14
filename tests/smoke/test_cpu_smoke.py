"""
CPU Smoke tests for CLI commands and Doctor diagnostics.
"""

from typer.testing import CliRunner
from viforge.cli.main import app
from viforge.utils.doctor import SystemDoctor

runner = CliRunner()


def test_doctor_diagnose():
    diag = SystemDoctor.diagnose()
    assert "python_version" in diag
    assert "cpu_count_physical" in diag
    assert diag["ram_total_gb"] > 0


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ViForge System Diagnostics" in result.stdout


def test_cli_list_commands():
    res_m = runner.invoke(app, ["list-methods"])
    assert res_m.exit_code == 0
    assert "qlora" in res_m.stdout

    res_e = runner.invoke(app, ["list-evaluators"])
    assert res_e.exit_code == 0
    assert "humaneval_plus" in res_e.stdout
