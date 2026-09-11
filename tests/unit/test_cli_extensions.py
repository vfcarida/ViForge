"""
Unit tests for new ViForge CLI commands (generate-synthetic, distributed-profile, export-distributed-config, slurm-job, ray-job, generate-model-card).
"""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from viforge.cli.main import app

runner = CliRunner()


@pytest.fixture
def sample_exp_yaml():
    return Path("configs/experiments/deepseek_v4_pro_software_engineering.yaml")


@pytest.mark.unit
def test_cli_generate_synthetic(tmp_path: Path):
    out_file = tmp_path / "synthetic_test.jsonl"
    res = runner.invoke(
        app,
        [
            "generate-synthetic",
            "--domain",
            "software_engineering",
            "--samples",
            "2",
            "--strategy",
            "deepen_constraints",
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert "Synthetic Dataset Generation Summary" in res.stdout
    assert out_file.exists()
    assert (tmp_path / "synthetic_test.manifest.json").exists()


@pytest.mark.unit
def test_cli_distributed_profile(sample_exp_yaml):
    res = runner.invoke(
        app,
        [
            "distributed-profile",
            str(sample_exp_yaml),
            "--gpus",
            "4",
            "--strategy",
            "zero3",
        ],
    )
    assert res.exit_code == 0
    assert "Multi-GPU Strategy Comparison" in res.stdout
    assert "ZERO3" in res.stdout
    assert "Recommended Strategy" in res.stdout


@pytest.mark.unit
def test_cli_export_distributed_config(tmp_path: Path):
    out_cfg = tmp_path / "accelerate_cfg.yaml"
    res = runner.invoke(
        app,
        [
            "export-distributed-config",
            "--strategy",
            "fsdp2",
            "--framework",
            "accelerate",
            "--output",
            str(out_cfg),
            "--gpus",
            "4",
        ],
    )
    assert res.exit_code == 0
    assert out_cfg.exists()
    assert "Distributed configuration exported" in res.stdout


@pytest.mark.unit
def test_cli_slurm_job(sample_exp_yaml, tmp_path: Path):
    out_sbatch = tmp_path / "run.sbatch"
    res = runner.invoke(
        app,
        [
            "slurm-job",
            str(sample_exp_yaml),
            "--output",
            str(out_sbatch),
            "--nodes",
            "2",
            "--gpus",
            "4",
            "--partition",
            "gpu",
        ],
    )
    assert res.exit_code == 0
    assert out_sbatch.exists()
    content = out_sbatch.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=2" in content
    assert "#SBATCH --gpus-per-node=4" in content


@pytest.mark.unit
def test_cli_ray_job(sample_exp_yaml, tmp_path: Path):
    out_ray = tmp_path / "ray.yaml"
    res = runner.invoke(
        app,
        [
            "ray-job",
            str(sample_exp_yaml),
            "--output",
            str(out_ray),
            "--workers",
            "2",
            "--gpus",
            "4",
        ],
    )
    assert res.exit_code == 0
    assert out_ray.exists()
    content = out_ray.read_text(encoding="utf-8")
    assert "kind: RayJob" in content


@pytest.mark.unit
def test_cli_generate_model_card(sample_exp_yaml, tmp_path: Path):
    out_card = tmp_path / "README.md"
    res = runner.invoke(
        app,
        [
            "generate-model-card",
            str(sample_exp_yaml),
            "--output",
            str(out_card),
        ],
    )
    assert res.exit_code == 0
    assert out_card.exists()
    content = out_card.read_text(encoding="utf-8")
    assert "Model Highlights" in content
    assert "Post-Training Pipeline Configuration" in content
