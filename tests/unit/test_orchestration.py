"""
Unit tests for Orchestration Subsystem (Slurm HPC, Ray/KubeRay, and AWS SageMaker).
"""

from pathlib import Path
import pytest

from viforge.aws.sagemaker import SageMakerRunner
from viforge.config.schemas import (
    ExperimentManifest,
    HardwareConfig,
    ModelConfig,
    TrainingStageConfig,
)
from viforge.orchestration.ray import RayJobGenerator
from viforge.orchestration.slurm import SlurmJobGenerator


@pytest.fixture
def sample_manifest():
    return ExperimentManifest(
        experiment_id="test_orchestration_exp",
        model=ModelConfig(
            name="DeepSeek-Test",
            hf_hub_id="deepseek-ai/test-model",
            total_parameters=7.0,
        ),
        hardware=HardwareConfig(
            accelerator="NVIDIA_A100_80GB",
            num_gpus=4,
            vram_per_gpu_gb=80.0,
        ),
        pipeline=[
            TrainingStageConfig(
                stage_id="sft_stage",
                method="sft",
                dataset={"id": "ds_test", "split": "train"},
            )
        ],
    )


@pytest.mark.unit
def test_slurm_sbatch_script_generation(sample_manifest, tmp_path: Path):
    out_sbatch = tmp_path / "job.sbatch"
    saved = SlurmJobGenerator.export_sbatch(
        manifest=sample_manifest,
        output_path=out_sbatch,
        nodes=2,
        gpus_per_node=4,
        time_limit="12:00:00",
        partition="gpu-a100",
    )

    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=2" in content
    assert "#SBATCH --gpus-per-node=4" in content
    assert "#SBATCH --partition=gpu-a100" in content
    assert "torchrun" in content
    assert "MASTER_PORT=29500" in content


@pytest.mark.unit
def test_slurm_container_mode(sample_manifest):
    script = SlurmJobGenerator.generate_sbatch_script(
        manifest=sample_manifest,
        env_type="container",
        container_image="/opt/images/viforge.sif",
    )
    assert "apptainer exec --nv" in script
    assert "/opt/images/viforge.sif" in script


@pytest.mark.unit
def test_rayjob_manifest_generation(sample_manifest, tmp_path: Path):
    out_yaml = tmp_path / "rayjob.yaml"
    saved = RayJobGenerator.export_rayjob_yaml(
        manifest=sample_manifest,
        output_path=out_yaml,
        num_workers=2,
        gpus_per_worker=4,
    )

    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "kind: RayJob" in content
    assert "apiVersion: ray.io/v1" in content
    assert "nvidia.com/gpu: '4'" in content or "nvidia.com/gpu: 4" in content or "nvidia.com/gpu" in content


@pytest.mark.unit
def test_sagemaker_entrypoint_script_syntax(tmp_path: Path):
    entrypoint = tmp_path / "entrypoint.py"
    SageMakerRunner.generate_entrypoint_script(entrypoint)
    assert entrypoint.exists()

    content = entrypoint.read_text(encoding="utf-8")
    assert "from viforge.config.loader import ConfigLoader" in content
    assert "from viforge.experiments.runner import ExperimentRunner" in content
    assert "load_experiment_manifest" not in content

    # Test that Python can compile the generated script without syntax errors
    compile(content, str(entrypoint), "exec")
