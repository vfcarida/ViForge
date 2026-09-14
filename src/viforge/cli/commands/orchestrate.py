"""
Cluster workload orchestration and batch job generation (Slurm HPC & KubeRay) CLI commands.
"""

from pathlib import Path
from typing import Optional
import typer

from viforge.cli.console import console


def register(app: typer.Typer) -> None:
    """Register cluster orchestration commands."""

    @app.command("slurm-job")
    def cli_slurm_job(
        config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
        output_path: Path = typer.Option(
            Path("scripts/train_cluster.sbatch"),
            "--output",
            "-o",
            help="Path for generated .sbatch script",
        ),
        nodes: int = typer.Option(1, "--nodes", "-n", help="Number of compute nodes"),
        gpus_per_node: int = typer.Option(4, "--gpus", "-g", help="GPUs per compute node"),
        partition: str = typer.Option("gpu", "--partition", "-p", help="Slurm partition name"),
        time_limit: str = typer.Option("24:00:00", "--time", "-t", help="Slurm execution time limit"),
        container: Optional[str] = typer.Option(
            None, "--container", help="Path to Apptainer/Singularity container image"
        ),
    ):
        """Generate production-ready Slurm HPC batch submission script."""
        from viforge.config.loader import ConfigLoader
        from viforge.orchestration.slurm import SlurmJobGenerator

        manifest = ConfigLoader.load_manifest(config_path)
        env_type = "container" if container else "venv"

        saved = SlurmJobGenerator.export_sbatch(
            manifest=manifest,
            output_path=output_path,
            config_path=config_path,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            partition=partition,
            time_limit=time_limit,
            env_type=env_type,
            container_image=container,
        )
        console.print(f"[bold green][OK] Slurm sbatch script generated:[/bold green] {saved}")
        console.print(f"[dim]Submit on cluster via: sbatch {saved}[/dim]")

    @app.command("ray-job")
    def cli_ray_job(
        config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
        output_path: Path = typer.Option(
            Path("configs/rayjob.yaml"), "--output", "-o", help="Path for KubeRay YAML manifest"
        ),
        workers: int = typer.Option(1, "--workers", "-w", help="Number of Ray worker replicas"),
        gpus_per_worker: int = typer.Option(4, "--gpus", "-g", help="GPUs per Ray worker"),
    ):
        """Generate Kubernetes KubeRay RayJob manifest."""
        from viforge.config.loader import ConfigLoader
        from viforge.orchestration.ray import RayJobGenerator

        manifest = ConfigLoader.load_manifest(config_path)
        saved = RayJobGenerator.export_rayjob_yaml(
            manifest=manifest,
            output_path=output_path,
            config_path=str(config_path),
            num_workers=workers,
            gpus_per_worker=gpus_per_worker,
        )
        console.print(f"[bold green][OK] KubeRay RayJob manifest generated:[/bold green] {saved}")
        console.print(f"[dim]Apply on Kubernetes via: kubectl apply -f {saved}[/dim]")
