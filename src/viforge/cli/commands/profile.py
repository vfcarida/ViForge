"""
System diagnostics and distributed memory profiling CLI commands.
"""

from pathlib import Path
import typer
from rich.panel import Panel
from rich.table import Table

from viforge.cli.console import console
from viforge.utils.doctor import SystemDoctor


def register(app: typer.Typer) -> None:
    """Register profiling and diagnostics commands."""

    @app.command("doctor")
    def run_doctor():
        """Inspect environment, GPU availability, CUDA, RAM, disk, and dependencies."""
        console.print(Panel.fit("[bold cyan]ViForge System Diagnostics (Doctor)[/bold cyan]"))
        diag = SystemDoctor.diagnose()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status / Value", style="green")

        table.add_row("Python Version", diag["python_version"])
        table.add_row("OS Platform", diag["os_platform"])
        table.add_row(
            "CPU Physical / Logical", f"{diag['cpu_count_physical']} / {diag['cpu_count_logical']}"
        )
        table.add_row(
            "RAM Total / Available", f"{diag['ram_total_gb']} GB / {diag['ram_available_gb']} GB"
        )
        table.add_row("Disk Total / Free", f"{diag['disk_total_gb']} GB / {diag['disk_free_gb']} GB")
        table.add_row("PyTorch Version", str(diag["pytorch_version"]))
        table.add_row(
            "CUDA Available",
            "[bold green]Yes[/bold green]"
            if diag["cuda_available"]
            else "[yellow]No (CPU Mode)[/yellow]",
        )

        if diag["cuda_available"]:
            for dev in diag["cuda_devices"]:
                table.add_row(f"GPU [{dev['index']}]", f"{dev['name']} ({dev['vram_gb']} GB VRAM)")
            table.add_row("BF16 Supported", str(diag["bfloat16_supported"]))

        for pkg in ["transformers", "peft", "trl", "bitsandbytes", "vllm", "boto3", "vipym"]:
            ver = diag.get(f"{pkg}_version", "Not Installed")
            status_style = "green" if ver != "Not Installed" else "yellow"
            table.add_row(f"Package: {pkg}", f"[{status_style}]{ver}[/{status_style}]")

        console.print(table)

    @app.command("distributed-profile")
    def cli_distributed_profile(
        config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
        gpus: int = typer.Option(4, "--gpus", "-g", help="Target number of cluster GPUs"),
        strategy: str = typer.Option(
            "zero3", "--strategy", "-s", help="Distributed strategy (ddp, zero1, zero2, zero3, fsdp2)"
        ),
        cpu_offload: bool = typer.Option(
            False, "--cpu-offload", help="Enable parameter/optimizer CPU offload"
        ),
    ):
        """Profile multi-GPU distributed VRAM partitioning and recommend optimal training topology."""
        from viforge.config.loader import ConfigLoader
        from viforge.config.schemas import HardwareConfig, HyperparametersConfig
        from viforge.training.profiler import ResourceProfiler

        manifest = ConfigLoader.load_manifest(config_path)
        hw = HardwareConfig(
            accelerator=manifest.hardware.accelerator,
            num_gpus=gpus,
            vram_per_gpu_gb=manifest.hardware.vram_per_gpu_gb,
            compute_dtype=manifest.hardware.compute_dtype,
        )
        hp = manifest.pipeline[0].hyperparameters if manifest.pipeline else HyperparametersConfig()

        rec = ResourceProfiler.recommend_distributed_strategy(manifest.model, hp, hw)

        console.print(
            Panel.fit(
                f"[bold cyan]Distributed Memory Profile: {manifest.model.name} ({gpus}x {hw.vram_per_gpu_gb:.0f}GB)[/bold cyan]"
            )
        )

        table = Table(title="Multi-GPU Strategy Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Weights / GPU", style="blue")
        table.add_column("Gradients / GPU", style="white")
        table.add_column("Optimizer / GPU", style="magenta")
        table.add_column("Total VRAM / GPU", style="yellow")
        table.add_column("Utilization", style="green")
        table.add_column("Fits in VRAM", style="bold")

        for s_name, p in rec["all_profiles"].items():
            fits_str = "[green]YES[/green]" if p["fits_in_vram"] else "[red]NO (OOM)[/red]"
            table.add_row(
                s_name.upper(),
                f"{p['weight_memory_per_gpu_gb']:.2f} GB",
                f"{p['gradient_memory_per_gpu_gb']:.2f} GB",
                f"{p['optimizer_memory_per_gpu_gb']:.2f} GB",
                f"{p['total_estimated_vram_per_gpu_gb']:.2f} GB",
                f"{p['utilization_pct']:.1f}%",
                fits_str,
            )
        console.print(table)

        console.print(
            f"\n[bold green]Recommended Strategy:[/bold green] [bold cyan]{rec['recommended_strategy'].upper()}[/bold cyan]"
        )
        console.print(f"[dim]{rec['rationale']}[/dim]\n")

    @app.command("export-distributed-config")
    def cli_export_distributed_config(
        strategy: str = typer.Option(
            "fsdp2", "--strategy", "-s", help="Distributed strategy (fsdp2, ddp, zero3)"
        ),
        framework: str = typer.Option(
            "accelerate", "--framework", "-f", help="Framework ('accelerate' or 'deepspeed')"
        ),
        output_path: Path = typer.Option(
            Path("configs/distributed_config.yaml"), "--output", "-o", help="Target output file"
        ),
        gpus: int = typer.Option(4, "--gpus", "-g", help="Target number of cluster GPUs"),
        vram: float = typer.Option(80.0, "--vram", help="VRAM per GPU in GB"),
    ):
        """Export standard HuggingFace Accelerate or DeepSpeed distributed training configurations."""
        from viforge.config.schemas import HardwareConfig, HyperparametersConfig
        from viforge.training.distributed import DistributedConfigGenerator

        hw = HardwareConfig(num_gpus=gpus, vram_per_gpu_gb=vram)
        hp = HyperparametersConfig()

        if framework.lower() == "deepspeed":
            z_stage = 3 if "3" in strategy else 2 if "2" in strategy else 1
            DistributedConfigGenerator.generate_deepspeed_config(
                hardware=hw,
                hyperparams=hp,
                zero_stage=z_stage,
                output_path=output_path,
            )
        else:
            DistributedConfigGenerator.generate_accelerate_config(
                hardware=hw,
                strategy=strategy,
                output_path=output_path,
            )

        console.print(
            f"[bold green][OK] Distributed configuration exported to:[/bold green] {output_path}"
        )
