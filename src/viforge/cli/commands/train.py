"""
Training execution, end-to-end campaign runner, methods registry, and adapter merger CLI commands.
"""

from pathlib import Path
import typer
from rich.panel import Panel

from viforge.cli.console import console
from viforge.experiments.runner import ExperimentRunner
from viforge.methods.base import method_registry


def register(app: typer.Typer) -> None:
    """Register training and campaign runner commands."""

    @app.command("list-methods")
    def list_methods():
        """List available post-training specialization methods."""
        methods = method_registry.list_all()
        console.print(f"[bold cyan]Registered Training Methods:[/bold cyan] {', '.join(methods)}")

    @app.command("train")
    def run_train(
        config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
        work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
    ):
        """Execute training pipeline stages with pre-flight VRAM profiling."""
        runner = ExperimentRunner.from_yaml(config_path, work_dir)
        runner.profile_and_validate()
        console.print(
            "[bold green]Pre-flight checks passed.[/bold green] Executing training stages..."
        )
        results = runner.run_training_stages()
        for stage_id, res in results.items():
            console.print(
                f"  [green][OK][/green] Stage [bold]{stage_id}[/bold] completed. Cost: ${res.get('estimated_cost_usd', 0):.2f}"
            )

    @app.command("run")
    def run_all(
        config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
        work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
        mock: bool = typer.Option(
            True, "--mock/--live", help="Use deterministic mock backend for fast CI/test execution"
        ),
        strict: bool = typer.Option(
            False,
            "--strict",
            help="Enforce strict production mode (fail-fast on any mock or fallback)",
        ),
    ):
        """Execute complete end-to-end ViForge experimentation campaign."""
        runner = ExperimentRunner.from_yaml(config_path, work_dir)
        backend_type = "mock" if mock else "huggingface"
        summary = runner.execute(backend_type=backend_type, strict=strict)

        console.print("\n[bold green]=== Campaign Completed Successfully ===[/bold green]")
        console.print(f"[bold]Model:[/bold] {summary.model_name}")
        console.print(f"[bold]Domain Gain:[/bold] [green]{summary.domain_gain_pct:+.2f}%[/green]")
        console.print(
            f"[bold]Retention Delta:[/bold] [yellow]{summary.retention_delta_pct:+.2f}%[/yellow]"
        )
        console.print(f"[bold]Total Training Cost:[/bold] ${summary.total_training_cost_usd:.2f}")
        console.print(f"\n[bold underline]Verdict:[/bold underline]\n{summary.verdict}\n")

    @app.command("merge-adapters")
    def cli_merge_adapters(
        base_model_id: str = typer.Argument(..., help="Base HuggingFace model ID or path"),
        adapter_path: Path = typer.Argument(..., help="Path to LoRA adapter directory"),
        output_dir: Path = typer.Option(
            Path("merged_model"), "--output", "-o", help="Target directory for merged weights"
        ),
        device: str = typer.Option("cpu", "--device", "-d", help="Device for merging (cpu/cuda)"),
    ):
        """Merge trained LoRA adapters into base model weights for zero-overhead inference."""
        from viforge.artifacts.merger import AdapterMerger

        console.print(Panel.fit("[bold cyan]ViForge Model & Adapter Merger[/bold cyan]"))
        out = AdapterMerger.merge_and_export(
            base_model_id=base_model_id,
            adapter_path=adapter_path,
            output_dir=output_dir,
            device=device,
        )
        console.print(f"[bold green][OK] Merged model saved to:[/bold green] {out}")
