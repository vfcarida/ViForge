"""
Artifact export, quantization, model cards, SBOM, and ViPym compression CLI commands.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.panel import Panel
from rich.table import Table

from viforge.artifacts.gguf import GGUFExporter
from viforge.artifacts.quantization import AWQQuantizer
from viforge.cli.console import console


def register(app: typer.Typer) -> None:
    """Register model export, quantization, SBOM, and ViPym commands."""

    @app.command("export-gguf")
    def export_gguf_cli(
        model_dir: Path = typer.Argument(
            ..., help="Directory of merged HuggingFace / Safetensors model"
        ),
        output_dir: Path = typer.Option(
            Path("exports/gguf"),
            "--output-dir",
            "-o",
            help="Output directory for GGUF and Modelfile",
        ),
        quant_type: str = typer.Option(
            "Q4_K_M", "--quant-type", "-q", help="Quantization type (Q4_K_M, Q5_K_M, Q8_0, F16)"
        ),
        system_prompt: str = typer.Option(
            "You are ViForge Specialist, an expert software engineering and reasoning AI.",
            "--system-prompt",
            "-s",
            help="System prompt for Ollama Modelfile",
        ),
    ):
        """Export model to GGUF format and generate ready-to-run Ollama Modelfile."""
        console.print(
            f"[bold cyan]Exporting model to GGUF ({quant_type}):[/bold cyan] {model_dir}"
        )
        res = GGUFExporter.export(
            merged_model_dir=model_dir,
            output_gguf_dir=output_dir,
            quant_type=quant_type,
            generate_ollama=True,
            system_prompt=system_prompt,
        )
        console.print(f"[bold green]GGUF Exported:[/bold green] {res['gguf_path']}")
        console.print(f"[bold green]Ollama Modelfile:[/bold green] {res['modelfile_path']}")
        console.print(
            f"[dim]Run locally with: ollama create {model_dir.name} -f {res['modelfile_path']}[/dim]"
        )

    @app.command("quantize")
    def quantize_cli(
        model_dir: Path = typer.Argument(..., help="Directory of merged model"),
        output_dir: Path = typer.Option(
            Path("exports/awq"), "--output-dir", "-o", help="Output directory for AWQ model"
        ),
        bits: int = typer.Option(4, "--bits", "-b", help="Quantization bitwidth (4 or 8)"),
        group_size: int = typer.Option(128, "--group-size", "-g", help="AWQ group size"),
    ):
        """Apply AWQ post-training quantization for low-VRAM deployment."""
        console.print(f"[bold cyan]Applying AWQ {bits}-bit quantization to:[/bold cyan] {model_dir}")
        res = AWQQuantizer.quantize(
            model_path=model_dir, output_dir=output_dir, bits=bits, group_size=group_size
        )
        console.print(f"[bold green]AWQ Model saved to:[/bold green] {res['output_dir']}")

    @app.command("generate-sbom")
    def cli_generate_sbom(
        output_path: Path = typer.Option(
            Path("dist/sbom.cyclonedx.json"), "--output", "-o", help="Output SBOM path"
        ),
        lockfile: Optional[Path] = typer.Option(
            None, "--lockfile", "-l", help="Specific lockfile to parse"
        ),
    ):
        """Generate CycloneDX v1.5 JSON Software Bill of Materials (SBOM) for supply-chain attestation."""
        from viforge.security.sbom import generate_sbom

        root_dir = Path(__file__).parent.parent.parent.parent
        target_lock = lockfile
        if not target_lock or not target_lock.exists():
            target_lock = root_dir / "requirements" / "all.lock"
            if not target_lock.exists():
                target_lock = root_dir / "requirements" / "base.lock"

        console.print(Panel.fit("[bold cyan]ViForge Supply-Chain SBOM Generator[/bold cyan]"))
        console.print(f"Reading lockfile: [green]{target_lock}[/green]")
        out = generate_sbom(target_lock, output_path)
        console.print(f"[bold green][OK] CycloneDX SBOM generated successfully:[/bold green] {out}")

    @app.command("export-vipym")
    def cli_export_vipym(
        model_or_config: Path = typer.Argument(
            ...,
            help="Path to merged model directory or ViForge experiment YAML manifest",
        ),
        recipe: str = typer.Option(
            "smoothquant_w8a8",
            "--recipe",
            "-r",
            help="Compression recipe (smoothquant_w8a8, autoround_w4a16, awq_w4a16, gptq_w4a16, distill_logit, fp8_kv, spinquant)",
        ),
        output_path: Path = typer.Option(
            Path("configs/vipym_export.yaml"),
            "--output",
            "-o",
            help="Destination path for ViPym experiment YAML",
        ),
        merged_model_dir: Optional[Path] = typer.Option(
            None,
            "--merged-model-dir",
            help="Path to merged model directory (if passing ViForge YAML manifest)",
        ),
        student_model: Optional[str] = typer.Option(
            None,
            "--student-model",
            "-s",
            help="Student model ID or path if using distillation recipe (e.g. Qwen/Qwen2.5-Coder-1.5B)",
        ),
        calibration_dataset: str = typer.Option(
            "wikitext",
            "--calibration-dataset",
            help="Calibration dataset name for quantization",
        ),
        calibration_samples: int = typer.Option(
            512,
            "--calibration-samples",
            help="Number of calibration samples",
        ),
    ):
        """Export ViForge specialist model into a validated ViPym compression manifest."""
        from viforge.integrations.vipym import ViPymExporter

        console.print(Panel.fit("[bold cyan]ViForge -> ViPym Experiment Exporter[/bold cyan]"))
        console.print(f"Target: [green]{model_or_config}[/green]")
        console.print(f"Recipe: [magenta]{recipe}[/magenta]")

        # Check if target is a file/manifest or directory
        if model_or_config.is_file() or str(model_or_config).endswith((".yaml", ".yml")):
            exported = ViPymExporter.export_from_manifest(
                manifest_or_path=model_or_config,
                recipe=recipe,
                output_yaml_path=output_path,
                merged_model_dir=merged_model_dir,
                student_model_id=student_model,
                calibration_dataset=calibration_dataset,
                calibration_samples=calibration_samples,
            )
        else:
            exported = ViPymExporter.export_from_model_dir(
                model_dir=model_or_config,
                recipe=recipe,
                output_yaml_path=output_path,
                student_model_id=student_model,
                calibration_dataset=calibration_dataset,
                calibration_samples=calibration_samples,
            )

        console.print(
            f"[bold green][OK] ViPym experiment manifest exported successfully:[/bold green] {exported}"
        )
        console.print(f"[dim]Run directly with: viforge compress-vipym {exported} --mock[/dim]")

    @app.command("compress-vipym")
    def cli_compress_vipym(
        model_or_config: Path = typer.Argument(
            ...,
            help="Path to ViPym YAML manifest, ViForge experiment YAML, or merged model directory",
        ),
        recipe: str = typer.Option(
            "smoothquant_w8a8",
            "--recipe",
            "-r",
            help="Compression recipe to use if passing ViForge manifest or model directory",
        ),
        work_dir: Path = typer.Option(
            Path("runs/vipym"),
            "--work-dir",
            "-w",
            help="Working directory for compressed artifacts and summaries",
        ),
        mock: bool = typer.Option(
            True,
            "--mock/--live",
            help="Simulate execution with deterministic mock stats (True) or execute live ViPym engine (False)",
        ),
        student_model: Optional[str] = typer.Option(
            None,
            "--student-model",
            "-s",
            help="Student model ID if using distillation recipe",
        ),
    ):
        """Execute ViPym compression pipeline on ViForge specialist model."""
        import yaml
        from viforge.integrations.vipym import ViPymExporter, ViPymRunner, is_vipym_available

        console.print(Panel.fit("[bold cyan]ViForge x ViPym Compression Orchestrator[/bold cyan]"))

        work_dir.mkdir(parents=True, exist_ok=True)

        # Determine if input is already a ViPym config or needs export
        config_to_run: Path
        if model_or_config.is_file() and str(model_or_config).endswith((".yaml", ".yml")):
            with open(model_or_config, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "compression_pipeline" in data:
                config_to_run = model_or_config
            else:
                # It's a ViForge manifest! Export it first
                export_dst = work_dir / f"vipym_{recipe}_{model_or_config.stem}.yaml"
                config_to_run = ViPymExporter.export_from_manifest(
                    manifest_or_path=model_or_config,
                    recipe=recipe,
                    output_yaml_path=export_dst,
                    student_model_id=student_model,
                )
        else:
            # Directory model
            export_dst = work_dir / f"vipym_{recipe}_{model_or_config.name}.yaml"
            config_to_run = ViPymExporter.export_from_model_dir(
                model_dir=model_or_config,
                recipe=recipe,
                output_yaml_path=export_dst,
                student_model_id=student_model,
            )

        console.print(f"ViPym Config: [green]{config_to_run}[/green]")
        console.print(
            f"Mode: [bold magenta]{'MOCK SIMULATION' if mock else 'LIVE VIPYM EXECUTION'}[/bold magenta]"
        )
        console.print(
            f"ViPym Installed: {'[green]Yes[/green]' if is_vipym_available() else '[yellow]No[/yellow]'}"
        )

        result = ViPymRunner.run_compression(
            vipym_config_path=config_to_run,
            work_dir=work_dir,
            mock=mock,
        )

        if result.get("status") == "Completed":
            console.print(
                "\n[bold green]=== ViPym Compression Completed Successfully ===[/bold green]"
            )
            table = Table(title="ViPym Compression Results")
            table.add_column("Metric / Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Experiment ID", str(result.get("experiment_id")))
            table.add_row("Method", str(result.get("method")))
            table.add_row("Scheme", str(result.get("scheme")))
            if result.get("is_mock"):
                table.add_row("Size Reduction", f"{result.get('size_reduction_pct', 0):.1f}%")
                table.add_row("Latency Speedup", f"{result.get('latency_speedup', 1.0):.2f}x")
                table.add_row("VRAM Saved", f"{result.get('vram_saved_gb', 0):.1f} GB")
            table.add_row("Output Directory", str(result.get("output_dir")))
            table.add_row("Execution Time", f"{result.get('execution_time_seconds', 0)}s")
            console.print(table)
        else:
            console.print(f"[bold red]ViPym Compression Failed:[/bold red] {result}")

    @app.command("generate-model-card")
    def cli_generate_model_card(
        config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
        output_path: Path = typer.Option(
            Path("README.md"), "--output", "-o", help="Path for output model card"
        ),
    ):
        """Generate Hugging Face Model Card (README.md) with metadata, tables, and Pareto recommendations."""
        from viforge.config.loader import ConfigLoader
        from viforge.reporting.model_card import ModelCardGenerator

        manifest = ConfigLoader.load_manifest(config_path)
        ModelCardGenerator.generate_model_card(
            manifest=manifest,
            summary=None,
            output_path=output_path,
        )
        console.print(f"[bold green][OK] Hugging Face Model Card generated:[/bold green] {output_path}")

    @app.command("push-to-hub")
    def cli_push_to_hub(
        model_dir: Path = typer.Argument(
            ..., help="Path to model or LoRA adapter directory to publish"
        ),
        repo_id: str = typer.Argument(
            ..., help="Hugging Face Hub repository ID (e.g. 'username/model-specialist')"
        ),
        token: Optional[str] = typer.Option(
            None, "--token", "-t", help="Hugging Face API token (defaults to HF_TOKEN env var)"
        ),
        private: bool = typer.Option(
            False, "--private", help="Create as a private repository"
        ),
        commit_message: str = typer.Option(
            "Upload model via ViForge", "--commit-message", "-m", help="Git commit message"
        ),
        model_card: Optional[Path] = typer.Option(
            None, "--model-card", "-c", help="Optional README.md model card to include"
        ),
    ):
        """Upload fine-tuned model weights, LoRA adapters, and Model Cards directly to Hugging Face Hub."""
        from viforge.artifacts.hub import HuggingFaceHubPublisher

        console.print(Panel.fit("[bold cyan]ViForge Hugging Face Hub Publisher[/bold cyan]"))
        console.print(f"Model Directory: [green]{model_dir}[/green]")
        console.print(f"Target Repo ID:  [bold magenta]{repo_id}[/bold magenta]")
        console.print(f"Visibility:      {'[yellow]Private[/yellow]' if private else '[blue]Public[/blue]'}")

        url = HuggingFaceHubPublisher.upload_model(
            model_dir=model_dir,
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
            model_card_path=model_card,
        )
        console.print(
            f"\n[bold green][OK] Model published successfully to Hugging Face Hub:[/bold green] {url}"
        )
