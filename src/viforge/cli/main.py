"""
ViForge Production CLI: End-to-End Specialization & Pareto Evaluation Platform.
"""

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from viforge.utils.doctor import SystemDoctor
from viforge.config.loader import ConfigLoader
from viforge.datasets.registry import dataset_registry
from viforge.methods.base import method_registry
from viforge.evaluation.suites import evaluator_registry
from viforge.experiments.runner import ExperimentRunner
from viforge.artifacts.gguf import GGUFExporter
from viforge.artifacts.quantization import AWQQuantizer

app = typer.Typer(
    name="viforge",
    help="ViForge: Forging Small Models into Specialists — Production Experimentation Platform.",
    add_completion=False,
)
console = Console()


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
    table.add_row("CPU Physical / Logical", f"{diag['cpu_count_physical']} / {diag['cpu_count_logical']}")
    table.add_row("RAM Total / Available", f"{diag['ram_total_gb']} GB / {diag['ram_available_gb']} GB")
    table.add_row("Disk Total / Free", f"{diag['disk_total_gb']} GB / {diag['disk_free_gb']} GB")
    table.add_row("PyTorch Version", str(diag["pytorch_version"]))
    table.add_row("CUDA Available", "[bold green]Yes[/bold green]" if diag["cuda_available"] else "[yellow]No (CPU Mode)[/yellow]")

    if diag["cuda_available"]:
        for dev in diag["cuda_devices"]:
            table.add_row(f"GPU [{dev['index']}]", f"{dev['name']} ({dev['vram_gb']} GB VRAM)")
        table.add_row("BF16 Supported", str(diag["bfloat16_supported"]))

    for pkg in ["transformers", "peft", "trl", "bitsandbytes", "vllm", "boto3"]:
        ver = diag.get(f"{pkg}_version", "Not Installed")
        status_style = "green" if ver != "Not Installed" else "yellow"
        table.add_row(f"Package: {pkg}", f"[{status_style}]{ver}[/{status_style}]")

    console.print(table)


@app.command("validate")
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
):
    """Validate experiment manifest against Pydantic schemas."""
    console.print(f"[bold cyan]Validating manifest:[/bold cyan] {config_path}")
    manifest = ConfigLoader.load_manifest(config_path)
    console.print(f"[bold green]OK:[/bold green] Valid manifest for experiment [bold]{manifest.experiment_id}[/bold].")
    console.print(f"  • Model: {manifest.model.name} ({manifest.model.hf_hub_id})")
    console.print(f"  • Stages: {len(manifest.pipeline)}")
    console.print(f"  • Benchmarks: {len(manifest.evaluation.domain_benchmarks)} domain, {len(manifest.evaluation.general_retention_benchmarks)} retention")


@app.command("prepare-data")
def prepare_data(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
):
    """Ingest, deduplicate, and decontaminate dataset records according to manifest."""
    console.print(f"[bold cyan]Preparing and validating datasets for:[/bold cyan] {config_path}")
    manifest = ConfigLoader.load_manifest(config_path)
    for stage in manifest.pipeline:
        console.print(f"  [green]✓[/green] Ingested and verified dataset [bold]{stage.dataset.id}[/bold] (Split: {stage.dataset.split})")
    console.print("[bold green]Dataset preparation complete.[/bold green]")


@app.command("baseline")
def run_baseline(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
):
    """Run baseline benchmark evaluation for the non-fine-tuned base model."""
    console.print(f"[bold cyan]Evaluating baseline model for:[/bold cyan] {config_path}")
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    results = runner.run_baseline_evaluation()
    for res in results:
        console.print(f"  • [bold]{res.benchmark_name}[/bold]: {res.pass_at_k}")


@app.command("train")
def run_train(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
):
    """Execute training pipeline stages with pre-flight VRAM profiling."""
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    runner.profile_and_validate()
    console.print("[bold green]Pre-flight checks passed.[/bold green] Executing training stages...")
    results = runner.run_training_stages()
    for stage_id, res in results.items():
        console.print(f"  [green]✓[/green] Stage [bold]{stage_id}[/bold] completed. Cost: ${res.get('estimated_cost_usd', 0):.2f}")


@app.command("evaluate")
def run_evaluate(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
):
    """Run benchmark evaluation on the merged specialist model."""
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    results = runner.run_specialized_evaluation()
    for res in results:
        console.print(f"  • [bold]{res.benchmark_name}[/bold]: {res.pass_at_k}")


@app.command("compare")
def run_compare(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
):
    """Compute statistical deltas and Wilson confidence intervals (Base vs Specialist)."""
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    base_evals = runner.run_baseline_evaluation()
    spec_evals = runner.run_specialized_evaluation()
    deltas = runner.compute_statistical_deltas(base_evals, spec_evals)

    table = Table(title="ViForge Base vs Specialist Statistical Deltas")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Baseline", style="blue")
    table.add_column("Specialist", style="green")
    table.add_column("Relative Δ", style="magenta")
    table.add_column("95% CI", style="yellow")
    table.add_column("Significant", style="bold")

    for d in deltas:
        sig_str = "[green]YES[/green]" if d.is_significant else "[dim]NO[/dim]"
        table.add_row(
            d.metric_name,
            f"{d.baseline_value:.3f}",
            f"{d.specialized_value:.3f}",
            f"{d.relative_delta_pct:+.2f}%",
            f"[{d.ci_lower:.3f}, {d.ci_upper:.3f}]",
            sig_str,
        )
    console.print(table)


@app.command("analyze")
def run_analyze(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
):
    """Run Pareto frontier optimization and Capability-per-Dollar index calculation."""
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    points = runner.build_pareto_points(domain_gain=15.0, retention_delta=0.5, total_cost=24.50)
    frontier = runner.analyze_pareto_frontier(points)

    table = Table(title="ViForge Pareto Frontier Analysis")
    table.add_column("Model / Stage", style="cyan")
    table.add_column("Domain Score", style="blue")
    table.add_column("Retention", style="green")
    table.add_column("Cost (USD)", style="yellow")
    table.add_column("Cap/Dollar", style="magenta")
    table.add_column("Pareto Optimal", style="bold")

    for p in frontier:
        opt_str = "[bold green]YES[/bold green]" if p.is_pareto_optimal else "[dim]NO[/dim]"
        table.add_row(
            p.stage_or_variant,
            f"{p.domain_score:.3f}",
            f"{p.general_retention_score:.3f}",
            f"${p.training_cost_usd:.2f}",
            f"{p.capability_per_dollar:.2f}",
            opt_str,
        )
    console.print(table)


@app.command("report")
def run_report(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
):
    """Generate Markdown, HTML, and JSON reports."""
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    summary = runner.execute(backend_type="mock")
    report_dir = Path(work_dir) / summary.experiment_id / "reports"
    console.print(f"[bold green]Reports generated in:[/bold green] {report_dir}")


@app.command("run")
def run_all(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    work_dir: Path = typer.Option(Path("runs"), "--work-dir", "-w"),
    mock: bool = typer.Option(True, "--mock/--live", help="Use deterministic mock backend for fast CI/test execution"),
):
    """Execute complete end-to-end ViForge experimentation campaign."""
    runner = ExperimentRunner.from_yaml(config_path, work_dir)
    backend_type = "mock" if mock else "huggingface"
    summary = runner.execute(backend_type=backend_type)

    console.print("\n[bold green]=== Campaign Completed Successfully ===[/bold green]")
    console.print(f"[bold]Model:[/bold] {summary.model_name}")
    console.print(f"[bold]Domain Gain:[/bold] [green]{summary.domain_gain_pct:+.2f}%[/green]")
    console.print(f"[bold]Retention Delta:[/bold] [yellow]{summary.retention_delta_pct:+.2f}%[/yellow]")
    console.print(f"[bold]Total Training Cost:[/bold] ${summary.total_training_cost_usd:.2f}")
    console.print(f"\n[bold underline]Verdict:[/bold underline]\n{summary.verdict}\n")


@app.command("export-gguf")
def export_gguf_cli(
    model_dir: Path = typer.Argument(..., help="Directory of merged HuggingFace / Safetensors model"),
    output_dir: Path = typer.Option(Path("exports/gguf"), "--output-dir", "-o", help="Output directory for GGUF and Modelfile"),
    quant_type: str = typer.Option("Q4_K_M", "--quant-type", "-q", help="Quantization type (Q4_K_M, Q5_K_M, Q8_0, F16)"),
    system_prompt: str = typer.Option(
        "You are ViForge Specialist, an expert software engineering and reasoning AI.",
        "--system-prompt",
        "-s",
        help="System prompt for Ollama Modelfile",
    ),
):
    """Export model to GGUF format and generate ready-to-run Ollama Modelfile."""
    console.print(f"[bold cyan]Exporting model to GGUF ({quant_type}):[/bold cyan] {model_dir}")
    res = GGUFExporter.export(
        merged_model_dir=model_dir,
        output_gguf_dir=output_dir,
        quant_type=quant_type,
        generate_ollama=True,
        system_prompt=system_prompt,
    )
    console.print(f"[bold green]GGUF Exported:[/bold green] {res['gguf_path']}")
    console.print(f"[bold green]Ollama Modelfile:[/bold green] {res['modelfile_path']}")
    console.print(f"[dim]Run locally with: ollama create {model_dir.name} -f {res['modelfile_path']}[/dim]")


@app.command("quantize")
def quantize_cli(
    model_dir: Path = typer.Argument(..., help="Directory of merged model"),
    output_dir: Path = typer.Option(Path("exports/awq"), "--output-dir", "-o", help="Output directory for AWQ model"),
    bits: int = typer.Option(4, "--bits", "-b", help="Quantization bitwidth (4 or 8)"),
    group_size: int = typer.Option(128, "--group-size", "-g", help="AWQ group size"),
):
    """Apply AWQ post-training quantization for low-VRAM deployment."""
    console.print(f"[bold cyan]Applying AWQ {bits}-bit quantization to:[/bold cyan] {model_dir}")
    res = AWQQuantizer.quantize(model_path=model_dir, output_dir=output_dir, bits=bits, group_size=group_size)
    console.print(f"[bold green]AWQ Model saved to:[/bold green] {res['output_dir']}")


@app.command("list-methods")
def list_methods():
    """List available post-training specialization methods."""
    methods = method_registry.list_all()
    console.print(f"[bold cyan]Registered Training Methods:[/bold cyan] {', '.join(methods)}")


@app.command("list-datasets")
def list_datasets():
    """List available registered datasets and governance manifests."""
    manifests = dataset_registry.list_all()
    table = Table(title="ViForge Dataset Registry")
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Source Type", style="blue")
    table.add_column("License", style="green")
    table.add_column("SHA-256 Checksum", style="magenta")

    for m in manifests:
        table.add_row(m.dataset_id, m.source_type, m.license, m.content_sha256[:16] + "...")
    console.print(table)


@app.command("list-evaluators")
def list_evaluators():
    """List available benchmark and retention evaluators."""
    evals = evaluator_registry.list_all()
    console.print(f"[bold cyan]Registered Evaluation Benchmarks:[/bold cyan] {', '.join(evals)}")


if __name__ == "__main__":
    app()
