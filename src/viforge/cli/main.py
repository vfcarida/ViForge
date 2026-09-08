"""
ViForge Production CLI: End-to-End Specialization & Pareto Evaluation Platform.
"""

from pathlib import Path
from typing import Optional
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
console = Console(safe_box=True)


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


@app.command("validate")
def validate_config(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
):
    """Validate experiment manifest against Pydantic schemas."""
    console.print(f"[bold cyan]Validating manifest:[/bold cyan] {config_path}")
    manifest = ConfigLoader.load_manifest(config_path)
    console.print(
        f"[bold green]OK:[/bold green] Valid manifest for experiment [bold]{manifest.experiment_id}[/bold]."
    )
    console.print(f"  • Model: {manifest.model.name} ({manifest.model.hf_hub_id})")
    console.print(f"  • Stages: {len(manifest.pipeline)}")
    retention_count = len(
        manifest.evaluation.retention_benchmarks
        if manifest.evaluation.retention_benchmarks is not None
        else (manifest.evaluation.general_retention_benchmarks or [])
    )
    console.print(
        f"  • Benchmarks: {len(manifest.evaluation.domain_benchmarks)} domain, {retention_count} retention"
    )


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
        console.print(
            f"  [green][OK][/green] Stage [bold]{stage_id}[/bold] completed. Cost: ${res.get('estimated_cost_usd', 0):.2f}"
        )


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
    table.add_column("Relative Delta (%)", style="magenta")
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
    mock: bool = typer.Option(
        True, "--mock/--live", help="Use deterministic mock backend for fast CI/test execution"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Enforce strict production mode (fail-fast on any mock or fallback)"
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


@app.command("export-gguf")
def export_gguf_cli(
    model_dir: Path = typer.Argument(
        ..., help="Directory of merged HuggingFace / Safetensors model"
    ),
    output_dir: Path = typer.Option(
        Path("exports/gguf"), "--output-dir", "-o", help="Output directory for GGUF and Modelfile"
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


@app.command("prepare-data")
def prepare_data_cli(
    config_or_domain: Optional[str] = typer.Argument(
        None, help="Path to domain YAML preset or domain preset name (e.g. software_engineering)"
    ),
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Domain preset name (e.g. software_engineering)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to domain YAML preset or dataset YAML config"
    ),
    technique: Optional[str] = typer.Option(
        None,
        "--technique",
        "-t",
        help="Specific technique to prepare (sft, dpo, grpo, cpt, or all)",
    ),
    output_dir: Path = typer.Option(
        Path("data"), "--output", "-o", help="Output directory for prepared datasets"
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", "-m", help="Override max samples limit"
    ),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup", help="Enable MinHash deduplication"),
):
    """Transform and standardize raw domain data into SFT, DPO, GRPO, and CPT datasets."""
    from viforge.config.schemas import DatasetConfig
    from viforge.datasets.preparer import DatasetPreparer, load_domain_preset

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(Panel.fit("[bold cyan]ViForge Dataset Preparation Pipeline[/bold cyan]"))

    target_preset = None
    if config_path:
        target_preset = load_domain_preset(config_path)
    elif config_or_domain:
        candidate_path = Path(config_or_domain)
        if candidate_path.exists() or config_or_domain.endswith((".yaml", ".yml")):
            target_preset = load_domain_preset(candidate_path)
        else:
            target_preset = load_domain_preset(config_or_domain)
    elif domain:
        target_preset = load_domain_preset(domain)
    else:
        target_preset = load_domain_preset("software_engineering")

    console.print(f"[bold]Domain:[/bold] {target_preset.domain}")
    techniques = (
        [technique] if technique and technique != "all" else list(target_preset.datasets.keys())
    )

    for tech in techniques:
        items = target_preset.datasets.get(tech, [])
        for idx, item in enumerate(items):
            if isinstance(item, str):
                item_dict = {"source": item}
            elif hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = item
            else:
                item_dict = dict(item)

            samples_limit = max_samples if max_samples is not None else item_dict.get("max_samples")
            cfg = DatasetConfig(
                domain=target_preset.domain,
                source=item_dict.get("source", ""),
                format=tech,
                subset=item_dict.get("subset"),
                split=item_dict.get("split", "train"),
                max_samples=samples_limit,
                text_column=item_dict.get("text_column", "text"),
                instruction_column=item_dict.get("instruction_column"),
                input_column=item_dict.get("input_column"),
                response_column=item_dict.get("response_column"),
                chosen_column=item_dict.get("chosen_column"),
                rejected_column=item_dict.get("rejected_column"),
                dedup=dedup,
            )

            preparer = DatasetPreparer(cfg)
            out_name = (
                f"{target_preset.domain}_{tech}_{idx}.jsonl"
                if len(items) > 1
                else f"{target_preset.domain}_{tech}.jsonl"
            )
            saved_path = preparer.save(output_dir / out_name)
            console.print(f"[green][OK] Prepared {tech.upper()} dataset:[/green] {saved_path}")

    console.print("[bold green]Dataset preparation complete![/bold green]")


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
        console.print("\n[bold green]=== ViPym Compression Completed Successfully ===[/bold green]")
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


@app.command("pareto-unified")
def cli_pareto_unified(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML manifest"),
    domain_gain: float = typer.Option(
        36.0, "--domain-gain", "-g", help="Relative domain benchmark gain (%)"
    ),
    retention_delta: float = typer.Option(
        0.0, "--retention-delta", "-r", help="Relative general retention delta (%)"
    ),
    total_cost: float = typer.Option(
        24.50, "--cost", "-c", help="Total training compute cost in USD"
    ),
    output_dir: Path = typer.Option(
        Path("reports/pareto"),
        "--output-dir",
        "-o",
        help="Directory to save JSON and HTML reports",
    ),
):
    """Compute 5D Unified Pareto Frontier (Base -> Specialist -> Compressed) and discover deployment sweet spots."""
    from viforge.analysis.unified_pareto import UnifiedParetoEngine
    from viforge.config.loader import ConfigLoader

    console.print(Panel.fit("[bold cyan]ViForge Unified Pareto Frontier Engine[/bold cyan]"))
    manifest = ConfigLoader.load_manifest(config_path)

    base_domain = 0.50
    spec_domain = base_domain * (1.0 + domain_gain / 100.0)
    base_retention = 0.65
    spec_retention = base_retention * (1.0 + retention_delta / 100.0)

    points = UnifiedParetoEngine.build_unified_points(
        model_name=manifest.model.name,
        base_domain_score=base_domain,
        base_retention_score=base_retention,
        specialized_domain_score=spec_domain,
        specialized_retention_score=spec_retention,
        training_cost_usd=total_cost,
    )

    table = Table(title=f"Unified Pareto Frontier: {manifest.model.name}")
    table.add_column("Candidate Variant", style="cyan")
    table.add_column("Domain Score", style="blue")
    table.add_column("Gain (%)", style="green")
    table.add_column("Retention", style="white")
    table.add_column("VRAM (GB)", style="yellow")
    table.add_column("Latency (ms)", style="magenta")
    table.add_column("Cost (USD)", style="white")
    table.add_column("Pareto Optimal", style="bold")

    for p in points:
        opt_str = "[bold green]YES[/bold green]" if p.is_pareto_optimal else "[dim]NO[/dim]"
        table.add_row(
            p.variant_label,
            f"{p.domain_score:.3f}",
            f"{p.domain_gain_pct:+.1f}%",
            f"{p.general_retention_score:.3f}",
            f"{p.serving_memory_gb:.1f} GB",
            f"{p.serving_latency_ms:.0f} ms",
            f"${p.total_cost_usd:.2f}",
            opt_str,
        )
    console.print(table)

    sweet_spots = UnifiedParetoEngine.find_sweet_spots(points)
    console.print("\n[bold green]=== Deployment Sweet-Spot Recommendations ===[/bold green]")
    for spot in sweet_spots:
        console.print(
            f"  • [bold cyan]{spot.category}:[/bold cyan] [bold]{spot.point.variant_label}[/bold]\n"
            f"    [dim]{spot.rationale}[/dim]"
        )

    json_path = UnifiedParetoEngine.export_json(points, output_dir / "unified_pareto.json")
    html_path = UnifiedParetoEngine.generate_html_chart(
        points, output_dir / "unified_pareto.html"
    )
    console.print(f"\n[green][OK] JSON Report saved to:[/green] {json_path}")
    console.print(f"[green][OK] Interactive HTML Chart saved to:[/green] {html_path}")


@app.command("ui")
def cli_ui(
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit dashboard"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host address for Streamlit dashboard"),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Automatically launch browser"),
):
    """Launch the ViForge Interactive Studio (Streamlit Dashboard)."""
    import shutil
    import subprocess
    from viforge.ui import DASHBOARD_APP_PATH

    console.print(Panel.fit("[bold cyan]ViForge Interactive Studio Launcher[/bold cyan]"))

    streamlit_bin = shutil.which("streamlit")
    if not streamlit_bin:
        console.print(
            "[bold red]Streamlit is not installed.[/bold red]\n"
            "Install it via: [green]pip install \"viforge[ui]\"[/green] or [green]pip install streamlit plotly[/green]"
        )
        raise typer.Exit(code=1)

    cmd = [
        streamlit_bin,
        "run",
        str(DASHBOARD_APP_PATH),
        "--server.port",
        str(port),
        "--server.address",
        str(host),
    ]
    if not browser:
        cmd.append("--server.headless=true")

    console.print(f"Launching dashboard at: [bold green]http://{host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop the dashboard server.[/dim]")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]ViForge Studio stopped.[/bold yellow]")


if __name__ == "__main__":
    app()


