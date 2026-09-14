"""
Model validation, benchmark evaluation, statistical comparison, and Pareto analysis CLI commands.
"""

from pathlib import Path
from typing import List
import typer
from rich.panel import Panel
from rich.table import Table

from viforge.cli.console import console
from viforge.config.loader import ConfigLoader
from viforge.evaluation.suites import evaluator_registry
from viforge.experiments.runner import ExperimentRunner


def register(app: typer.Typer) -> None:
    """Register evaluation, validation, and Pareto analysis commands."""

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
        points = runner.build_pareto_points(
            domain_gain=15.0, retention_delta=0.5, total_cost=24.50
        )
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

    @app.command("list-evaluators")
    def list_evaluators():
        """List available benchmark and retention evaluators."""
        evals = evaluator_registry.list_all()
        console.print(f"[bold cyan]Registered Evaluation Benchmarks:[/bold cyan] {', '.join(evals)}")

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

    @app.command("compare-campaigns")
    def cli_compare_campaigns(
        campaign_files: List[Path] = typer.Argument(
            ..., help="List of Pareto JSON report files from distinct models/campaigns"
        ),
        output_dir: Path = typer.Option(
            Path("reports/cross_model_pareto"),
            "--output-dir",
            "-o",
            help="Directory to save cross-model JSON and HTML comparison",
        ),
    ):
        """Cross-model Multi-Campaign Pareto Benchmark: compare multiple model architectures on a unified frontier."""
        from viforge.analysis.unified_pareto import MultiCampaignParetoComparator

        console.print(
            Panel.fit("[bold cyan]ViForge Cross-Model Multi-Campaign Pareto Comparator[/bold cyan]")
        )

        campaigns = {}
        for f in campaign_files:
            pts = MultiCampaignParetoComparator.load_campaign(f)
            model_key = pts[0].model_name if pts else f.stem
            campaigns[model_key] = pts
            console.print(
                f"  • Loaded campaign [bold green]{model_key}[/bold green]: {len(pts)} candidate variants"
            )

        global_points = MultiCampaignParetoComparator.compute_global_frontier(campaigns)

        table = Table(title="Cross-Model Multi-Campaign Pareto Benchmark")
        table.add_column("Model Family", style="cyan")
        table.add_column("Variants", style="magenta")
        table.add_column("Best Domain", style="blue")
        table.add_column("Best Retention", style="green")
        table.add_column("Min VRAM", style="yellow")
        table.add_column("Min Latency", style="white")
        table.add_column("Global Optimal", style="bold green")

        for model_name, pts in campaigns.items():
            global_opt_count = sum(
                1 for p in global_points if p.model_name == model_name and p.is_pareto_optimal
            )
            best_domain = max((p.domain_score for p in pts), default=0.0)
            best_ret = max((p.general_retention_score for p in pts), default=0.0)
            min_vram = min((p.serving_memory_gb for p in pts), default=0.0)
            min_lat = min((p.serving_latency_ms for p in pts), default=0.0)
            table.add_row(
                model_name,
                str(len(pts)),
                f"{best_domain:.3f}",
                f"{best_ret:.3f}",
                f"{min_vram:.1f} GB",
                f"{min_lat:.0f} ms",
                str(global_opt_count),
            )
        console.print(table)

        json_path = MultiCampaignParetoComparator.export_multi_model_json(
            campaigns, global_points, output_dir / "multi_model_pareto.json"
        )
        html_path = MultiCampaignParetoComparator.generate_multi_model_html_chart(
            campaigns, global_points, output_dir / "multi_model_pareto.html"
        )
        console.print(f"\n[green][OK] Cross-Model JSON Report saved to:[/green] {json_path}")
        console.print(f"[green][OK] Interactive Cross-Model HTML Chart saved to:[/green] {html_path}")

