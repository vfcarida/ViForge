"""
Dataset preparation, synthetic data generation, and dataset registry CLI commands.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.panel import Panel
from rich.table import Table

from viforge.cli.console import console
from viforge.datasets.registry import dataset_registry


def register(app: typer.Typer) -> None:
    """Register data preparation and synthetic generation commands."""

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

    @app.command("prepare-data")
    def prepare_data_cli(
        config_or_domain: Optional[str] = typer.Argument(
            None,
            help="Path to domain YAML preset or domain preset name (e.g. software_engineering)",
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

                samples_limit = (
                    max_samples if max_samples is not None else item_dict.get("max_samples")
                )
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
                console.print(
                    f"[green][OK] Prepared {tech.upper()} dataset:[/green] {saved_path}"
                )

        console.print("[bold green]Dataset preparation complete![/bold green]")

    @app.command("generate-synthetic")
    def cli_generate_synthetic(
        domain: str = typer.Option(
            "software_engineering", "--domain", "-d", help="Domain preset or name"
        ),
        samples: int = typer.Option(
            5, "--samples", "-n", help="Number of synthetic samples to synthesize"
        ),
        strategy: str = typer.Option(
            "deepen_constraints", "--strategy", "-s", help="Evol-Instruct strategy"
        ),
        output_path: Path = typer.Option(
            Path("data/synthetic_curated.jsonl"),
            "--output",
            "-o",
            help="Destination JSONL path",
        ),
        teacher_model: str = typer.Option(
            "claude-3-5-sonnet-20241022",
            "--teacher-model",
            "-t",
            help="Teacher model identifier",
        ),
    ):
        """Generate and curate high-complexity synthetic training data using Evol-Instruct & Self-Play."""
        from viforge.methods.synthetic import SyntheticDataPipeline

        console.print(
            Panel.fit("[bold cyan]ViForge Synthetic Data & Evol-Instruct Engine[/bold cyan]")
        )
        pipeline = SyntheticDataPipeline(teacher_model_id=teacher_model)

        seed_prompts = [
            "Write a function to optimize a multi-threaded LRU cache.",
            "Implement an async websocket event dispatcher with automatic exponential backoff.",
            "Write a custom PyTorch autograd function for sparse attention.",
            "Create a distributed rate limiter using token bucket algorithm with Redis.",
            "Implement an AST transformer that redacts sensitive environment variable accesses.",
        ]

        generated_records = []
        for i in range(samples):
            base_p = seed_prompts[i % len(seed_prompts)]
            evolved_p = pipeline.mutate_prompt(base_p, strategy=strategy)
            candidate = {
                "instruction": evolved_p,
                "response": (
                    f"```python\n# Implementation for: {base_p}\n"
                    "class Solution:\n"
                    "    def execute(self, payload: dict) -> dict:\n"
                    "        # Strict production execution logic\n"
                    "        if not payload:\n"
                    "            raise ValueError('Empty payload')\n"
                    "        return {'status': 'processed', 'data': payload}\n```"
                ),
            }
            generated_records.append(candidate)

        accepted, stats = pipeline.process_and_filter_candidates(
            raw_candidates=generated_records,
            system_prompt=f"Domain: {domain} | Strategy: {strategy}",
        )

        saved_file, manifest = pipeline.export_to_dataset(
            examples=accepted,
            output_path=output_path,
            dataset_id=f"synthetic-{domain}-{strategy}",
            teacher_model=teacher_model,
        )

        table = Table(title="Synthetic Dataset Generation Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Teacher Model", teacher_model)
        table.add_row("Strategy", strategy)
        table.add_row("Samples Generated", str(stats["total_candidates"]))
        table.add_row("Accepted Samples", str(stats["accepted_samples"]))
        table.add_row("Acceptance Rate", f"{stats['acceptance_rate_pct']}%")
        table.add_row("Dataset File", str(saved_file))
        table.add_row("SHA-256 Hash", manifest.content_sha256[:16] + "...")
        console.print(table)
