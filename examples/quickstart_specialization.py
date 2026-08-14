"""
ViForge Quickstart: Programmatic Specialization and Pareto Evaluation Example.
"""

from pathlib import Path
from viforge.config.loader import ConfigLoader
from viforge.experiments.runner import ExperimentRunner


def main():
    manifest_path = Path("configs/experiments/deepseek_v4_pro_software_engineering.yaml")
    print(f"Loading experiment configuration from: {manifest_path}")

    runner = ExperimentRunner.from_yaml(manifest_path, work_dir=Path("runs"))

    # 1. Run Pre-flight Checks
    print("\n--- 1. Running Pre-flight Feasibility Checks ---")
    preflight = runner.run_preflight_checks()
    for stg in preflight["stages"]:
        print(f"Stage {stg['stage_id']}: VRAM {stg['vram']['total_estimated_vram_gb']}GB (Passed)")

    # 2. Execute Full Campaign (using mock backend for local demonstration)
    print("\n--- 2. Executing Experiment Campaign ---")
    summary = runner.execute(backend_type="mock")

    print("\n--- 3. Specialization Results ---")
    print(f"Model: {summary.model_name}")
    print(f"Domain Score: {summary.baseline_domain_score:.1%} -> {summary.specialized_domain_score:.1%} ({summary.domain_gain_pct:+.1f}%)")
    print(f"General Retention: {summary.baseline_retention_score:.1%} -> {summary.specialized_retention_score:.1%} ({summary.retention_delta_pct:+.1f}%)")
    print(f"Total Training Cost: ${summary.total_training_cost_usd:.2f}")
    print(f"\nVerdict: {summary.verdict}")


if __name__ == "__main__":
    main()
