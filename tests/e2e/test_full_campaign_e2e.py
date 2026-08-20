"""
End-to-End Campaign Orchestration Test for DeepSeek V4 Pro Specialization.
"""

from pathlib import Path

import pytest

from viforge.experiments.runner import ExperimentRunner


@pytest.mark.e2e
@pytest.mark.slow
def test_full_campaign_e2e_execution(tmp_path: Path, sample_config_path: Path):
    manifest_path = sample_config_path
    assert manifest_path.exists()

    runner = ExperimentRunner.from_yaml(manifest_path, work_dir=tmp_path)
    summary = runner.execute(backend_type="mock")

    assert summary.experiment_id == "deepseek_v4_pro_software_engineering_master"
    assert summary.model_name == "DeepSeek V4 Pro"
    assert len(summary.stages) == 3
    assert summary.domain_gain_pct > 0
    assert len(summary.pareto_frontier) >= 2
    assert "HYPOTHESIS CONFIRMED" in summary.verdict

    # Check generated report files
    reports_dir = tmp_path / summary.experiment_id / "reports"
    assert (reports_dir / f"{summary.experiment_id}_report.json").exists()
    assert (reports_dir / f"{summary.experiment_id}_report.md").exists()
    assert (reports_dir / f"{summary.experiment_id}_report.html").exists()
