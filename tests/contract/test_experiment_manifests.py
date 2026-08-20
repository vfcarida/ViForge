"""
Contract tests for experiment manifests in configs/experiments/.

Validates that every YAML manifest in the experiments directory:
- Parses without error via ConfigLoader
- Has a non-empty pipeline
- Has domain and retention benchmarks
- DAG dependencies reference existing stage_ids
- Hyperparameters satisfy basic sanity bounds
"""

from pathlib import Path

import pytest

from viforge.config.loader import ConfigLoader
from viforge.config.schemas import ExperimentManifest

EXPERIMENTS_DIR = Path(__file__).parents[2] / "configs" / "experiments"

ALL_MANIFESTS = list(EXPERIMENTS_DIR.glob("*.yaml"))


def _load_manifest(path: Path) -> ExperimentManifest:
    return ConfigLoader.load_manifest(path)


# --------------------------------------------------------------------------- #
# Parametric: every manifest in configs/experiments/ must load cleanly         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("manifest_path", ALL_MANIFESTS, ids=lambda p: p.stem)
@pytest.mark.contract
def test_manifest_loads(manifest_path: Path):
    """Every experiment YAML must parse into a valid ExperimentManifest."""
    m = _load_manifest(manifest_path)
    assert m.experiment_id, "experiment_id must not be empty"
    assert m.model.hf_hub_id, "model.hf_hub_id must not be empty"
    assert len(m.pipeline) >= 1, "pipeline must have at least one stage"


@pytest.mark.parametrize("manifest_path", ALL_MANIFESTS, ids=lambda p: p.stem)
@pytest.mark.contract
def test_manifest_dag_consistency(manifest_path: Path):
    """All depends_on references must point to existing stage_ids."""
    m = _load_manifest(manifest_path)
    stage_ids = {s.stage_id for s in m.pipeline}
    for stage in m.pipeline:
        if stage.depends_on is not None:
            assert stage.depends_on in stage_ids, (
                f"Stage '{stage.stage_id}' depends_on '{stage.depends_on}' "
                f"which does not exist. Available: {sorted(stage_ids)}"
            )


@pytest.mark.parametrize("manifest_path", ALL_MANIFESTS, ids=lambda p: p.stem)
@pytest.mark.contract
def test_manifest_evaluation_benchmarks(manifest_path: Path):
    """Every manifest must specify at least one domain and one retention benchmark."""
    m = _load_manifest(manifest_path)
    domain_bm = m.evaluation.domain_benchmarks
    retention_bm = m.evaluation.retention_benchmarks or m.evaluation.general_retention_benchmarks
    assert len(domain_bm) >= 1, "Must have at least one domain benchmark"
    assert retention_bm and len(retention_bm) >= 1, "Must have at least one retention benchmark"


@pytest.mark.parametrize("manifest_path", ALL_MANIFESTS, ids=lambda p: p.stem)
@pytest.mark.contract
def test_manifest_hyperparameter_sanity(manifest_path: Path):
    """Key hyperparameters must be within reasonable bounds."""
    m = _load_manifest(manifest_path)
    for stage in m.pipeline:
        hp = stage.hyperparameters
        assert 0 < hp.learning_rate <= 1.0, (
            f"Stage '{stage.stage_id}': learning_rate {hp.learning_rate} out of range"
        )
        assert 1 <= hp.num_epochs <= 20, (
            f"Stage '{stage.stage_id}': num_epochs {hp.num_epochs} out of range"
        )
        assert 128 <= hp.max_seq_len <= 131072, (
            f"Stage '{stage.stage_id}': max_seq_len {hp.max_seq_len} out of range"
        )
        assert hp.lora_rank >= 0, (
            f"Stage '{stage.stage_id}': lora_rank {hp.lora_rank} must be non-negative"
        )


# --------------------------------------------------------------------------- #
# Specific: master case study manifest                                          #
# --------------------------------------------------------------------------- #

MASTER_MANIFEST = EXPERIMENTS_DIR / "deepseek_v4_pro_software_engineering.yaml"


@pytest.mark.contract
def test_case_study_manifest_techniques():
    """Master manifest must contain all 5 required technique stages."""
    m = _load_manifest(MASTER_MANIFEST)
    methods = {s.method.value for s in m.pipeline}
    assert "cpt" in methods, "CPT stage missing from master manifest"
    assert "qlora" in methods, "QLoRA stage missing from master manifest"
    assert "lora" in methods, "LoRA stage missing from master manifest"
    assert "dpo" in methods, "DPO stage missing from master manifest"


@pytest.mark.contract
def test_case_study_manifest_evaluation_config():
    """Master manifest must have HumanEval+, SWE-bench, MMLU-Pro, GSM8K, ARC."""
    m = _load_manifest(MASTER_MANIFEST)

    domain_names = {b.name if hasattr(b, "name") else b for b in m.evaluation.domain_benchmarks}
    retention_names = {
        b.name if hasattr(b, "name") else b
        for b in (
            m.evaluation.retention_benchmarks or m.evaluation.general_retention_benchmarks or []
        )
    }

    assert "humaneval_plus" in domain_names, "HumanEval+ missing from domain benchmarks"
    assert "swe_bench_lite" in domain_names, "SWE-bench Lite missing from domain benchmarks"
    assert "mmlu_pro" in retention_names, "MMLU-Pro missing from retention benchmarks"
    assert "gsm8k" in retention_names, "GSM8K missing from retention benchmarks"
    assert "arc_challenge" in retention_names, "ARC-Challenge missing from retention benchmarks"


@pytest.mark.contract
def test_case_study_manifest_reproducibility_settings():
    """Evaluation must use temperature=0 and fixed seed for reproducibility."""
    m = _load_manifest(MASTER_MANIFEST)
    assert m.evaluation.sampling.temperature == 0.0, "temperature must be 0.0 for reproducibility"
    assert m.evaluation.sampling.seed == 42, "seed must be 42 for reproducibility"


@pytest.mark.contract
def test_case_study_manifest_retention_threshold():
    """Retention threshold must enforce ≤ 10% degradation limit."""
    m = _load_manifest(MASTER_MANIFEST)
    assert m.evaluation.retention_threshold >= 0.90, (
        "retention_threshold must be >= 0.90 (≤10% degradation allowed)"
    )


@pytest.mark.contract
def test_case_study_manifest_qlora_entry_point():
    """QLoRA r=32 stage must not depend on any other stage (24GB GPU entry point)."""
    m = _load_manifest(MASTER_MANIFEST)
    qlora_stages = [s for s in m.pipeline if s.method.value == "qlora"]
    assert len(qlora_stages) >= 1, "No QLoRA stage found"
    for stage in qlora_stages:
        assert stage.depends_on is None, (
            f"QLoRA stage '{stage.stage_id}' must not depend on other stages "
            f"(must be runnable standalone on a 24GB GPU)"
        )
        assert stage.hyperparameters.lora_rank == 32, (
            f"Expected QLoRA r=32, got r={stage.hyperparameters.lora_rank}"
        )


@pytest.mark.contract
def test_case_study_manifest_dpo_depends_on_lora():
    """DPO stage must depend on a LoRA SFT checkpoint (not CPT or QLoRA)."""
    m = _load_manifest(MASTER_MANIFEST)
    dpo_stages = [s for s in m.pipeline if s.method.value == "dpo"]
    assert len(dpo_stages) >= 1, "No DPO stage found"
    stage_map = {s.stage_id: s for s in m.pipeline}
    for dpo in dpo_stages:
        assert dpo.depends_on is not None, "DPO stage must have a depends_on"
        parent = stage_map.get(dpo.depends_on)
        assert parent is not None, f"DPO depends_on '{dpo.depends_on}' not found"
        assert parent.method.value == "lora", (
            f"DPO should depend on a LoRA stage, not '{parent.method.value}'"
        )
