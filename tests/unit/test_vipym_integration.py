"""
Unit test suite for ViForge x ViPym Downstream Compression Integration.
"""

from pathlib import Path
import pytest
import yaml
from typer.testing import CliRunner

from viforge.cli.main import app
from viforge.integrations.vipym import (
    RECIPE_CATALOG,
    CompressionRecipe,
    RecipeDefinition,
    ViPymExporter,
    ViPymRunner,
    is_vipym_available,
)


@pytest.fixture
def cli_runner():
    return CliRunner()


def test_recipe_catalog_coverage():
    """Verify that all enum entries map to valid RecipeDefinitions."""
    for recipe_enum in CompressionRecipe:
        assert recipe_enum in RECIPE_CATALOG
        defn = RECIPE_CATALOG[recipe_enum]
        assert isinstance(defn, RecipeDefinition)
        assert defn.method
        assert defn.scheme
        assert defn.description

    # Check string resolution
    resolved = ViPymExporter.get_recipe_definition("smoothquant_w8a8")
    assert resolved.recipe == CompressionRecipe.SMOOTHQUANT_W8A8

    # Check invalid recipe resolution
    with pytest.raises(ValueError, match="Unknown compression recipe"):
        ViPymExporter.get_recipe_definition("invalid_recipe_xyz")


def test_benchmark_mapping():
    """Verify benchmark translation from ViForge names to ViPym evaluation suite IDs."""
    inputs = ["humaneval_plus", "swe_bench_lite", "gsm8k", "arc_challenge", "mmlu_pro", "custom_unknown"]
    mapped = ViPymExporter.map_benchmarks(inputs)
    assert "humaneval" in mapped
    assert "swe_bench" in mapped
    assert "gsm8k" in mapped
    assert "arc" in mapped
    assert "mmlu" in mapped


def test_build_vipym_config_dict_quantization():
    """Verify generation of quantization config dictionary conforming to ViPym schema."""
    config = ViPymExporter.build_vipym_config_dict(
        model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        recipe=CompressionRecipe.AUTOROUND_W4A16,
        experiment_id="test_autoround",
        calibration_dataset="wikitext",
        calibration_samples=256,
        evaluation_suites=["humaneval", "gsm8k"],
    )

    assert config["experiment_id"] == "test_autoround"
    assert config["model"]["id"] == "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
    assert len(config["compression_pipeline"]) == 1

    stage = config["compression_pipeline"][0]
    assert stage["method"] == "autoround"
    assert stage["scheme"] == "W4A16"
    assert stage["calibration"]["dataset_name"] == "wikitext"
    assert stage["calibration"]["num_samples"] == 256
    assert config["evaluation"]["suites"] == ["humaneval", "gsm8k"]


def test_build_vipym_config_dict_distillation():
    """Verify distillation configuration correctly sets teacher and student model targets."""
    config = ViPymExporter.build_vipym_config_dict(
        model_id="/path/to/specialist_teacher",
        recipe=CompressionRecipe.DISTILL_LOGIT,
        student_model_id="Qwen/Qwen2.5-Coder-1.5B",
    )

    stage = config["compression_pipeline"][0]
    assert stage["method"] == "distill_logit"
    assert stage["scheme"] == "DISTILL_STUDENT"
    assert stage["parameters"]["teacher_model"] == "/path/to/specialist_teacher"
    # Target model to compress/train is the student
    assert config["model"]["id"] == "Qwen/Qwen2.5-Coder-1.5B"


def test_export_from_model_dir(tmp_path: Path):
    """Test exporting from a mock model directory to a YAML file."""
    model_dir = tmp_path / "dummy_merged_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    out_yaml = tmp_path / "exports" / "vipym_sq.yaml"
    res_path = ViPymExporter.export_from_model_dir(
        model_dir=model_dir,
        recipe=CompressionRecipe.SMOOTHQUANT_W8A8,
        output_yaml_path=out_yaml,
        calibration_samples=128,
    )

    assert res_path == out_yaml
    assert out_yaml.exists()

    with open(out_yaml, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert loaded["compression_pipeline"][0]["method"] == "smoothquant"
    assert loaded["compression_pipeline"][0]["scheme"] == "W8A8"
    assert loaded["compression_pipeline"][0]["calibration"]["num_samples"] == 128


def test_export_from_manifest(tmp_path: Path):
    """Test exporting from an existing ViForge experiment YAML manifest."""
    manifest_path = Path("configs/experiments/exp_003_deepseek_v4_pro_cpt_sft_dpo.yaml")
    assert manifest_path.exists(), "Sample experiment manifest must exist for testing"

    out_yaml = tmp_path / "vipym_manifest_export.yaml"
    res_path = ViPymExporter.export_from_manifest(
        manifest_or_path=manifest_path,
        recipe=CompressionRecipe.SPINQUANT,
        output_yaml_path=out_yaml,
    )

    assert res_path == out_yaml
    assert out_yaml.exists()

    with open(out_yaml, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    assert loaded["compression_pipeline"][0]["method"] == "spinquant"
    assert "humaneval" in loaded["evaluation"]["suites"]


def test_vipym_runner_mock(tmp_path: Path):
    """Test ViPymRunner executing in mock simulation mode."""
    cfg_path = tmp_path / "test_vipym.yaml"
    ViPymExporter.export_from_model_dir(
        model_dir="mock/model",
        recipe=CompressionRecipe.AUTOROUND_W4A16,
        output_yaml_path=cfg_path,
    )

    work_dir = tmp_path / "runs_vipym"
    result = ViPymRunner.run_compression(
        vipym_config_path=cfg_path,
        work_dir=work_dir,
        mock=True,
    )

    assert result["status"] == "Completed"
    assert result["is_mock"] is True
    assert result["method"] == "autoround"
    assert result["scheme"] == "W4A16"
    assert result["size_reduction_pct"] > 0
    assert result["latency_speedup"] > 1.0
    assert (Path(result["output_dir"]) / "compression_summary.json").exists()


def test_is_vipym_available():
    """Verify that is_vipym_available returns boolean without error."""
    avail = is_vipym_available()
    assert isinstance(avail, bool)


def test_cli_export_vipym(cli_runner, tmp_path: Path):
    """Test CLI export-vipym command end-to-end."""
    out_yaml = tmp_path / "cli_vipym.yaml"
    result = cli_runner.invoke(
        app,
        [
            "export-vipym",
            "configs/experiments/exp_003_deepseek_v4_pro_cpt_sft_dpo.yaml",
            "--recipe",
            "smoothquant_w8a8",
            "--output",
            str(out_yaml),
        ],
    )
    assert result.exit_code == 0
    assert out_yaml.exists()
    assert "exported successfully" in result.output


def test_cli_compress_vipym_mock(cli_runner, tmp_path: Path):
    """Test CLI compress-vipym command in mock simulation mode."""
    cfg_path = tmp_path / "cli_test_vipym.yaml"
    ViPymExporter.export_from_model_dir(
        model_dir="dummy/model",
        recipe=CompressionRecipe.SMOOTHQUANT_W8A8,
        output_yaml_path=cfg_path,
    )

    result = cli_runner.invoke(
        app,
        [
            "compress-vipym",
            str(cfg_path),
            "--work-dir",
            str(tmp_path / "runs"),
            "--mock",
        ],
    )
    assert result.exit_code == 0
    assert "Completed Successfully" in result.output
    assert "smoothquant" in result.output
