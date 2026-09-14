"""
Unit tests for ViForge Pydantic v2 schemas and validation contracts.
"""

import pytest

from viforge.config.schemas import (
    BenchmarkItemConfig,
    DatasetReference,
    EvaluationConfig,
    ExperimentManifest,
    HardwareConfig,
    HyperparametersConfig,
    ModelConfig,
    ModelType,
    QuantizationType,
    TrainingMethodType,
    TrainingStageConfig,
)


@pytest.mark.unit
def test_model_config_validation():
    config = ModelConfig(
        name="DeepSeek V4 Pro",
        hf_hub_id="deepseek-ai/deepseek-v4-pro",
        total_parameters=14.5,
        expected_license="MIT",
    )
    assert config.model_type == ModelType.CAUSAL_LM
    assert config.pricing.prompt_usd_per_1m == 0.14
    assert "q_proj" in config.target_modules


@pytest.mark.unit
def test_experiment_manifest_roundtrip():
    manifest = ExperimentManifest(
        experiment_id="exp_test_001",
        model=ModelConfig(
            name="DeepSeek V4 Pro",
            hf_hub_id="deepseek-ai/deepseek-v4-pro",
            total_parameters=14.5,
            expected_license="MIT",
        ),
        hardware=HardwareConfig(num_gpus=1, vram_per_gpu_gb=80.0),
        pipeline=[
            TrainingStageConfig(
                stage_id="stage_1_qlora",
                method=TrainingMethodType.QLORA,
                dataset=DatasetReference(id="viforge-test-data", split="train"),
                hyperparameters=HyperparametersConfig(
                    quantization=QuantizationType.NF4,
                    lora_rank=64,
                ),
            )
        ],
        evaluation=EvaluationConfig(domain_benchmarks=[BenchmarkItemConfig(name="humaneval_plus")]),
    )

    data = manifest.model_dump()
    reconstructed = ExperimentManifest.model_validate(data)
    assert reconstructed.experiment_id == "exp_test_001"
    assert reconstructed.pipeline[0].hyperparameters.quantization == QuantizationType.NF4


@pytest.mark.unit
def test_model_registry_defaults():
    from viforge.models.registry import model_registry

    all_models = model_registry.list_all()
    hub_ids = {m.hf_hub_id for m in all_models}

    assert "deepseek-ai/deepseek-v4-pro" in hub_ids
    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in hub_ids
    assert "meta-llama/Llama-3.2-1B-Instruct" in hub_ids
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in hub_ids

    llama = model_registry.get_config("meta-llama/Llama-3.2-1B-Instruct")
    assert llama.total_parameters == 1.23
    assert llama.context_window == 131072
    assert "q_proj" in llama.target_modules

    smol = model_registry.get_config("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    assert smol.total_parameters == 1.71
    assert smol.expected_license == "Apache-2.0"

