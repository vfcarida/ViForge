"""
Unit tests for ViForge Pydantic v2 schemas and validation contracts.
"""

from pathlib import Path
import pytest
from viforge.config.schemas import (
    ModelConfig,
    ModelType,
    ExperimentManifest,
    TrainingStageConfig,
    TrainingMethodType,
    DatasetReference,
    HyperparametersConfig,
    QuantizationType,
    HardwareConfig,
    EvaluationConfig,
    BenchmarkItemConfig,
)


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
        evaluation=EvaluationConfig(
            domain_benchmarks=[BenchmarkItemConfig(name="humaneval_plus")]
        ),
    )

    data = manifest.model_dump()
    reconstructed = ExperimentManifest.model_validate(data)
    assert reconstructed.experiment_id == "exp_test_001"
    assert reconstructed.pipeline[0].hyperparameters.quantization == QuantizationType.NF4
