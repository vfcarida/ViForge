"""
Unit tests for ViForge Pre-flight Resource and Memory Profiler.
"""

import pytest
from viforge.config.schemas import (
    ModelConfig,
    HyperparametersConfig,
    HardwareConfig,
    QuantizationType,
    CostTrackingConfig,
)
from viforge.training.profiler import ResourceProfiler
from viforge.training.cost_estimator import TrainingCostEstimator


def test_qlora_memory_profiling():
    model_conf = ModelConfig(
        name="DeepSeek V4 Pro",
        hf_hub_id="deepseek-ai/deepseek-v4-pro",
        total_parameters=14.5,
    )
    hyperparams = HyperparametersConfig(
        quantization=QuantizationType.NF4,
        lora_rank=64,
        per_device_batch_size=4,
        max_seq_len=4096,
        gradient_checkpointing=True,
    )
    hardware = HardwareConfig(accelerator="NVIDIA_A100_80GB", num_gpus=1, vram_per_gpu_gb=80.0)

    profile = ResourceProfiler.validate_preflight(model_conf, hyperparams, hardware)
    assert profile["total_estimated_vram_gb"] < 40.0
    assert profile["utilization_pct"] < 50.0


def test_memory_exceeded_fail_fast():
    model_conf = ModelConfig(
        name="DeepSeek V4 Pro",
        hf_hub_id="deepseek-ai/deepseek-v4-pro",
        total_parameters=14.5,
    )
    hyperparams = HyperparametersConfig(
        quantization=QuantizationType.NONE,
        lora_rank=0,
        per_device_batch_size=16,
        max_seq_len=8192,
        gradient_checkpointing=False,
    )
    hardware = HardwareConfig(accelerator="NVIDIA_A10G", num_gpus=1, vram_per_gpu_gb=24.0)

    with pytest.raises(RuntimeError) as exc_info:
        ResourceProfiler.validate_preflight(model_conf, hyperparams, hardware)

    assert "Pre-flight Resource Validation FAILED" in str(exc_info.value)
    assert "remediation" in str(exc_info.value)


def test_cost_estimator():
    model_conf = ModelConfig(
        name="DeepSeek V4 Pro",
        hf_hub_id="deepseek-ai/deepseek-v4-pro",
        total_parameters=14.5,
    )
    hyperparams = HyperparametersConfig(quantization=QuantizationType.NF4, num_epochs=1)
    hardware = HardwareConfig(num_gpus=1, vram_per_gpu_gb=80.0)
    cost_cfg = CostTrackingConfig(ec2_hourly_rate_usd=7.40)

    cost_profile = TrainingCostEstimator.estimate_cost(
        model_config=model_conf,
        hyperparams=hyperparams,
        hardware=hardware,
        cost_config=cost_cfg,
        total_tokens=10_000_000,
    )
    assert cost_profile["total_petaflops"] > 0
    assert cost_profile["estimated_cost_usd"] > 0
