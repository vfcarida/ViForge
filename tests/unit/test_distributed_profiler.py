"""
Unit tests for Distributed Multi-GPU Profiling (FSDP2, ZeRO-1/2/3) and Config Generation.
"""

from pathlib import Path
import pytest

from viforge.config.schemas import (
    ComputeDtype,
    HardwareConfig,
    HyperparametersConfig,
    ModelConfig,
    QuantizationType,
)
from viforge.training.distributed import DistributedConfigGenerator
from viforge.training.profiler import ResourceProfiler


@pytest.fixture
def sample_model():
    return ModelConfig(
        name="DeepSeek-14B",
        hf_hub_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        total_parameters=14.0,
    )


@pytest.fixture
def sample_hyperparams():
    return HyperparametersConfig(
        learning_rate=2e-4,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_len=2048,
        gradient_checkpointing=True,
        quantization=QuantizationType.NONE,
        lora_rank=0,  # Full fine-tuning for maximum memory test
    )


@pytest.fixture
def sample_hardware():
    return HardwareConfig(
        accelerator="NVIDIA_A100_80GB",
        num_gpus=4,
        vram_per_gpu_gb=80.0,
        compute_dtype=ComputeDtype.BFLOAT16,
    )


@pytest.mark.unit
def test_zero_partitioning_math(sample_model, sample_hyperparams, sample_hardware):
    # Single GPU / DDP
    ddp_prof = ResourceProfiler.profile_distributed_vram(
        sample_model, sample_hyperparams, sample_hardware, strategy="ddp"
    )

    # ZeRO-1 (Optimizer partitioned)
    zero1_prof = ResourceProfiler.profile_distributed_vram(
        sample_model, sample_hyperparams, sample_hardware, strategy="zero1"
    )
    assert zero1_prof["optimizer_memory_per_gpu_gb"] < ddp_prof["optimizer_memory_per_gpu_gb"]
    assert zero1_prof["weight_memory_per_gpu_gb"] == ddp_prof["weight_memory_per_gpu_gb"]

    # ZeRO-2 (Optimizer + Grads partitioned)
    zero2_prof = ResourceProfiler.profile_distributed_vram(
        sample_model, sample_hyperparams, sample_hardware, strategy="zero2"
    )
    assert zero2_prof["gradient_memory_per_gpu_gb"] < zero1_prof["gradient_memory_per_gpu_gb"]

    # ZeRO-3 / FSDP2 (Full sharding)
    fsdp_prof = ResourceProfiler.profile_distributed_vram(
        sample_model, sample_hyperparams, sample_hardware, strategy="fsdp2"
    )
    assert fsdp_prof["weight_memory_per_gpu_gb"] < zero2_prof["weight_memory_per_gpu_gb"]
    assert fsdp_prof["total_estimated_vram_per_gpu_gb"] < zero2_prof["total_estimated_vram_per_gpu_gb"]


@pytest.mark.unit
def test_zero3_cpu_offload(sample_model, sample_hyperparams, sample_hardware):
    offload_prof = ResourceProfiler.profile_distributed_vram(
        sample_model, sample_hyperparams, sample_hardware, strategy="zero3", cpu_offload=True
    )
    assert offload_prof["optimizer_memory_per_gpu_gb"] == 0.0
    assert offload_prof["cpu_ram_required_gb"] > 0


@pytest.mark.unit
def test_recommend_distributed_strategy(sample_model, sample_hyperparams, sample_hardware):
    rec = ResourceProfiler.recommend_distributed_strategy(
        sample_model, sample_hyperparams, sample_hardware
    )
    assert "recommended_strategy" in rec
    assert rec["num_gpus"] == 4
    assert len(rec["all_profiles"]) >= 4


@pytest.mark.unit
def test_generate_accelerate_config(sample_hardware, tmp_path: Path):
    out_yaml = tmp_path / "accelerate_config.yaml"
    cfg = DistributedConfigGenerator.generate_accelerate_config(
        hardware=sample_hardware,
        strategy="fsdp2",
        output_path=out_yaml,
    )
    assert out_yaml.exists()
    assert cfg["distributed_type"] == "FSDP"
    assert cfg["num_processes"] == 4
    assert "fsdp_config" in cfg


@pytest.mark.unit
def test_generate_deepspeed_config(sample_hardware, sample_hyperparams, tmp_path: Path):
    out_json = tmp_path / "deepspeed_config.json"
    cfg = DistributedConfigGenerator.generate_deepspeed_config(
        hardware=sample_hardware,
        hyperparams=sample_hyperparams,
        zero_stage=3,
        cpu_offload=True,
        output_path=out_json,
    )
    assert out_json.exists()
    assert cfg["zero_optimization"]["stage"] == 3
    assert "offload_param" in cfg["zero_optimization"]
    assert cfg["zero_optimization"]["offload_param"]["device"] == "cpu"
