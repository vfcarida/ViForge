"""
Unit tests for ViForge Reference-Free Alignment Methods: ORPO and SimPO.
"""

from pathlib import Path
import pytest
import torch

from viforge.config.schemas import (
    DatasetReference,
    HardwareConfig,
    HyperparametersConfig,
    ModelConfig,
    TrainingMethodType,
    TrainingStageConfig,
)
from viforge.methods.base import method_registry
from viforge.methods.reference_free import (
    ORPOMethod,
    SimPOMethod,
    compute_orpo_loss,
    compute_simpo_loss,
)
from viforge.training.profiler import ResourceProfiler


def test_compute_orpo_loss_numerical_properties():
    """Verify mathematical properties and gradient flow of ORPO loss."""
    chosen_logps = torch.tensor([-1.5, -2.0], requires_grad=True)
    rejected_logps = torch.tensor([-3.5, -4.0], requires_grad=True)
    chosen_nll = torch.tensor([0.8, 1.2], requires_grad=True)

    loss, metrics = compute_orpo_loss(
        chosen_logps=chosen_logps,
        rejected_logps=rejected_logps,
        chosen_nll=chosen_nll,
        alpha=0.1,
    )

    assert loss.ndim == 0
    assert loss.item() > 0
    assert metrics["orpo_loss"] > 0
    assert metrics["odds_ratio_loss"] > 0
    assert metrics["log_odds_ratio_mean"] > 0

    # Test gradient backward
    loss.backward()
    assert chosen_logps.grad is not None
    assert rejected_logps.grad is not None
    assert chosen_nll.grad is not None

    # Verify preference monotonicity: if chosen has worse (lower) logp than rejected, loss is higher
    worse_chosen = torch.tensor([-5.0, -5.0])
    better_rejected = torch.tensor([-1.0, -1.0])
    worse_loss, _ = compute_orpo_loss(worse_chosen, better_rejected, chosen_nll.detach(), alpha=0.1)
    assert worse_loss.item() > loss.item()


def test_compute_simpo_loss_numerical_properties():
    """Verify length-normalization, target margin gamma, and gradients of SimPO loss."""
    chosen_logps = torch.tensor([-20.0, -25.0], requires_grad=True)
    rejected_logps = torch.tensor([-45.0, -50.0], requires_grad=True)
    chosen_len = torch.tensor([20.0, 25.0])
    rejected_len = torch.tensor([30.0, 35.0])

    loss, metrics = compute_simpo_loss(
        chosen_logps=chosen_logps,
        rejected_logps=rejected_logps,
        chosen_lengths=chosen_len,
        rejected_lengths=rejected_len,
        beta=2.0,
        gamma=0.5,
    )

    assert loss.ndim == 0
    assert loss.item() > 0
    assert metrics["simpo_loss"] > 0
    assert "reward_margin_mean" in metrics

    loss.backward()
    assert chosen_logps.grad is not None
    assert rejected_logps.grad is not None

    # Verify margin impact: higher target margin gamma leads to higher loss
    loss_high_gamma, _ = compute_simpo_loss(
        chosen_logps.detach(),
        rejected_logps.detach(),
        chosen_len,
        rejected_len,
        beta=2.0,
        gamma=2.0,
    )
    assert loss_high_gamma.item() > loss.item()


def test_orpo_method_registry_and_execution(tmp_path: Path):
    """Verify ORPO method resolution in registry and mock stage execution."""
    assert "orpo" in method_registry.list_all()
    method = method_registry.get("orpo")
    assert isinstance(method, ORPOMethod)
    assert method.method_name == "orpo"

    stage_cfg = TrainingStageConfig(
        stage_id="stage_orpo",
        method=TrainingMethodType.ORPO,
        dataset=DatasetReference(id="dummy_ds"),
        hyperparameters=HyperparametersConfig(orpo_alpha=0.15, learning_rate=1e-5),
    )

    metrics = method.execute_stage(
        model=None,
        tokenizer=None,
        train_data_path=tmp_path / "train.jsonl",
        eval_data_path=None,
        stage_config=stage_cfg,
        output_dir=tmp_path / "outputs",
    )

    assert metrics.method == "orpo"
    assert metrics.stage_id == "stage_orpo"
    assert metrics.training_loss > 0
    assert metrics.tokens_per_second > 0


def test_simpo_method_registry_and_execution(tmp_path: Path):
    """Verify SimPO method resolution in registry and mock stage execution."""
    assert "simpo" in method_registry.list_all()
    method = method_registry.get("simpo")
    assert isinstance(method, SimPOMethod)
    assert method.method_name == "simpo"

    stage_cfg = TrainingStageConfig(
        stage_id="stage_simpo",
        method=TrainingMethodType.SIMPO,
        dataset=DatasetReference(id="dummy_ds"),
        hyperparameters=HyperparametersConfig(beta=2.0, simpo_gamma=0.6, learning_rate=5e-6),
    )

    metrics = method.execute_stage(
        model=None,
        tokenizer=None,
        train_data_path=tmp_path / "train.jsonl",
        eval_data_path=None,
        stage_config=stage_cfg,
        output_dir=tmp_path / "outputs",
    )

    assert metrics.method == "simpo"
    assert metrics.stage_id == "stage_simpo"
    assert metrics.training_loss > 0
    assert metrics.tokens_per_second > 0


def test_resource_profiler_reference_free_vram_saving():
    """Verify that ORPO and SimPO require zero reference model overhead compared to DPO."""
    model_cfg = ModelConfig(
        name="TestCoder-7B",
        hf_hub_id="test/coder-7b",
        total_parameters=7.0,
    )
    hp_cfg = HyperparametersConfig(
        per_device_batch_size=2,
        max_seq_len=2048,
    )
    hw_cfg = HardwareConfig(
        num_gpus=1,
        vram_per_gpu_gb=80.0,
    )

    dpo_profile = ResourceProfiler.profile_vram_gb(
        model_config=model_cfg,
        hyperparams=hp_cfg,
        hardware=hw_cfg,
        method=TrainingMethodType.DPO,
    )

    orpo_profile = ResourceProfiler.profile_vram_gb(
        model_config=model_cfg,
        hyperparams=hp_cfg,
        hardware=hw_cfg,
        method=TrainingMethodType.ORPO,
    )

    simpo_profile = ResourceProfiler.profile_vram_gb(
        model_config=model_cfg,
        hyperparams=hp_cfg,
        hardware=hw_cfg,
        method=TrainingMethodType.SIMPO,
    )

    # Standard DPO requires reference model memory
    assert dpo_profile["ref_model_memory_gb"] > 0
    # Reference-free methods require zero reference model memory
    assert orpo_profile["ref_model_memory_gb"] == 0.0
    assert simpo_profile["ref_model_memory_gb"] == 0.0

    # Total estimated VRAM must be significantly lower for ORPO/SimPO than DPO
    assert orpo_profile["total_estimated_vram_gb"] < dpo_profile["total_estimated_vram_gb"]
    assert simpo_profile["total_estimated_vram_gb"] < dpo_profile["total_estimated_vram_gb"]
