"""
Unit and real training tests for ViForge Specialization Methods (LoRA, QLoRA, SFT, DPO, GRPO, CPT).
"""

from pathlib import Path

import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from viforge.config.schemas import (
    DatasetReference,
    HyperparametersConfig,
    TrainingMethodType,
    TrainingStageConfig,
)
from viforge.methods.base import method_registry
from viforge.methods.cpt import CPTMethod
from viforge.methods.peft_lora import LoRAMethod, QLoRAMethod
from viforge.methods.preference import DPOMethod, GRPOMethod
from viforge.methods.sft import SFTMethod


@pytest.mark.unit
def test_all_six_methods_registered():
    methods = method_registry.list_all()
    for required in ["lora", "qlora", "sft", "dpo", "grpo", "cpt"]:
        assert required in methods
        inst = method_registry.get(required)
        assert inst.method_name == required
        assert inst.__doc__ is not None
        assert "Expected Dataset Format" in inst.__doc__


@pytest.mark.unit
def test_mock_execution_paths(tmp_path: Path):
    stage_cfg = TrainingStageConfig(
        stage_id="stage_001",
        method=TrainingMethodType.LORA,
        dataset=DatasetReference(id="ds_001", path=str(tmp_path / "data.parquet")),
        hyperparameters=HyperparametersConfig(
            learning_rate=2e-4,
            num_epochs=1,
            per_device_batch_size=2,
            max_seq_len=128,
            lora_rank=8,
            lora_alpha=16,
        ),
    )

    methods = [
        LoRAMethod(),
        QLoRAMethod(),
        SFTMethod(),
        DPOMethod(),
        GRPOMethod(),
        CPTMethod(),
    ]
    for method in methods:
        metrics = method.execute_stage(
            model=None,
            tokenizer=None,
            train_data_path=tmp_path / "train.parquet",
            eval_data_path=None,
            stage_config=stage_cfg,
            output_dir=tmp_path / "checkpoints",
        )
        assert metrics.stage_id == "stage_001"
        assert metrics.training_loss > 0
        assert metrics.tokens_processed > 0
        assert metrics.wall_clock_seconds > 0
        assert metrics.trainable_parameters > 0
        assert metrics.total_parameters > 0


@pytest.mark.gpu
@pytest.mark.slow
def test_lora_real_training(tmp_path: Path):
    """Train LoRA on lightweight in-memory GPT2 configuration for real execution validation."""
    # 1. Instantiate tiny in-memory model & tokenizer
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 64
    config.vocab_size = 1000
    model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure training stage
    stage_cfg = TrainingStageConfig(
        stage_id="lora_test_stage",
        method=TrainingMethodType.LORA,
        dataset=DatasetReference(id="ds_train", path=str(tmp_path / "train.json")),
        hyperparameters=HyperparametersConfig(
            learning_rate=5e-4,
            num_epochs=2,
            per_device_batch_size=2,
            max_seq_len=128,
            lora_rank=4,
            lora_alpha=8,
            gradient_checkpointing=False,
        ),
    )

    # 3. Create dummy dataset file
    dataset_file = tmp_path / "train.json"
    dataset_file.write_text(
        '[\n  {"text": "def add(a, b): return a + b\\n"},\n  {"text": "def sub(a, b): return a - b\\n"}\n]',
        encoding="utf-8",
    )

    # 4. Execute stage with real model
    lora_method = LoRAMethod()
    metrics = lora_method.execute_stage(
        model=model,
        tokenizer=tokenizer,
        train_data_path=dataset_file,
        eval_data_path=None,
        stage_config=stage_cfg,
        output_dir=tmp_path / "output",
    )

    # 5. Verify metrics and artifact creation
    assert metrics.stage_id == "lora_test_stage"
    assert metrics.method == "lora"
    assert metrics.training_loss > 0
    assert metrics.trainable_parameters > 0
    assert metrics.trainable_ratio_pct < 100.0

    adapter_dir = tmp_path / "output" / "lora_test_stage" / "adapter"
    assert adapter_dir.exists()
    assert (adapter_dir / "adapter_config.json").exists()
