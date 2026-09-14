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
from viforge.methods.preference import DPOMethod, GRPOMethod, KTOMethod
from viforge.methods.sft import SFTMethod


@pytest.mark.unit
def test_all_seven_methods_registered():
    methods = method_registry.list_all()
    for required in ["lora", "qlora", "sft", "dpo", "kto", "grpo", "cpt"]:
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
        KTOMethod(),
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


@pytest.mark.unit
def test_lora_real_autograd_gradient_flow():
    """Verify real autograd backward pass and parameter updates on LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model

    # 1. Tiny 2-layer causal language model in memory
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 64
    config.vocab_size = 500
    base_model = AutoModelForCausalLM.from_config(config)

    # 2. Attach LoRA adapter
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)

    # 3. Check parameter trainability
    trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
    frozen_params = [p for p in lora_model.parameters() if not p.requires_grad]
    assert len(trainable_params) > 0
    assert len(frozen_params) > 0

    # Capture initial weights of all LoRA parameters
    initial_weights = [p.clone().detach() for p in trainable_params]

    # 4. Forward pass
    input_ids = torch.randint(0, 500, (2, 16))
    outputs = lora_model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    assert loss is not None
    assert loss.item() > 0.0

    # 5. Backward pass
    loss.backward()

    # Verify gradients flowed into LoRA adapters (specifically lora_B, which has non-zero grad at step 0)
    total_grad = sum(torch.sum(torch.abs(p.grad)).item() for p in trainable_params if p.grad is not None)
    assert total_grad > 0.0, "LoRA parameters must receive non-zero gradients"

    # 6. Optimizer step and parameter delta check
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-2)
    optimizer.step()

    total_delta = sum(
        torch.sum(torch.abs(p - init_w)).item()
        for p, init_w in zip(trainable_params, initial_weights)
    )
    assert total_delta > 0.0, "LoRA weights must be updated after optimizer step"


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
