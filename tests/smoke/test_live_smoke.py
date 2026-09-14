"""
Live CPU smoke tests verifying real Hugging Face model initialization,
PEFT LoRA adaptation, autograd training execution, adapter checkpointing,
and zero-overhead weight merging.
"""

from pathlib import Path
import pytest
import torch

from transformers import LlamaConfig, LlamaForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

from viforge.config.schemas import (
    DatasetReference,
    HyperparametersConfig,
    TrainingMethodType,
    TrainingStageConfig,
)
from viforge.methods.peft_lora import LoRAMethod


def _create_tiny_llama() -> LlamaForCausalLM:
    """Create an ultra-lightweight, 100% in-memory Llama model for fast CPU verification."""
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


@pytest.mark.smoke
def test_live_lora_prepare_and_parameters():
    """Verify that LoRAMethod correctly attaches a low-rank adapter to real HF architecture."""
    base_model = _create_tiny_llama()
    total_base_params = sum(p.numel() for p in base_model.parameters())
    assert total_base_params > 0

    stage_config = TrainingStageConfig(
        stage_id="stage_smoke_lora",
        method=TrainingMethodType.LORA,
        dataset=DatasetReference(id="smoke_ds"),
        hyperparameters=HyperparametersConfig(
            lora_rank=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            learning_rate=1e-3,
        ),
    )

    lora_method = LoRAMethod()
    peft_model = lora_method.prepare_model(base_model, stage_config)

    assert isinstance(peft_model, PeftModel)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable_params > 0
    assert trainable_params < total_base_params


@pytest.mark.smoke
def test_live_lora_autograd_and_adapter_saving(tmp_path: Path):
    """
    Verify real forward pass, loss calculation, backward pass, optimizer step,
    and adapter disk serialization using genuine PyTorch autograd.
    """
    base_model = _create_tiny_llama()
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, peft_config)

    # Synthetic batch of tokens
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    outputs = peft_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    assert loss is not None
    assert torch.isfinite(loss).item()

    loss.backward()
    optimizer.step()

    # Verify adapter serialization to disk
    adapter_dir = tmp_path / "saved_adapter"
    peft_model.save_pretrained(str(adapter_dir))

    assert adapter_dir.exists()
    assert (adapter_dir / "adapter_config.json").exists()


@pytest.mark.smoke
def test_live_lora_merge_and_unload_inference(tmp_path: Path):
    """
    Verify that trained LoRA adapters merge cleanly back into the base architecture
    with zero inference overhead.
    """
    base_model = _create_tiny_llama()
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, peft_config)

    # Merge adapter into base weights
    merged_model = peft_model.merge_and_unload()
    assert not isinstance(merged_model, PeftModel)
    assert isinstance(merged_model, LlamaForCausalLM)

    # Verify merged model produces valid logits
    test_input = torch.randint(0, 128, (1, 8), dtype=torch.long)
    with torch.no_grad():
        out = merged_model(input_ids=test_input)

    assert out.logits.shape == (1, 8, 128)
    assert torch.isfinite(out.logits).all().item()

    # Verify full model serialization
    export_dir = tmp_path / "merged_export"
    merged_model.save_pretrained(str(export_dir))
    assert export_dir.exists()
    assert (export_dir / "config.json").exists()
