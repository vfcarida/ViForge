"""
ViForge Adapter Merger and Standalone Model Exporter.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from viforge.utils.logging import logger


class AdapterMerger:
    """
    Merges trained LoRA/PEFT adapters into base model weights for zero-overhead inference.
    Supports single adapter merging and weighted linear combinations of multiple adapters.
    """

    @classmethod
    def merge_and_export(
        cls,
        base_model_id: str,
        adapter_path: Path,
        output_dir: Path,
        device: str = "cpu",
        safe_serialization: bool = True,
    ) -> Path:
        """Merge a single LoRA adapter into base model weights."""
        return cls.merge_multiple_adapters(
            base_model_id=base_model_id,
            adapters=[(adapter_path, 1.0)],
            output_dir=output_dir,
            device=device,
            safe_serialization=safe_serialization,
        )

    @classmethod
    def merge_multiple_adapters(
        cls,
        base_model_id: str,
        adapters: List[Tuple[Path, float]],
        output_dir: Path,
        device: str = "cpu",
        combination_type: str = "linear",
        safe_serialization: bool = True,
    ) -> Path:
        """
        Merge multiple LoRA adapters with linear blending weights into base model weights.

        Args:
            base_model_id: HuggingFace model ID or local directory.
            adapters: List of (adapter_path, weight) tuples.
            output_dir: Target export directory for merged safetensors.
            device: Computation device ("cpu", "cuda", "auto").
            combination_type: Merging strategy ("linear", "ties", "dare").
            safe_serialization: Whether to export as safetensors.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        if not adapters:
            raise ValueError("At least one adapter must be provided for merging.")

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning("PyTorch/Transformers not available for merging. Skipping live export.")
            return output_dir

        logger.info(f"Loading base model '{base_model_id}' on device '{device}'...")
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        if len(adapters) == 1:
            adapter_path, _ = adapters[0]
            logger.info(f"Loading single LoRA adapter from '{adapter_path}'...")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            adapter_names = []
            weights = []
            # Load first adapter as base PEFT wrapper
            first_path, first_weight = adapters[0]
            peft_model = PeftModel.from_pretrained(
                base_model, str(first_path), adapter_name="adapter_0"
            )
            adapter_names.append("adapter_0")
            weights.append(first_weight)

            # Load remaining adapters
            for i, (ad_path, w) in enumerate(adapters[1:], start=1):
                name = f"adapter_{i}"
                logger.info(f"Loading secondary adapter '{name}' from '{ad_path}' (weight={w})...")
                peft_model.load_adapter(str(ad_path), adapter_name=name)
                adapter_names.append(name)
                weights.append(w)

            logger.info(
                f"Combining adapters {adapter_names} with weights {weights} (combination={combination_type})..."
            )
            try:
                peft_model.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=weights,
                    adapter_name="composite_specialist",
                    combination_type=combination_type,
                )
                peft_model.set_adapter("composite_specialist")
            except Exception as e:
                logger.warning(
                    f"add_weighted_adapter failed: {e}. Falling back to standard merge of active adapter."
                )

        logger.info("Executing merge_and_unload()...")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged standalone model to '{output_dir}'...")
        merged_model.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
        tokenizer.save_pretrained(str(output_dir))

        logger.info("Adapter merging completed successfully.")
        return output_dir
