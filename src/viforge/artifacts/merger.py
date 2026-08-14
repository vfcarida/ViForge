"""
ViForge Adapter Merger and Standalone Model Exporter.
"""

from pathlib import Path
from viforge.utils.logging import logger


class AdapterMerger:
    """Merges trained LoRA/PEFT adapters into base model weights for zero-overhead inference."""

    @classmethod
    def merge_and_export(
        cls,
        base_model_id: str,
        adapter_path: Path,
        output_dir: Path,
        device: str = "cpu",
        safe_serialization: bool = True,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError:
            logger.warning("PyTorch/Transformers not available for merging. Skipping live export.")
            return output_dir

        logger.info(f"Loading base model '{base_model_id}' on device '{device}'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        logger.info(f"Loading LoRA adapter from '{adapter_path}'...")
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

        logger.info("Executing merge_and_unload()...")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged standalone model to '{output_dir}'...")
        merged_model.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
        tokenizer.save_pretrained(str(output_dir))

        logger.info("Adapter merging completed successfully.")
        return output_dir
