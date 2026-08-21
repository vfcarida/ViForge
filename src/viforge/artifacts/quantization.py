"""
ViForge Post-Training Quantization Engine (AWQ, GPTQ).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from viforge.utils.logging import logger


class AWQQuantizer:
    """
    Activation-aware Weight Quantization (AWQ) for high-performance low-VRAM serving.
    Integrates with AutoAWQ when installed on CUDA environments, and provides
    defensive metadata exports for offline/testing environments.
    """

    SUPPORTED_BITS = [4, 8]
    SUPPORTED_GROUP_SIZES = [32, 64, 128]

    @classmethod
    def is_autoawq_available(cls) -> bool:
        """Check if AutoAWQ library is installed and importable."""
        try:
            import awq  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def quantize(
        cls,
        model_path: Path,
        output_dir: Path,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        calibration_dataset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs AWQ calibration on model weights and exports quantized safetensors.
        """
        if bits not in cls.SUPPORTED_BITS:
            raise ValueError(f"Unsupported bit precision: {bits}. Supported: {cls.SUPPORTED_BITS}")
        if group_size not in cls.SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsupported group size: {group_size}. Supported: {cls.SUPPORTED_GROUP_SIZES}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / "quant_config.json"
        weights_file = output_dir / "model.safetensors"

        quant_metadata = {
            "quant_method": "awq",
            "bits": bits,
            "group_size": group_size,
            "zero_point": zero_point,
            "version": "0.2.0",
            "modules_to_not_convert": ["lm_head"],
            "source_model": str(model_path),
        }

        is_real = False
        if cls.is_autoawq_available():
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info(f"AutoAWQ: initiating {bits}-bit calibration on GPU...")
                    from awq import AutoAWQForCausalLM
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        str(model_path), trust_remote_code=True
                    )
                    model = AutoAWQForCausalLM.from_pretrained(
                        str(model_path), low_cpu_mem_usage=True, trust_remote_code=True
                    )

                    quant_config = {
                        "zero_point": zero_point,
                        "q_group_size": group_size,
                        "w_bit": bits,
                        "version": "GEMM",
                    }
                    calib_data = calibration_dataset or [
                        "def solve(n: int) -> int:\n    return n * (n + 1) // 2\n",
                        "class Node:\n    def __init__(self, val=0):\n        self.val = val\n",
                    ]
                    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
                    model.save_quantized(str(output_dir))
                    tokenizer.save_pretrained(str(output_dir))
                    is_real = True
                    logger.info(
                        f"AutoAWQ: successfully exported {bits}-bit weights to {output_dir}"
                    )
            except Exception as e:
                logger.warning(
                    f"AutoAWQ execution encountered an error: {e}. Falling back to metadata stub."
                )

        if not is_real:
            # Write standard AWQ config metadata
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(quant_metadata, f, indent=2)

            with open(weights_file, "w", encoding="utf-8") as f:
                f.write(f"AWQ_{bits}BIT_WEIGHTS_MOCK:{json.dumps(quant_metadata)}")

            logger.info(
                f"AWQ {bits}-bit metadata export completed at {output_dir} (is_real_quantization=False)"
            )

        return {
            "output_dir": str(output_dir),
            "config_path": str(config_file),
            "weights_path": str(weights_file),
            "quant_metadata": quant_metadata,
            "bits": bits,
            "group_size": group_size,
            "is_real_quantization": is_real,
        }
