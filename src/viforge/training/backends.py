"""
ViForge Compute Acceleration Backends (Accelerate, FSDP, DeepSpeed).
"""

from typing import Any, Dict
from viforge.config.schemas import HardwareConfig
from viforge.utils.logging import logger


class BackendManager:
    """Configures distributed training runtimes (Accelerate, FSDP, DeepSpeed)."""

    @classmethod
    def setup_backend(cls, hardware: HardwareConfig) -> Dict[str, Any]:
        backend_info = {
            "accelerator": hardware.accelerator,
            "num_gpus": hardware.num_gpus,
            "compute_dtype": hardware.compute_dtype.value,
            "flash_attention_2": hardware.enable_flash_attention_2,
        }

        if hardware.fsdp_config:
            backend_info["mode"] = "fsdp"
            backend_info["fsdp_config"] = hardware.fsdp_config
            logger.info("Configured PyTorch FSDP distributed backend.")
        elif hardware.deepspeed_config:
            backend_info["mode"] = "deepspeed"
            backend_info["deepspeed_config"] = hardware.deepspeed_config
            logger.info("Configured DeepSpeed ZeRO backend.")
        else:
            backend_info["mode"] = "standard_accelerate"
            logger.info("Configured Accelerate standard backend.")

        return backend_info
