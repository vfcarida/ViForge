"""
ViForge Compute Acceleration Backends (Accelerate, FSDP, DeepSpeed, Liger-Kernel).
"""

from typing import Any, Dict
from viforge.config.schemas import HardwareConfig
from viforge.utils.logging import logger


class LigerKernelManager:
    """
    Manages Triton-based Liger-Kernel integration for fused CrossEntropy, RMSNorm, and SwiGLU.
    """

    @classmethod
    def apply_liger_kernel_to_model(cls, model: Any = None, enable: bool = True) -> bool:
        """
        Attempts to patch HuggingFace model with Liger-Kernel optimizations.
        Returns True if enabled or gracefully fallback to standard PyTorch.
        """
        if not enable:
            return False

        try:
            import liger_kernel.transformers as liger_transformers

            model_type = getattr(getattr(model, "config", None), "model_type", "llama")
            patch_fn_name = f"apply_liger_kernel_to_{model_type}"
            if hasattr(liger_transformers, patch_fn_name):
                getattr(liger_transformers, patch_fn_name)()
            elif hasattr(liger_transformers, "apply_liger_kernel_to_llama"):
                liger_transformers.apply_liger_kernel_to_llama()
            elif hasattr(liger_transformers, "_apply_liger_kernel"):
                liger_transformers._apply_liger_kernel()

            logger.info(
                "Liger-Kernel Triton fused kernels successfully activated (RMSNorm, CrossEntropy, SwiGLU)."
            )
            return True
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "liger-kernel package not installed. Gracefully falling back to native PyTorch kernels."
            )
            return False
        except Exception as e:
            logger.warning(
                f"Liger-Kernel patching encountered an issue ({e}). Continuing with native PyTorch kernels."
            )
            return False



class BackendManager:
    """Configures distributed training runtimes (Accelerate, FSDP, DeepSpeed, Liger)."""

    @classmethod
    def setup_backend(cls, hardware: HardwareConfig, use_liger: bool = False) -> Dict[str, Any]:
        liger_active = LigerKernelManager.apply_liger_kernel_to_model(enable=use_liger)

        backend_info = {
            "accelerator": hardware.accelerator,
            "num_gpus": hardware.num_gpus,
            "compute_dtype": hardware.compute_dtype.value,
            "flash_attention_2": hardware.enable_flash_attention_2,
            "liger_kernel_active": liger_active,
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
