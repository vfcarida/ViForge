"""
ViForge Pre-flight Resource and Memory Profiler.
Validates GPU VRAM, CPU RAM, disk space, and sequence length feasibility before allocation.
"""

from typing import Any, Dict, Optional, Tuple
from viforge.config.schemas import (
    ModelConfig,
    HyperparametersConfig,
    HardwareConfig,
    QuantizationType,
)
from viforge.utils.logging import logger


class ResourceProfiler:
    """Pre-flight analytical resource and memory profiler."""

    SAFETY_MARGIN: float = 0.85

    @classmethod
    def estimate_trainable_parameters(
        cls, model_config: ModelConfig, hyperparams: HyperparametersConfig
    ) -> Tuple[int, int, float]:
        total_params = int(model_config.total_parameters * 1e9)

        if (
            hyperparams.quantization
            in [QuantizationType.NF4, QuantizationType.FP4, QuantizationType.INT8]
            or hyperparams.lora_rank > 0
        ):
            num_targets = len(hyperparams.target_modules or model_config.target_modules)
            approx_hidden_dim = (
                4096 if total_params < 10e9 else 5120 if total_params < 20e9 else 8192
            )
            approx_layers = 32 if total_params < 10e9 else 48 if total_params < 20e9 else 64

            lora_params = int(
                2 * approx_hidden_dim * hyperparams.lora_rank * num_targets * approx_layers
            )
            trainable_params = min(lora_params, int(total_params * 0.05))
        else:
            trainable_params = total_params

        trainable_ratio = (trainable_params / total_params) * 100.0
        return total_params, trainable_params, trainable_ratio

    @classmethod
    def profile_vram_gb(
        cls,
        model_config: ModelConfig,
        hyperparams: HyperparametersConfig,
        hardware: HardwareConfig,
        method: Optional[Any] = None,
    ) -> Dict[str, float]:
        total_params, trainable_params, _ = cls.estimate_trainable_parameters(
            model_config, hyperparams
        )

        # 1. Base weights
        if hyperparams.quantization in [QuantizationType.NF4, QuantizationType.FP4]:
            bytes_per_base_param = 0.55
        elif hyperparams.quantization == QuantizationType.INT8:
            bytes_per_base_param = 1.05
        elif hardware.compute_dtype.value in ["bfloat16", "float16"]:
            bytes_per_base_param = 2.0
        else:
            bytes_per_base_param = 4.0

        frozen_params = total_params - trainable_params
        weight_mem_bytes = (frozen_params * bytes_per_base_param) + (trainable_params * 2.0)
        weight_mem_gb = weight_mem_bytes / (1024**3)

        # Reference model overhead (Standard DPO requires frozen reference policy in memory;
        # Reference-free methods like ORPO, SimPO, KTO, and SFT require 0.0 GB reference overhead)
        if method is not None and hasattr(method, "value"):
            method_str = str(method.value).lower()
        else:
            method_str = str(method or "").lower()
        if method_str == "dpo":
            ref_model_gb = weight_mem_gb
        else:
            ref_model_gb = 0.0

        # 2. Gradients (2 bytes per trainable param in half precision)
        grad_mem_gb = (trainable_params * 2.0) / (1024**3)

        # 3. Optimizer state
        opt_bytes = 2.0 if "8bit" in hyperparams.optimizer.lower() else 8.0
        opt_mem_gb = (trainable_params * opt_bytes) / (1024**3)

        # 4. Activations
        batch_size = hyperparams.per_device_batch_size
        seq_len = hyperparams.max_seq_len
        hidden_dim = 4096 if total_params < 10e9 else 5120 if total_params < 20e9 else 8192
        num_layers = 32 if total_params < 10e9 else 48 if total_params < 20e9 else 64

        if hyperparams.gradient_checkpointing:
            act_mem_bytes = (
                2 * batch_size * seq_len * hidden_dim * num_layers * 2.0 / hardware.num_gpus
            )
        else:
            act_mem_bytes = (
                10 * batch_size * seq_len * hidden_dim * num_layers * 2.0 / hardware.num_gpus
            )

        act_mem_gb = max(0.5, act_mem_bytes / (1024**3))
        overhead_gb = 1.5
        total_vram_gb = (
            weight_mem_gb + ref_model_gb + grad_mem_gb + opt_mem_gb + act_mem_gb + overhead_gb
        )

        return {
            "weight_memory_gb": round(weight_mem_gb, 2),
            "ref_model_memory_gb": round(ref_model_gb, 2),
            "gradient_memory_gb": round(grad_mem_gb, 2),
            "optimizer_memory_gb": round(opt_mem_gb, 2),
            "activation_memory_gb": round(act_mem_gb, 2),
            "overhead_gb": round(overhead_gb, 2),
            "total_estimated_vram_gb": round(total_vram_gb, 2),
            "available_vram_per_gpu_gb": round(hardware.vram_per_gpu_gb, 2),
            "utilization_pct": round((total_vram_gb / hardware.vram_per_gpu_gb) * 100.0, 1),
        }

    @classmethod
    def validate_preflight(
        cls,
        model_config: ModelConfig,
        hyperparams: HyperparametersConfig,
        hardware: HardwareConfig,
        method: Optional[Any] = None,
    ) -> Dict[str, float]:
        profile = cls.profile_vram_gb(model_config, hyperparams, hardware, method=method)
        total_est = profile["total_estimated_vram_gb"]
        max_allowed = hardware.vram_per_gpu_gb * cls.SAFETY_MARGIN

        if total_est > max_allowed:
            rec = []
            if hyperparams.quantization == QuantizationType.NONE:
                rec.append("Enable QLoRA 4-bit quantization ('nf4')")
            if hyperparams.per_device_batch_size > 1:
                rec.append(
                    f"Reduce per_device_batch_size from {hyperparams.per_device_batch_size} to 1 or 2"
                )
            if not hyperparams.gradient_checkpointing:
                rec.append("Enable gradient_checkpointing = True")
            if "8bit" not in hyperparams.optimizer:
                rec.append("Use 8-bit paged optimizer ('paged_adamw_8bit')")
            if hyperparams.max_seq_len > 2048:
                rec.append(f"Reduce max_seq_len from {hyperparams.max_seq_len} to 2048")

            recommendation_msg = (
                "; ".join(rec)
                if rec
                else "Allocate GPUs with larger VRAM (e.g. A100 80GB or H100)."
            )
            raise RuntimeError(
                f"Pre-flight Resource Validation FAILED: required {total_est:.2f} GB, available {hardware.vram_per_gpu_gb:.2f} GB. "
                f"Actionable remediation: {recommendation_msg}"
            )

        logger.info(
            f"Pre-flight resource validation PASSED: {total_est:.1f} GB / {hardware.vram_per_gpu_gb:.1f} GB "
            f"({profile['utilization_pct']}% utilization per GPU)."
        )
        return profile

    @classmethod
    def profile_distributed_vram(
        cls,
        model_config: ModelConfig,
        hyperparams: HyperparametersConfig,
        hardware: HardwareConfig,
        strategy: str = "zero3",
        cpu_offload: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculates per-GPU VRAM consumption across distributed training strategies:
        - ddp: DistributedDataParallel (replicated weights, grads, optimizer)
        - zero1: ZeRO-1 (optimizer partitioned across GPUs)
        - zero2: ZeRO-2 (optimizer + gradients partitioned across GPUs)
        - zero3 / fsdp2: Full parameter, gradient, and optimizer state sharding
        - Supports cpu_offload for parameters and optimizer states in ZeRO-3.
        """
        single_prof = cls.profile_vram_gb(model_config, hyperparams, hardware)
        num_gpus = max(1, hardware.num_gpus)
        strat = strategy.lower().strip()

        weight_mem = single_prof["weight_memory_gb"]
        grad_mem = single_prof["gradient_memory_gb"]
        opt_mem = single_prof["optimizer_memory_gb"]
        act_mem = single_prof["activation_memory_gb"]
        overhead = single_prof["overhead_gb"]

        approx_layers = 32 if (model_config.total_parameters * 1e9) < 10e9 else 48

        cpu_ram_needed_gb = 0.0

        if strat == "ddp":
            w_per_gpu = weight_mem
            g_per_gpu = grad_mem
            o_per_gpu = opt_mem
        elif strat == "zero1":
            w_per_gpu = weight_mem
            g_per_gpu = grad_mem
            o_per_gpu = opt_mem / num_gpus
        elif strat == "zero2":
            w_per_gpu = weight_mem
            g_per_gpu = grad_mem / num_gpus
            o_per_gpu = opt_mem / num_gpus
        elif strat in ("zero3", "fsdp2", "fsdp"):
            if cpu_offload:
                w_per_gpu = (weight_mem / approx_layers) + 0.4
                g_per_gpu = grad_mem / num_gpus
                o_per_gpu = 0.0
                cpu_ram_needed_gb = round(opt_mem + weight_mem * 1.1, 2)
            else:
                w_per_gpu = weight_mem / num_gpus
                g_per_gpu = grad_mem / num_gpus
                o_per_gpu = opt_mem / num_gpus
        else:
            raise ValueError(f"Unknown distributed strategy '{strategy}'. Valid: ddp, zero1, zero2, zero3, fsdp2")

        total_vram_per_gpu = w_per_gpu + g_per_gpu + o_per_gpu + act_mem + overhead
        util_pct = (total_vram_per_gpu / hardware.vram_per_gpu_gb) * 100.0

        return {
            "strategy": strat,
            "num_gpus": num_gpus,
            "cpu_offload": cpu_offload,
            "weight_memory_per_gpu_gb": round(w_per_gpu, 2),
            "gradient_memory_per_gpu_gb": round(g_per_gpu, 2),
            "optimizer_memory_per_gpu_gb": round(o_per_gpu, 2),
            "activation_memory_per_gpu_gb": round(act_mem, 2),
            "overhead_gb": round(overhead, 2),
            "total_estimated_vram_per_gpu_gb": round(total_vram_per_gpu, 2),
            "available_vram_per_gpu_gb": round(hardware.vram_per_gpu_gb, 2),
            "utilization_pct": round(util_pct, 1),
            "cpu_ram_required_gb": cpu_ram_needed_gb,
            "fits_in_vram": total_vram_per_gpu <= hardware.vram_per_gpu_gb * cls.SAFETY_MARGIN,
        }

    @classmethod
    def recommend_distributed_strategy(
        cls,
        model_config: ModelConfig,
        hyperparams: HyperparametersConfig,
        hardware: HardwareConfig,
    ) -> Dict[str, Any]:
        """
        Determines the optimal distributed training strategy with minimal inter-node communication overhead.
        """
        strategies_to_test = [
            ("ddp", False),
            ("zero1", False),
            ("zero2", False),
            ("fsdp2", False),
            ("zero3", True),
        ]

        profiles = []
        best_strategy = None
        best_profile = None

        for strat, offload in strategies_to_test:
            prof = cls.profile_distributed_vram(
                model_config, hyperparams, hardware, strategy=strat, cpu_offload=offload
            )
            profiles.append(prof)
            if prof["fits_in_vram"] and best_strategy is None:
                best_strategy = f"{strat}_offload" if offload else strat
                best_profile = prof

        if best_strategy is None:
            best_strategy = "insufficient_vram"
            rationale = (
                f"Model ({model_config.total_parameters}B) requires more VRAM than cluster provides "
                f"({hardware.num_gpus}x {hardware.vram_per_gpu_gb}GB). Consider QLoRA 4-bit, reducing context length, "
                f"or scaling to 8x A100/H100 nodes."
            )
        else:
            reasons = {
                "ddp": "Model fits within single GPU memory; standard DDP provides maximum throughput and minimal communication.",
                "zero1": "Optimizer states partitioned across GPUs; avoids sharding weights while fitting comfortably.",
                "zero2": "Optimizer and gradient states partitioned; balances communication efficiency with substantial memory savings.",
                "fsdp2": "Full parameter sharding (FSDP2) eliminates memory duplication; optimal for multi-billion parameter architectures.",
                "zero3_offload": "ZeRO-3 with CPU offloading required to fit weights/optimizer within available GPU VRAM.",
            }
            rationale = reasons.get(best_strategy, "Recommended for optimal memory and throughput trade-off.")

        return {
            "recommended_strategy": best_strategy,
            "rationale": rationale,
            "num_gpus": hardware.num_gpus,
            "best_profile": best_profile,
            "all_profiles": {p["strategy"] + ("_offload" if p["cpu_offload"] else ""): p for p in profiles},
        }

