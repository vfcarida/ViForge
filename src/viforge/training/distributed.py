"""
ViForge Distributed Training Configuration Generator (Accelerate, FSDP2, DeepSpeed).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from viforge.config.schemas import HardwareConfig, HyperparametersConfig, ModelConfig
from viforge.utils.logging import logger


class DistributedConfigGenerator:
    """
    Generates production configurations for multi-GPU training runners:
    - Hugging Face Accelerate (FSDP2, DeepSpeed ZeRO-3, DDP)
    - DeepSpeed JSON configurations (ZeRO-1, ZeRO-2, ZeRO-3)
    """

    @classmethod
    def generate_accelerate_config(
        cls,
        hardware: HardwareConfig,
        strategy: str = "fsdp2",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Generates standard accelerate_config.yaml dictionary for HuggingFace Accelerate launcher.
        """
        strat = strategy.lower().strip()
        num_gpus = max(1, hardware.num_gpus)
        dtype_str = hardware.compute_dtype.value if hasattr(hardware.compute_dtype, "value") else str(hardware.compute_dtype)

        config: Dict[str, Any] = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "MULTI_GPU" if strat == "ddp" else "FSDP" if "fsdp" in strat else "DEEPSPEED",
            "mixed_precision": "bf16" if "bfloat16" in dtype_str else "fp16" if "float16" in dtype_str else "no",
            "num_machines": 1,
            "num_processes": num_gpus,
            "machine_rank": 0,
            "main_process_ip": None,
            "main_process_port": None,
            "main_training_function": "main",
            "use_cpu": False,
        }

        if "fsdp" in strat:
            config["fsdp_config"] = {
                "fsdp_version": 2 if "2" in strat else 1,
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_forward_prefetch": True,
                "fsdp_state_dict_type": "FULL_STATE_DICT",
                "fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_sync_module_states": True,
                "fsdp_use_orig_params": True,
            }
        elif "deepspeed" in strat or "zero" in strat:
            config["deepspeed_config"] = {
                "deepspeed_config_file": "deepspeed_config.json",
                "zero3_init_flag": "zero3" in strat,
            }

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Generated Accelerate distributed configuration at: {out_p}")

        return config

    @classmethod
    def generate_deepspeed_config(
        cls,
        hardware: HardwareConfig,
        hyperparams: HyperparametersConfig,
        zero_stage: int = 3,
        cpu_offload: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Generates standard DeepSpeed JSON configuration.
        """
        dtype_str = hardware.compute_dtype.value if hasattr(hardware.compute_dtype, "value") else str(hardware.compute_dtype)
        is_bf16 = "bfloat16" in dtype_str
        is_fp16 = "float16" in dtype_str

        zero_config: Dict[str, Any] = {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
        }

        if zero_stage == 3:
            zero_config["stage3_prefetch_bucket_size"] = 2e8
            zero_config["stage3_param_persistence_threshold"] = 1e5
            zero_config["stage3_max_live_parameters"] = 1e9
            zero_config["stage3_max_reuse_distance"] = 1e9
            zero_config["stage3_gather_16bit_weights_on_model_save"] = True

        if cpu_offload:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            if zero_stage == 3:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        config: Dict[str, Any] = {
            "train_micro_batch_size_per_gpu": hyperparams.per_device_batch_size,
            "gradient_accumulation_steps": hyperparams.gradient_accumulation_steps,
            "steps_per_print": 10,
            "gradient_clipping": 1.0,
            "zero_optimization": zero_config,
            "bf16": {"enabled": is_bf16},
            "fp16": {"enabled": is_fp16 and not is_bf16},
            "wall_clock_breakdown": False,
        }

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Generated DeepSpeed configuration at: {out_p}")

        return config
