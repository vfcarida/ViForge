"""
ViForge Training Duration, FLOPs, and Cost Estimator.
"""

from typing import Any, Dict
from viforge.config.schemas import ModelConfig, HyperparametersConfig, HardwareConfig, CostTrackingConfig
from viforge.training.profiler import ResourceProfiler


class TrainingCostEstimator:
    """Estimates computational FLOPs, training wall-clock hours, and AWS dollar cost."""

    HARDWARE_TFLOPS = {
        "NVIDIA_A100_80GB": 312.0,
        "NVIDIA_A100_40GB": 312.0,
        "NVIDIA_H100_80GB": 989.0,
        "NVIDIA_A10G": 125.0,
        "NVIDIA_RTX_4090": 165.0,
        "NVIDIA_L40S": 366.0,
    }

    MODEL_FLOP_UTILIZATION: float = 0.42

    @classmethod
    def estimate_cost(
        cls,
        model_config: ModelConfig,
        hyperparams: HyperparametersConfig,
        hardware: HardwareConfig,
        cost_config: CostTrackingConfig,
        total_tokens: int,
    ) -> Dict[str, Any]:
        total_params, trainable_params, _ = ResourceProfiler.estimate_trainable_parameters(model_config, hyperparams)

        if trainable_params == total_params:
            total_flops = 6.0 * total_params * total_tokens * hyperparams.num_epochs
        else:
            flops_forward = 2.0 * total_params * total_tokens * hyperparams.num_epochs
            flops_backward = 4.0 * trainable_params * total_tokens * hyperparams.num_epochs
            total_flops = flops_forward + flops_backward

        hw_peak_tflops = cls.HARDWARE_TFLOPS.get(hardware.accelerator, 300.0)
        effective_tflops_per_gpu = hw_peak_tflops * cls.MODEL_FLOP_UTILIZATION
        total_cluster_tflops = effective_tflops_per_gpu * hardware.num_gpus

        cluster_flops_per_sec = total_cluster_tflops * 1e12
        estimated_duration_seconds = max(60.0, total_flops / cluster_flops_per_sec)
        estimated_duration_hours = estimated_duration_seconds / 3600.0

        estimated_tokens_per_sec = (total_tokens * hyperparams.num_epochs) / estimated_duration_seconds
        hourly_rate = cost_config.sagemaker_hourly_rate_usd or cost_config.ec2_hourly_rate_usd
        estimated_cost_usd = estimated_duration_hours * hourly_rate

        return {
            "total_tokens_with_epochs": total_tokens * hyperparams.num_epochs,
            "total_petaflops": round(total_flops / 1e15, 3),
            "estimated_tokens_per_second": round(estimated_tokens_per_sec, 1),
            "estimated_duration_seconds": round(estimated_duration_seconds, 1),
            "estimated_duration_hours": round(estimated_duration_hours, 3),
            "hourly_rate_usd": round(hourly_rate, 2),
            "estimated_cost_usd": round(estimated_cost_usd, 2),
        }
