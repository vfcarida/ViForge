"""
ViForge System Diagnostics & Environment Inspector (`viforge doctor`).
"""

import sys
import os
import shutil
from typing import Dict, Any
import psutil


class SystemDoctor:
    """Inspects environment, GPU availability, CPU, RAM, disk space, and dependencies."""

    @classmethod
    def diagnose(cls) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "python_version": sys.version.split()[0],
            "os_platform": sys.platform,
            "cpu_count_physical": psutil.cpu_count(logical=False) or psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        }

        # Disk space
        total_d, used_d, free_d = shutil.disk_usage(os.getcwd())
        report["disk_total_gb"] = round(total_d / (1024**3), 2)
        report["disk_free_gb"] = round(free_d / (1024**3), 2)

        # PyTorch & CUDA check
        try:
            import torch
            report["pytorch_version"] = torch.__version__
            report["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                report["cuda_device_count"] = torch.cuda.device_count()
                report["cuda_devices"] = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    }
                    for i in range(torch.cuda.device_count())
                ]
                report["bfloat16_supported"] = torch.cuda.is_bf16_supported()
            else:
                report["cuda_device_count"] = 0
                report["cuda_devices"] = []
                report["bfloat16_supported"] = False
        except ImportError:
            report["pytorch_version"] = "Not Installed"
            report["cuda_available"] = False

        # Key packages
        for pkg in ["transformers", "peft", "trl", "bitsandbytes", "vllm", "boto3"]:
            try:
                mod = __import__(pkg)
                report[f"{pkg}_version"] = getattr(mod, "__version__", "Installed")
            except ImportError:
                report[f"{pkg}_version"] = "Not Installed"

        return report
